from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import threading
import time
from dataclasses import asdict
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .benchmark import DeterministicTestModel, run_benchmark
from .document import DocumentIndex
from .file_parser import MAX_FILE_BYTES, ParsedDocument, parse_uploaded_document
from .llm import LLMError, LLMSettings, OpenAICompatibleClient
from .orchestrator import DEFAULT_AGENT_COUNT, SOURCE_SHARD_BYTE_LIMIT, MultiAgentResearchSystem
from .tokens import estimate_messages_tokens, estimate_tokens
from .web_search import WebSearchResult, search_web


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
TEST_DOCUMENT = PROJECT_ROOT / "test_data" / "enterprise_research_128k.md"
TEST_CASES = PROJECT_ROOT / "test_data" / "test_cases.json"
MAX_REQUEST_BYTES = 30 * 1024 * 1024
SESSION_COOKIE = "context_atlas_session"
DIRECT_READ_PAGE_CHARACTERS = 12_000
GENERAL_CHAT_INPUT_BUDGET = 24_000
PAGE_RANGE_PATTERN = re.compile(
    r"(?:第\s*)?(\d{1,4})\s*(?:[-—–~～]|至|到)\s*(\d{1,4})\s*页|第\s*(\d{1,4})\s*页"
)
PAGE_HEADING_PATTERN = re.compile(r"(?m)^##\s+第\s+(\d+)\s+页\s*$")


class AuthManager:
    """Small local RBAC boundary that can later be replaced by an OIDC adapter."""

    def __init__(self, username: str, password: str, *, session_ttl: int = 28_800) -> None:
        if not username or not password:
            raise ValueError("测试账号和密码不能为空")
        self.username = username
        self.session_ttl = session_ttl
        self._salt = secrets.token_bytes(16)
        self._password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), self._salt, 240_000
        )
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def verify_password(self, username: str, password: str) -> bool:
        if not hmac.compare_digest(username.encode("utf-8"), self.username.encode("utf-8")):
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), self._salt, 240_000
        )
        return hmac.compare_digest(candidate, self._password_hash)

    def login(self, username: str, password: str) -> tuple[str, dict[str, Any]] | None:
        if not self.verify_password(username, password):
            return None
        token = secrets.token_urlsafe(32)
        session = {
            "username": self.username,
            "role": "tester",
            "expires_at": int(time.time()) + self.session_ttl,
        }
        with self._lock:
            self._sessions[token] = session
        return token, session.copy()

    def resolve(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            if int(session["expires_at"]) <= int(time.time()):
                self._sessions.pop(token, None)
                return None
            return session.copy()

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)


TEST_USERNAME = os.getenv("CONTEXT_ATLAS_TEST_USER", "tester").strip() or "tester"
_configured_password = os.getenv("CONTEXT_ATLAS_TEST_PASSWORD", "").strip()
DEFAULT_TEST_PASSWORD = "123456"
AUTH = AuthManager(TEST_USERNAME, _configured_password or DEFAULT_TEST_PASSWORD)


class VerifiedAPIProfiles:
    """Keeps already-tested API settings in server memory behind random handles."""

    def __init__(self, *, ttl_seconds: int = 28_800) -> None:
        self.ttl_seconds = ttl_seconds
        self._profiles: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register(self, settings: LLMSettings) -> dict[str, Any]:
        token = secrets.token_urlsafe(32)
        verified_at = int(time.time())
        profile = {
            "settings": settings,
            "verified_at": verified_at,
            "expires_at": verified_at + self.ttl_seconds,
        }
        with self._lock:
            self._profiles[token] = profile
        return {
            "profile_token": token,
            "base_url": settings.base_url,
            "model": settings.model,
            "timeout_seconds": settings.timeout_seconds,
            "verified_at": profile["verified_at"],
            "expires_at": profile["expires_at"],
        }

    def resolve(self, token: str) -> LLMSettings | None:
        if not token:
            return None
        with self._lock:
            profile = self._profiles.get(token)
            if profile is None:
                return None
            if int(profile["expires_at"]) <= int(time.time()):
                self._profiles.pop(token, None)
                return None
            return profile["settings"]


VERIFIED_API_PROFILES = VerifiedAPIProfiles()


def settings_from_payload(payload: dict[str, Any]) -> LLMSettings:
    base_url = str(payload.get("base_url", "")).strip().rstrip("/")
    api_key = str(payload.get("api_key", "")).strip()
    model = str(payload.get("model", "")).strip()
    if not base_url or not api_key or not model:
        raise ValueError("请完整填写 Base URL、API Key 和模型名。")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("Base URL 必须以 http:// 或 https:// 开头。")
    timeout = float(payload.get("timeout_seconds", 90))
    if timeout < 5 or timeout > 600:
        raise ValueError("超时时间必须在 5至600 秒之间。")
    return LLMSettings(base_url=base_url, api_key=api_key, model=model, timeout_seconds=timeout)


class ResearchApplication:
    def __init__(self) -> None:
        self.index = DocumentIndex()
        self.document: dict[str, Any] | None = None
        self.document_text = ""
        self.lock = threading.RLock()

    def load_text(self, name: str, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("文档内容为空。")
        parsed = ParsedDocument(
            name=Path(name).name or "document.txt",
            text=text,
            source_format=Path(name).suffix.lower().lstrip(".") or "text",
            metadata={},
        )
        return self._load_parsed(parsed, source_bytes=len(text.encode("utf-8")))

    def load_upload(self, name: str, data: bytes) -> dict[str, Any]:
        parsed = parse_uploaded_document(name, data)
        return self._load_parsed(parsed, source_bytes=len(data))

    def _load_parsed(self, parsed: ParsedDocument, *, source_bytes: int) -> dict[str, Any]:
        new_index = DocumentIndex()
        chunks = new_index.add_text(parsed.name, parsed.text, source_bytes=source_bytes)
        indexed_bytes = len(parsed.text.encode("utf-8"))
        document = {
            "name": parsed.name,
            "bytes": source_bytes,
            "indexed_bytes": indexed_bytes,
            "extracted_characters": len(parsed.text),
            "estimated_tokens": estimate_tokens(parsed.text),
            "chunks": len(chunks),
            "sections": new_index.hierarchy_stats["parent_sections"],
            "architecture": "LangGraph deterministic loop + Supervisor + Specialists + Validator",
            "source_format": parsed.source_format,
            "parser_metadata": parsed.metadata,
            "exceeds_64kb_bytes": source_bytes > 65_536,
            "exceeds_64k_tokens": estimate_tokens(parsed.text) > 65_536,
            "exceeds_shard_byte_limit": indexed_bytes > SOURCE_SHARD_BYTE_LIMIT,
            "shard_byte_limit": SOURCE_SHARD_BYTE_LIMIT,
        }
        with self.lock:
            self.index = new_index
            self.document = document
            self.document_text = parsed.text
        return document

    @staticmethod
    def _is_direct_read_request(question: str, payload: dict[str, Any]) -> bool:
        if payload.get("document_read") is True:
            return True
        compact = re.sub(r"\s+", "", question.casefold())
        analysis_terms = ("总结", "概括", "分析", "评价", "比较", "风险", "结论", "解释", "为什么")
        if any(term in compact for term in analysis_terms):
            return False
        explicit_terms = (
            "输出pdf内容", "显示pdf内容", "读取pdf内容", "阅读pdf内容",
            "输出全文", "显示全文", "读取全文", "查看全文",
            "输出原文", "显示原文", "查看原文", "逐页输出", "全部内容",
        )
        return any(term in compact for term in explicit_terms)

    @staticmethod
    def _requested_page_range(question: str) -> tuple[int, int] | None:
        match = PAGE_RANGE_PATTERN.search(question)
        if not match:
            return None
        if match.group(3):
            start = end = int(match.group(3))
        else:
            start, end = int(match.group(1)), int(match.group(2))
        if start < 1 or end < start or end - start > 100:
            raise ValueError("页码范围无效；一次最多读取连续 101 页。")
        return start, end

    @staticmethod
    def _is_page_content_request(question: str) -> bool:
        compact = re.sub(r"\s+", "", question.casefold())
        actions = ("输出", "显示", "读取", "查看", "原文", "内容", "提取")
        analysis = ("总结", "概括", "分析", "比较", "评价", "为什么", "结论", "风险")
        return any(term in compact for term in actions) and not any(term in compact for term in analysis)

    def _extract_pages(self, start_page: int, end_page: int) -> tuple[str, int, int]:
        with self.lock:
            text = self.document_text
        matches = list(PAGE_HEADING_PATTERN.finditer(text))
        pages: dict[int, tuple[int, int, str]] = {}
        for index, match in enumerate(matches):
            page = int(match.group(1))
            block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            pages[page] = (match.start(), block_end, text[match.start():block_end].strip())
        missing = [page for page in range(start_page, end_page + 1) if page not in pages]
        if missing:
            available = sorted(pages)
            range_label = f"{available[0]}–{available[-1]}" if available else "无"
            raise ValueError(
                f"解析文本中缺少第 {missing[0]} 页"
                f"{'等页面' if len(missing) > 1 else ''}；当前可定位页码范围：{range_label}。"
            )
        selected = [pages[page] for page in range(start_page, end_page + 1)]
        return "\n\n".join(item[2] for item in selected), selected[0][0], selected[-1][1]

    def _read_page_range(
        self,
        question: str,
        page_range: tuple[int, int],
    ) -> dict[str, Any]:
        start_page, end_page = page_range
        content, start_character, end_character = self._extract_pages(start_page, end_page)
        result = self._read_document({"read_offset": start_character}, question)
        label = f"第 {start_page} 页" if start_page == end_page else f"第 {start_page}–{end_page} 页"
        result["answer"] = content
        result["citations"] = [{
            "artifact_id": "direct_page_text",
            "chunk_id": f"pages_{start_page}_{end_page}",
            "document": result["document_read"]["document_name"],
            "section": label,
            "excerpt": content[:500],
            "document_bytes": int((self.document or {}).get("bytes", 0)),
            "indexed_bytes": int((self.document or {}).get("indexed_bytes", len(self.document_text.encode("utf-8")))),
            "chunk_bytes": len(content.encode("utf-8")),
            "source_exceeds_shard_limit": bool((self.document or {}).get("exceeds_shard_byte_limit", False)),
            "shard_byte_limit": int((self.document or {}).get("shard_byte_limit", SOURCE_SHARD_BYTE_LIMIT)),
        }]
        result["trace"][0]["detail"] = f"按页标记精确返回{label}，未调用 LLM 或语义检索"
        result["stop_reason"] = "direct_page_read"
        result["document_read"].update({
            "start_character": start_character,
            "end_character": end_character,
            "has_previous": False,
            "has_more": False,
            "requested_start_page": start_page,
            "requested_end_page": end_page,
        })
        return result

    @staticmethod
    def _looks_like_general_chat(question: str) -> bool:
        compact = re.sub(r"\s+", "", question.casefold())
        general_starts = (
            "你好", "您好", "你是谁", "你能做什么", "谢谢", "帮我写", "写一段",
            "翻译", "润色", "改写", "生成代码", "写代码", "解释一下",
        )
        return compact.startswith(general_starts)

    @staticmethod
    def _bounded_conversation_context(payload: dict[str, Any]) -> str:
        """Keep recent dialogue outside the document index and inject only a bounded view."""
        lines: list[str] = []
        used = 0
        raw_history = payload.get("history", [])
        if not isinstance(raw_history, list):
            return ""
        for item in reversed(raw_history[-20:]):
            if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
                continue
            role = "用户" if item.get("role") == "user" else "助手"
            limit = 1_200 if role == "用户" else 2_000
            content = re.sub(r"\s+", " ", str(item.get("content", "")).strip())[:limit]
            if not content:
                continue
            line = f"{role}：{content}"
            if used + len(line) > 6_000:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(reversed(lines))

    def _general_chat(
        self,
        payload: dict[str, Any],
        question: str,
        model: OpenAICompatibleClient,
        web_results: list[WebSearchResult] | None = None,
    ) -> dict[str, Any]:
        history: list[dict[str, str]] = []
        raw_history = payload.get("history", [])
        if not isinstance(raw_history, list):
            raw_history = []
        for item in raw_history[-20:]:
            if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
                continue
            content = str(item.get("content", "")).strip()[:8_000]
            if content:
                history.append({"role": str(item["role"]), "content": content})
        web_results = web_results or []
        system = {
            "role": "system",
            "content": (
                "你是企业智能助手。保留底层大语言模型的通用问答、写作、解释和推理能力。"
                "直接回答用户问题；不知道或需要实时信息时明确说明，不得伪造文档引用。"
                "如果提供了联网检索结果，优先用这些结果回答，并使用[网页1]格式标注来源；"
                "搜索摘要可能不完整，不要把摘要中没有的信息说成确定事实。"
            ),
        }
        source_message: list[dict[str, str]] = []
        if web_results:
            packed_sources = "\n\n".join(
                f"[网页{index}] {item.title}\nURL: {item.url}\n摘要: {item.snippet or '无摘要'}"
                for index, item in enumerate(web_results, start=1)
            )
            source_message = [{
                "role": "user",
                "content": f"以下是本次实时联网检索结果：\n\n{packed_sources}",
            }]
        messages = [system, *history, *source_message, {"role": "user", "content": question}]
        while len(messages) > 2 and estimate_messages_tokens(messages) > GENERAL_CHAT_INPUT_BUDGET:
            messages.pop(1)
        prompt_tokens = estimate_messages_tokens(messages)
        answer = model.chat(messages, temperature=0.3, max_tokens=3_000).strip()
        usage = getattr(model, "last_usage", None)
        measured = int(usage.get("prompt_tokens") or prompt_tokens) if isinstance(usage, dict) else prompt_tokens
        return {
            "answer": answer,
            "citations": [
                {
                    "artifact_id": f"web_search_{index:02d}",
                    "chunk_id": f"web_{index:02d}",
                    "document": item.title,
                    "section": "联网检索",
                    "excerpt": item.snippet,
                    "url": item.url,
                    "source_type": "web",
                    "document_bytes": len((item.snippet or "").encode("utf-8")),
                    "indexed_bytes": len((item.snippet or "").encode("utf-8")),
                    "chunk_bytes": len((item.snippet or "").encode("utf-8")),
                    "source_exceeds_shard_limit": False,
                    "shard_byte_limit": SOURCE_SHARD_BYTE_LIMIT,
                }
                for index, item in enumerate(web_results, start=1)
            ],
            "tasks": [{
                "task_id": "direct_chat_01",
                "objective": question,
                "query": question,
                "agent_type": "general_llm",
                "priority": 100,
                "input_budget": GENERAL_CHAT_INPUT_BUDGET,
            }],
            "findings": [],
            "trace": ([{
                "node": "web_search",
                "role": "联网检索器",
                "status": "completed",
                "detail": f"检索并返回 {len(web_results)} 个外部来源",
            }] if web_results else []) + [{
                "node": "general_assistant",
                "role": "原始 LLM 助手",
                "status": "completed",
                "detail": (
                    "结合联网来源直接回答，并携带预算内的最近对话"
                    if web_results else
                    "通用问题直接调用模型，并携带预算内的最近对话；未启用文档证据门禁"
                ),
            }],
            "validation": {
                "approved": True,
                "decision_source": "direct_llm_response",
                "hard_checks": {"passed": True, "checks": {"within_context_budget": measured <= GENERAL_CHAT_INPUT_BUDGET}},
                "semantic_checks": {"passed": True, "notes": "通用聊天不强制要求文档引用。"},
            },
            "context_metrics": {
                "architecture": "Direct LLM chat with bounded conversation memory",
                "control_plane": "direct_chat_router",
                "context_limit_tokens": 64_000,
                "hard_safe_input_tokens": 61_000,
                "max_single_agent_prompt_tokens": measured,
                "max_window_utilization_percent": round(measured / 64_000 * 100, 2),
                "all_agent_calls_within_limit": measured + 5_000 <= 64_000,
                "isolated_specialist_contexts": True,
                "supervisor_received_raw_document": False,
                "task_count": 1,
                "specialist_counts": {"general_llm": 1},
                "model_calls": 1,
                "planning_source": "direct_chat_router",
                "intent": "general_chat",
                "structured_output_retries": 0,
                "retrieval_strategies": [],
                "web_search_enabled": bool(web_results),
                "web_source_count": len(web_results),
                "conversation_memory": {
                    "enabled": bool(history),
                    "messages": len(history),
                    "estimated_tokens": estimate_messages_tokens(history) if history else 0,
                    "stored_outside_model_window": True,
                },
                "by_role": {"general_llm": {"calls": 1, "max_prompt_tokens": measured}},
                "calls": [],
            },
            "stop_reason": "direct_llm_answer",
            "execution_mode": "general_chat",
            "web_sources": [item.as_dict() for item in web_results],
            "capacity_report": {
                "document_tokens_estimate": int((self.document or {}).get("estimated_tokens", 0)),
                "document_exceeds_64k": bool((self.document or {}).get("exceeds_64k_tokens", False)),
                "max_single_agent_prompt_tokens": measured,
                "context_limit_tokens": 64_000,
                "max_window_utilization_percent": round(measured / 64_000 * 100, 2),
                "all_agent_calls_within_limit": measured + 5_000 <= 64_000,
                "isolated_specialist_contexts": True,
                "supervisor_received_raw_document": False,
                "task_count": 1,
                "specialist_counts": {"general_llm": 1},
                "control_plane": "direct_chat_router",
                "model_calls": 1,
                "divide_and_conquer_verified": False,
                "architecture": "Direct LLM chat with bounded conversation memory",
                "default_agents": 1,
                "desired_agents": 1,
                "allocated_agents": 1,
                "max_agents": 1,
                "source_indexed_bytes": 0,
                "largest_source_indexed_bytes": 0,
                "shard_byte_limit": SOURCE_SHARD_BYTE_LIMIT,
                "source_exceeds_shard_limit": False,
                "multi_agent_sharding_active": False,
            },
        }

    def _read_document(self, payload: dict[str, Any], question: str) -> dict[str, Any]:
        with self.lock:
            text = self.document_text
            document = dict(self.document or {})
        if not text:
            raise ValueError("当前文档没有可读取的文本内容。")
        try:
            requested_offset = int(payload.get("read_offset", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("原文读取位置必须是整数。") from exc
        start = max(0, min(requested_offset, max(0, len(text) - 1)))
        end = min(len(text), start + DIRECT_READ_PAGE_CHARACTERS)
        if end < len(text):
            newline = text.rfind("\n", start + DIRECT_READ_PAGE_CHARACTERS // 2, end)
            if newline > start:
                end = newline
        content = text[start:end].strip()
        previous_offset = max(0, start - DIRECT_READ_PAGE_CHARACTERS)
        direct = {
            "enabled": True,
            "document_name": document.get("name"),
            "source_format": document.get("source_format"),
            "parser_metadata": document.get("parser_metadata", {}),
            "start_character": start,
            "end_character": end,
            "total_characters": len(text),
            "page_characters": DIRECT_READ_PAGE_CHARACTERS,
            "has_previous": start > 0,
            "has_more": end < len(text),
            "previous_offset": previous_offset,
            "next_offset": end,
        }
        document_tokens = int(document.get("estimated_tokens", 0))
        capacity = {
            "document_tokens_estimate": document_tokens,
            "document_exceeds_64k": document_tokens > 65_536,
            "max_single_agent_prompt_tokens": 0,
            "context_limit_tokens": 64_000,
            "max_window_utilization_percent": 0,
            "all_agent_calls_within_limit": True,
            "isolated_specialist_contexts": True,
            "supervisor_received_raw_document": False,
            "task_count": 1,
            "specialist_counts": {"direct_document_reader": 1},
            "control_plane": "deterministic_code",
            "model_calls": 0,
            "divide_and_conquer_verified": False,
            "execution_mode": "direct_document_read",
            "architecture": "Deterministic document reader outside the LLM context window",
            "default_agents": 0,
            "desired_agents": 0,
            "allocated_agents": 0,
            "max_agents": 0,
            "source_indexed_bytes": int(document.get("indexed_bytes", len(text.encode("utf-8")))),
            "largest_source_indexed_bytes": int(document.get("indexed_bytes", len(text.encode("utf-8")))),
            "shard_byte_limit": int(document.get("shard_byte_limit", SOURCE_SHARD_BYTE_LIMIT)),
            "source_exceeds_shard_limit": bool(document.get("exceeds_shard_byte_limit", False)),
            "multi_agent_sharding_active": False,
        }
        return {
            "answer": content,
            "citations": [{
                "artifact_id": "direct_document_text",
                "chunk_id": f"characters_{start}_{end}",
                "document": document.get("name"),
                "section": f"原文字符 {start + 1}–{end}",
                "excerpt": content[:500],
                "document_bytes": int(document.get("bytes", 0)),
                "indexed_bytes": int(document.get("indexed_bytes", len(text.encode("utf-8")))),
                "chunk_bytes": len(content.encode("utf-8")),
                "source_exceeds_shard_limit": bool(document.get("exceeds_shard_byte_limit", False)),
                "shard_byte_limit": int(document.get("shard_byte_limit", SOURCE_SHARD_BYTE_LIMIT)),
            }],
            "tasks": [{
                "task_id": "direct_read_01",
                "objective": question,
                "query": "direct document content",
                "agent_type": "direct_document_reader",
                "priority": 100,
                "input_budget": 0,
            }],
            "findings": [],
            "trace": [{
                "node": "document_reader",
                "role": "确定性原文读取器",
                "status": "completed",
                "detail": f"直接返回已解析文本字符 {start + 1}–{end}，未调用 LLM",
            }],
            "validation": {
                "approved": True,
                "decision_source": "deterministic_document_reader",
                "hard_checks": {"passed": True, "checks": {"parsed_text_available": True}},
                "semantic_checks": {"passed": True, "notes": "原文直出不需要模型语义校验。"},
            },
            "context_metrics": {
                "architecture": capacity["architecture"],
                "control_plane": "deterministic_code",
                "context_limit_tokens": 64_000,
                "hard_safe_input_tokens": 61_000,
                "max_single_agent_prompt_tokens": 0,
                "max_window_utilization_percent": 0,
                "all_agent_calls_within_limit": True,
                "isolated_specialist_contexts": True,
                "supervisor_received_raw_document": False,
                "task_count": 1,
                "specialist_counts": {"direct_document_reader": 1},
                "model_calls": 0,
                "by_role": {},
                "calls": [],
            },
            "stop_reason": "direct_document_read",
            "capacity_report": capacity,
            "document_read": direct,
        }

    def load_bundled_test(self) -> dict[str, Any]:
        return self.load_text(TEST_DOCUMENT.name, TEST_DOCUMENT.read_text(encoding="utf-8"))

    def inspect_bundled_test(self) -> dict[str, Any]:
        """Describe the test corpus without replacing the business-agent index."""
        text = TEST_DOCUMENT.read_text(encoding="utf-8")
        index = DocumentIndex()
        chunks = index.add_text(TEST_DOCUMENT.name, text)
        tokens = estimate_tokens(text)
        return {
            "name": TEST_DOCUMENT.name,
            "bytes": len(text.encode("utf-8")),
            "estimated_tokens": tokens,
            "chunks": len(chunks),
            "sections": index.hierarchy_stats["parent_sections"],
            "architecture": "LangGraph deterministic loop + Supervisor + Specialists + Validator",
            "exceeds_64kb_bytes": len(text.encode("utf-8")) > 65_536,
            "exceeds_64k_tokens": tokens > 65_536,
        }

    def answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError("请输入调研问题。")
        mode = str(payload.get("mode", "offline"))
        if mode not in {"offline", "live"}:
            raise ValueError("运行模式必须是 offline 或 live。")
        scope = str(payload.get("answer_scope", "auto"))
        if scope not in {"auto", "general", "document"}:
            raise ValueError("回答范围必须是 auto、general 或 document。")
        with self.lock:
            has_document = bool(self.index.chunks)
            index = self.index
            document = dict(self.document or {})
            document_text = self.document_text
        conversation_context = self._bounded_conversation_context(payload)

        resolved_scope = scope
        if scope == "auto":
            resolved_scope = "general" if not has_document or self._looks_like_general_chat(question) else "document"
        if resolved_scope == "general":
            if mode != "live":
                raise ValueError("通用问答需要选择“智能回答”并配置真实 API；流程演示模型只用于架构验收。")
            client = OpenAICompatibleClient(settings_from_payload(payload.get("api", {})))
            web_results = search_web(question, limit=5) if payload.get("web_search") is True else []
            return self._general_chat(payload, question, client, web_results)

        if not has_document:
            raise ValueError("文档问答需要先导入 PDF、Word、PPT 或文本；也可以切换到“通用问答”。")

        requested_pages = self._requested_page_range(question)
        if requested_pages and self._is_page_content_request(question):
            return self._read_page_range(question, requested_pages)
        if self._is_direct_read_request(question, payload):
            return self._read_document(payload, question)

        if payload.get("web_search") is True and mode != "live":
            raise ValueError("联网检索需要选择“智能回答”，流程演示不会访问外部网络。")
        web_results = search_web(question, limit=5) if payload.get("web_search") is True else []

        model = (
            DeterministicTestModel()
            if mode == "offline"
            else OpenAICompatibleClient(settings_from_payload(payload.get("api", {})))
        )
        if requested_pages:
            page_text, _, _ = self._extract_pages(*requested_pages)
            page_index = DocumentIndex()
            page_index.add_text(
                f"{document.get('name', 'document')} 第{requested_pages[0]}-{requested_pages[1]}页",
                page_text,
                source_bytes=len(page_text.encode("utf-8")),
            )
            for position, item in enumerate(web_results, start=1):
                web_text = f"# 联网来源 {position}\n\n标题：{item.title}\nURL：{item.url}\n摘要：{item.snippet}"
                page_index.add_text(
                    f"联网来源 {position}：{item.title}｜{item.url}",
                    web_text,
                    source_bytes=len(web_text.encode("utf-8")),
                )
            index = page_index
        elif web_results:
            combined_index = DocumentIndex()
            combined_index.add_text(
                str(document.get("name") or "document"),
                document_text,
                source_bytes=int(document.get("bytes", len(document_text.encode("utf-8")))),
            )
            for position, item in enumerate(web_results, start=1):
                web_text = f"# 联网来源 {position}\n\n标题：{item.title}\nURL：{item.url}\n摘要：{item.snippet}"
                combined_index.add_text(
                    f"联网来源 {position}：{item.title}｜{item.url}",
                    web_text,
                    source_bytes=len(web_text.encode("utf-8")),
                )
            index = combined_index
        max_workers = int(payload.get("max_workers", 8))
        default_agents = int(payload.get("default_agents", DEFAULT_AGENT_COUNT))
        reduce_fan_in = int(payload.get("reduce_fan_in", 4))
        system = MultiAgentResearchSystem(
            model,
            index,
            max_workers=max_workers,
            default_agents=default_agents,
            reduce_fan_in=reduce_fan_in,
        )
        result = asdict(system.answer(question, conversation_context=conversation_context))
        result["execution_mode"] = "offline_demo" if mode == "offline" else "live_multi_agent"
        result["web_sources"] = [item.as_dict() for item in web_results]
        if web_results:
            result["citations"].extend({
                "artifact_id": f"web_search_{position:02d}",
                "chunk_id": f"web_{position:02d}",
                "document": item.title,
                "section": "联网检索候选来源",
                "excerpt": item.snippet,
                "url": item.url,
                "source_type": "web",
                "document_bytes": len((item.snippet or "").encode("utf-8")),
                "indexed_bytes": len((item.snippet or "").encode("utf-8")),
                "chunk_bytes": len((item.snippet or "").encode("utf-8")),
                "source_exceeds_shard_limit": False,
                "shard_byte_limit": SOURCE_SHARD_BYTE_LIMIT,
            } for position, item in enumerate(web_results, start=1))
        if requested_pages:
            result["page_scope"] = {
                "start_page": requested_pages[0],
                "end_page": requested_pages[1],
                "retrieval_restricted_to_requested_pages": True,
            }
        metrics = result["context_metrics"]
        allocation = metrics.get("agent_allocation", {})
        document_tokens = int(document.get("estimated_tokens", 0))
        result["capacity_report"] = {
            "document_tokens_estimate": document_tokens,
            "document_exceeds_64k": document_tokens > 65_536,
            "max_single_agent_prompt_tokens": metrics["max_single_agent_prompt_tokens"],
            "context_limit_tokens": metrics["context_limit_tokens"],
            "max_window_utilization_percent": metrics["max_window_utilization_percent"],
            "all_agent_calls_within_limit": metrics["all_agent_calls_within_limit"],
            "isolated_specialist_contexts": metrics["isolated_specialist_contexts"],
            "supervisor_received_raw_document": False,
            "task_count": metrics["task_count"],
            "specialist_counts": metrics["specialist_counts"],
            "control_plane": metrics["control_plane"],
            "model_calls": metrics["model_calls"],
            "divide_and_conquer_verified": (
                (
                    document_tokens > 65_536
                    or bool(allocation.get("source_exceeds_shard_limit", False))
                )
                and int(allocation.get("allocated_agents", 0)) > 1
                and metrics["all_agent_calls_within_limit"]
                and result["validation"].get("approved", False)
            ),
            "architecture": metrics["architecture"],
            "planning_source": metrics.get("planning_source"),
            "intent": metrics.get("intent"),
            "structured_output_retries": metrics.get("structured_output_retries", 0),
            "retrieval_strategies": metrics.get("retrieval_strategies", []),
            "default_agents": allocation.get("default_agents", default_agents),
            "desired_agents": allocation.get("desired_agents", metrics["task_count"]),
            "allocated_agents": allocation.get("allocated_agents", metrics["task_count"]),
            "max_agents": allocation.get("max_agents", max_workers),
            "agent_allocation_reason": allocation.get("allocation_reason", ""),
            "source_indexed_bytes": allocation.get("source_indexed_bytes", 0),
            "largest_source_indexed_bytes": allocation.get("largest_source_indexed_bytes", 0),
            "shard_byte_limit": allocation.get("shard_byte_limit", SOURCE_SHARD_BYTE_LIMIT),
            "target_shard_bytes": allocation.get("target_shard_bytes", SOURCE_SHARD_BYTE_LIMIT),
            "source_exceeds_shard_limit": allocation.get("source_exceeds_shard_limit", False),
            "multi_agent_sharding_active": allocation.get("multi_agent_sharding_active", False),
            "exhaustive_scan": allocation.get("exhaustive_scan", False),
        }
        return result

    def benchmark(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "offline"))
        if mode not in {"offline", "live"}:
            raise ValueError("测试模式必须是 offline 或 live。")
        max_workers = int(payload.get("max_workers", 8))
        reduce_fan_in = int(payload.get("reduce_fan_in", 4))
        max_replans = int(payload.get("max_replans", 1))
        model = None
        if mode == "live":
            profile_token = str(payload.get("api_profile_token", "")).strip()
            settings = VERIFIED_API_PROFILES.resolve(profile_token)
            if settings is None:
                raise ValueError("已验证 API 配置不存在或已过期，请返回研究系统重新测试连接。")
            model = OpenAICompatibleClient(settings)
        return run_benchmark(
            TEST_DOCUMENT,
            TEST_CASES,
            mode=mode,
            model=model,
            max_workers=max_workers,
            reduce_fan_in=reduce_fan_in,
            max_replans=max_replans,
        )


APPLICATION = ResearchApplication()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ContextAtlas/0.9"

    def log_message(self, format: str, *args: Any) -> None:
        # Never include request bodies or API credentials in local logs.
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")

    def _send_json(
        self,
        payload: dict[str, Any],
        status: int = 200,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _session_token(self) -> str | None:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return None
        item = cookie.get(SESSION_COOKIE)
        return item.value if item else None

    def _tester_session(self) -> dict[str, Any] | None:
        session = AUTH.resolve(self._session_token())
        return session if session and session.get("role") == "tester" else None

    def _require_tester(self) -> bool:
        if self._tester_session():
            return True
        self._send_json(
            {"ok": False, "error": "需要测试员权限，请先登录测试中心。"},
            HTTPStatus.UNAUTHORIZED,
        )
        return False

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        if content_length > MAX_REQUEST_BYTES:
            raise ValueError("请求超过 20 MB 限制。")
        data = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求必须是 JSON 对象。")
        return data

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._send_json({"document": APPLICATION.document})
            return
        if path == "/api/auth/me":
            session = self._tester_session()
            self._send_json(
                {
                    "ok": True,
                    "result": {
                        "authenticated": bool(session),
                        "user": session.get("username") if session else None,
                        "role": session.get("role") if session else None,
                    },
                }
            )
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = target.read_bytes()
        mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if mime_type.startswith("text/") else mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; font-src 'self'")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/auth/login":
                username = str(payload.get("username", ""))[:128]
                password = str(payload.get("password", ""))[:512]
                login = AUTH.login(username, password)
                if login is None:
                    self._send_json(
                        {"ok": False, "error": "用户名或密码错误。"},
                        HTTPStatus.UNAUTHORIZED,
                    )
                    return
                token, session = login
                cookie = (
                    f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; "
                    f"Max-Age={AUTH.session_ttl}"
                )
                self._send_json(
                    {"ok": True, "result": {"user": session["username"], "role": session["role"]}},
                    headers={"Set-Cookie": cookie},
                )
                return
            if path == "/api/auth/logout":
                AUTH.logout(self._session_token())
                self._send_json(
                    {"ok": True, "result": {"logged_out": True}},
                    headers={
                        "Set-Cookie": f"{SESSION_COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
                    },
                )
                return
            if path in {
                "/api/load-test-document",
                "/api/benchmark",
                "/api/test/document",
                "/api/test/benchmark",
                "/api/test/api-connection",
                "/api/test/api-profile",
            }:
                if not self._require_tester():
                    return
            if path == "/api/documents":
                name = str(payload.get("name", "document.txt"))
                encoded = payload.get("data_base64")
                if isinstance(encoded, str) and encoded:
                    try:
                        data = base64.b64decode(encoded, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise ValueError("上传文件的 Base64 数据无效。") from exc
                    if len(data) > MAX_FILE_BYTES:
                        raise ValueError("文件超过 20 MB 限制。")
                    result = APPLICATION.load_upload(name, data)
                else:
                    result = APPLICATION.load_text(name, str(payload.get("text", "")))
            elif path == "/api/load-test-document":
                result = APPLICATION.load_bundled_test()
            elif path == "/api/test/document":
                result = APPLICATION.inspect_bundled_test()
            elif path == "/api/test/api-profile":
                profile_token = str(payload.get("api_profile_token", "")).strip()
                settings = VERIFIED_API_PROFILES.resolve(profile_token)
                if settings is None:
                    raise ValueError("已验证 API 配置不存在或已过期，请返回研究系统重新测试连接。")
                result = {
                    "active": True,
                    "base_url": settings.base_url,
                    "model": settings.model,
                    "timeout_seconds": settings.timeout_seconds,
                }
            elif path in {"/api/test-connection", "/api/test/api-connection"}:
                if path == "/api/test/api-connection":
                    profile_token = str(payload.get("api_profile_token", "")).strip()
                    settings = VERIFIED_API_PROFILES.resolve(profile_token)
                    if settings is None:
                        raise ValueError("已验证 API 配置不存在或已过期，请返回研究系统重新测试连接。")
                else:
                    settings = settings_from_payload(payload.get("api", {}))
                client = OpenAICompatibleClient(settings)
                response = client.chat(
                    [
                        {"role": "system", "content": "你是 API 连接测试器。"},
                        {"role": "user", "content": "只回复 OK"},
                    ],
                    temperature=0,
                    max_tokens=16,
                )
                result = {"connected": True, "response": response[:80]}
                if path == "/api/test-connection":
                    result.update(VERIFIED_API_PROFILES.register(settings))
            elif path == "/api/ask":
                result = APPLICATION.answer(payload)
            elif path in {"/api/benchmark", "/api/test/benchmark"}:
                result = APPLICATION.benchmark(payload)
            else:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json({"ok": True, "result": result})
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except LLMError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_GATEWAY)
        except Exception as exc:  # pragma: no cover - final local-server safety net
            self._send_json(
                {"ok": False, "error": f"服务器错误：{type(exc).__name__}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Context Atlas local workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Context Atlas: http://{args.host}:{args.port}")
    print(f"测试中心: http://{args.host}:{args.port}/test.html")
    if not _configured_password:
        print(f"本地测试账号: {TEST_USERNAME}")
        print(f"本地测试密码: {DEFAULT_TEST_PASSWORD}")
        print("生产使用请设置 CONTEXT_ATLAS_TEST_USER 和 CONTEXT_ATLAS_TEST_PASSWORD")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
