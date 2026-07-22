from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .tokens import estimate_messages_tokens


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 90.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        values = {
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "").strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise LLMError(f"缺少运行环境变量：{', '.join(missing)}")
        return cls(
            base_url=values["OPENAI_BASE_URL"],
            api_key=values["OPENAI_API_KEY"],
            model=values["OPENAI_MODEL"],
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90")),
        )


def safe_error_message(body: str, api_key: str) -> str:
    """Extract a useful upstream error without reflecting credentials."""
    message = body.strip()
    try:
        payload = json.loads(message)
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("type") or message)
            else:
                message = str(payload.get("message") or payload.get("detail") or message)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if api_key:
        message = message.replace(api_key, "[redacted]")
    message = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[redacted]", message)
    return message[:300] or "上游服务未返回错误说明"


class OpenAICompatibleClient:
    """Chat Completions client using system curl for enterprise TLS compatibility.

    The company gateway currently closes Python TLS connections during the
    handshake, while the system curl client negotiates successfully. The API key
    is supplied to curl through stdin and never appears in process arguments.
    """

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()
        self.last_usage: dict[str, int | str | None] | None = None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2_000,
    ) -> str:
        self.last_usage = None
        curl = shutil.which("curl")
        if curl is None:
            raise LLMError("系统未安装 curl，无法访问企业LLM网关")

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="context-atlas-request-",
                suffix=".json",
                delete=False,
            ) as payload_file:
                json.dump(payload, payload_file, ensure_ascii=False)
                payload_path = Path(payload_file.name)
            payload_path.chmod(0o600)

            headers = (
                f"Authorization: Bearer {self.settings.api_key}\n"
                "Content-Type: application/json\n"
                "Accept: application/json\n"
            )
            command = [
                curl,
                "--silent",
                "--show-error",
                "--location",
                "--request",
                "POST",
                "--max-time",
                str(self.settings.timeout_seconds),
                "--header",
                "@-",
                "--data-binary",
                f"@{payload_path}",
                "--write-out",
                "\n%{http_code}",
                f"{self.settings.base_url}/chat/completions",
            ]
            completed = subprocess.run(
                command,
                input=headers,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds + 5,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError("企业LLM请求超时") from exc
        except OSError as exc:
            raise LLMError(f"无法启动LLM请求：{type(exc).__name__}") from exc
        finally:
            if payload_path is not None:
                payload_path.unlink(missing_ok=True)

        if completed.returncode != 0:
            detail = completed.stderr.strip()[:240] or f"curl exit {completed.returncode}"
            raise LLMError(f"LLM连接失败：{detail}")
        try:
            response_body, status_text = completed.stdout.rsplit("\n", 1)
            status_code = int(status_text)
        except (ValueError, TypeError) as exc:
            raise LLMError("LLM响应缺少HTTP状态码") from exc
        if status_code >= 400:
            detail = safe_error_message(response_body, self.settings.api_key)
            raise LLMError(f"LLM HTTP {status_code}：{detail}")

        try:
            response = json.loads(response_body)
            content = response["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM响应不兼容Chat Completions格式") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM返回空内容")
        raw_usage = response.get("usage") if isinstance(response, dict) else None
        raw_usage = raw_usage if isinstance(raw_usage, dict) else {}

        def optional_int(value: object) -> int | None:
            if isinstance(value, bool):
                return None
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        prompt_tokens = optional_int(raw_usage.get("prompt_tokens"))
        completion_tokens = optional_int(raw_usage.get("completion_tokens"))
        total_tokens = optional_int(raw_usage.get("total_tokens"))
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_prompt_tokens": estimate_messages_tokens(messages),
            "requested_max_tokens": max_tokens,
            "source": "api" if prompt_tokens is not None else "estimate",
        }
        return content.strip()
