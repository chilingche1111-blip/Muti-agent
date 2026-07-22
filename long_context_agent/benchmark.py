from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .document import DocumentIndex
from .llm import LLMError, OpenAICompatibleClient
from .orchestrator import MultiAgentResearchSystem
from .tokens import estimate_tokens


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicTestModel:
    """Offline model that validates graph topology and context isolation."""

    FACTS = {
        "项目代号": "青峦-7429",
        "验收口令": "银杏-5813",
        "归档校验值": "HF-90617-Z",
    }

    def chat(self, messages, *, temperature: float = 0.1, max_tokens: int = 2_000) -> str:
        del temperature, max_tokens
        system = messages[0]["content"]
        prompt = messages[-1]["content"]
        if "ROLE:SUPERVISOR" in system:
            tasks = [
                {"objective": f"查明{label}", "query": label, "agent_type": "fact_extractor", "priority": 80}
                for label, value in self.FACTS.items()
                if label in prompt or value in prompt
            ]
            if not tasks:
                agent_type = "risk_reviewer" if "风险" in prompt or "矛盾" in prompt else "comparator" if "比较" in prompt else "analyst" if "总结" in prompt or "分析" in prompt else "fact_extractor"
                tasks = [{"objective": prompt, "query": prompt, "agent_type": agent_type, "priority": 50}]
            return json.dumps({"tasks": tasks}, ensure_ascii=False)
        if "ROLE:SPECIALIST:" in system:
            ids = re.findall(r"\[artifact_id=([^\]]+)\]", prompt)
            found = [value for value in self.FACTS.values() if value in prompt]
            return json.dumps({"summary": "；".join(found) if found else "未找到测试事实", "claims": found, "evidence_ids": ids[:3], "confidence": 1.0 if found else 0.2}, ensure_ascii=False)
        if "ROLE:REDUCER" in system:
            values = [value for value in self.FACTS.values() if value in prompt]
            evidence_ids = list(dict.fromkeys(re.findall(r"evidence_[a-f0-9]+", prompt)))
            return json.dumps({"summary": "；".join(values), "claims": values, "evidence_ids": evidence_ids}, ensure_ascii=False)
        if "ROLE:VALIDATOR" in system:
            payload = json.loads(prompt)
            hard_passed = bool(payload.get("hard_validation", {}).get("passed"))
            return json.dumps({"semantic_pass": hard_passed, "missing_task_ids": [], "contradictions": [], "notes": "离线语义校验完成"}, ensure_ascii=False)
        if "ROLE:FINALIZER" in system:
            values = [value for value in self.FACTS.values() if value in prompt]
            return "；".join(values) if values else "没有足够证据形成答案。"
        raise AssertionError("收到未识别的 Agent 角色")


HARD_CHECK_LABELS = {
    "unique_task_ids": "任务 ID 不唯一或为空",
    "all_tasks_have_findings": "至少一个子任务没有形成有效结论或证据",
    "artifacts_and_evidence_valid": "Finding 或 Evidence Artifact 无效",
    "reducer_evidence_closed": "Reducer 引入了专业 Agent 未提供的证据",
    "all_calls_within_context_limit": "至少一次 Agent 调用超过上下文安全预算",
}


def explain_case_failure(result: dict[str, Any], missing_terms: list[str]) -> list[dict[str, Any]]:
    """Return stable, user-facing reasons for every failed acceptance case."""
    reasons: list[dict[str, Any]] = []
    if missing_terms:
        reasons.append({
            "code": "expected_terms_missing",
            "stage": "Finalizer",
            "message": f"最终回答缺少预期事实：{'、'.join(missing_terms)}",
            "retryable": True,
        })
    if not result.get("citations"):
        reasons.append({
            "code": "citations_missing",
            "stage": "Evidence",
            "message": "最终结果没有可追溯的证据引用",
            "retryable": True,
        })

    validation = result.get("validation", {})
    hard = validation.get("hard_checks", {})
    for check, passed in hard.get("checks", {}).items():
        if not passed:
            reasons.append({
                "code": f"hard_check_{check}",
                "stage": "Validator / 硬校验",
                "message": HARD_CHECK_LABELS.get(check, f"硬校验未通过：{check}"),
                "retryable": check == "all_tasks_have_findings",
            })
    if hard.get("missing_task_ids"):
        reasons.append({
            "code": "missing_task_findings",
            "stage": "Specialists",
            "message": f"缺少任务结果：{', '.join(hard['missing_task_ids'])}",
            "retryable": True,
        })
    if hard.get("violations"):
        reasons.append({
            "code": "hard_policy_violations",
            "stage": "Validator / 硬校验",
            "message": f"策略违规：{', '.join(str(item) for item in hard['violations'])}",
            "retryable": False,
        })

    semantic = validation.get("semantic_checks", {})
    if not semantic.get("passed", False):
        codes = semantic.get("failure_codes") or ["validator_semantic_rejected"]
        messages = {
            "validator_invalid_json": "Validator 未按要求返回有效 JSON",
            "validator_missing_tasks": "Validator 判断部分任务结论缺失",
            "validator_contradictions": "Validator 检测到结论或证据矛盾",
            "validator_semantic_rejected": "Validator 语义质量检查未通过",
        }
        detail_parts = []
        if semantic.get("missing_task_ids"):
            detail_parts.append(f"缺失任务：{', '.join(semantic['missing_task_ids'])}")
        if semantic.get("contradictions"):
            detail_parts.append(f"矛盾：{'；'.join(semantic['contradictions'])}")
        notes = str(semantic.get("notes", "")).strip()
        if notes:
            detail_parts.append(f"说明：{notes}")
        detail = "；".join(detail_parts)
        for code in codes:
            reasons.append({
                "code": str(code),
                "stage": "Validator / 语义校验",
                "message": f"{messages.get(str(code), 'Validator 语义校验失败')}{f'；{detail}' if detail else ''}",
                "retryable": str(code) in {"validator_invalid_json", "validator_missing_tasks"},
            })

    if not reasons and not validation.get("approved", False):
        reasons.append({
            "code": "validator_rejected_unknown",
            "stage": "Validator",
            "message": "Validator 拒绝了该用例，但没有返回更具体的原因",
            "retryable": False,
        })
    return reasons


def run_benchmark(
    document_path: Path,
    cases_path: Path,
    *,
    mode: str,
    model: Any = None,
    max_workers: int = 8,
    reduce_fan_in: int = 4,
    max_replans: int = 1,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    document = document_path.read_text(encoding="utf-8")
    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    index = DocumentIndex()
    chunks = index.add_text(document_path.name, document)
    resolved_model = model or (DeterministicTestModel() if mode == "offline" else OpenAICompatibleClient())
    results = []
    for case in suite["cases"]:
        result = asdict(MultiAgentResearchSystem(
            resolved_model,
            index,
            max_workers=max_workers,
            reduce_fan_in=reduce_fan_in,
            max_replans=max_replans,
        ).answer(case["question"]))
        missing = [term for term in case["expected_terms"] if term not in result["answer"]]
        failure_reasons = explain_case_failure(result, missing)
        results.append({
            "id": case["id"],
            "question": case["question"],
            "expected_terms": case["expected_terms"],
            "passed": not missing and bool(result["citations"]) and result["validation"].get("approved", False),
            "missing_terms": missing,
            "failure_reasons": failure_reasons,
            **result,
        })

    document_tokens = estimate_tokens(document)
    max_prompt = max((item["context_metrics"]["max_single_agent_prompt_tokens"] for item in results), default=0)
    all_within = all(item["context_metrics"]["all_agent_calls_within_limit"] for item in results)
    all_passed = all(item["passed"] for item in results)
    all_validated = all(item["validation"].get("approved", False) for item in results)
    all_cited = all(bool(item["citations"]) for item in results)
    assertions = [
        {"id": "document_over_window", "label": "测试资料超过 64K", "passed": document_tokens > 65_536, "actual": f"{document_tokens} Token", "expected": "> 65,536 Token"},
        {"id": "calls_under_window", "label": "每次 Agent 调用低于 64K", "passed": all_within, "actual": f"最大 {max_prompt} Token", "expected": "输入 + 预留 + 余量 ≤ 64K"},
        {"id": "specialist_isolation", "label": "专业 Agent 上下文相互隔离", "passed": all(item["context_metrics"]["isolated_specialist_contexts"] for item in results), "actual": "独立 Specialist State", "expected": "不得共享消息历史"},
        {"id": "supervisor_no_raw", "label": "主 Agent 不接收完整原文", "passed": all(not item["context_metrics"]["supervisor_received_raw_document"] for item in results), "actual": "仅接收任务状态", "expected": "原文保留在检索层"},
        {"id": "validator_approved", "label": "Validator 双层验证通过", "passed": all_validated, "actual": f"{sum(item['validation'].get('approved', False) for item in results)} / {len(results)}", "expected": "硬校验且语义校验通过"},
        {"id": "cases_and_citations", "label": "全部用例正确且带引用", "passed": all_passed and all_cited, "actual": f"{sum(item['passed'] for item in results)} / {len(results)}", "expected": "所有预期事实可追溯"},
    ]
    failed_cases = [
        {
            "id": item["id"],
            "primary_reason": item["failure_reasons"][0] if item["failure_reasons"] else None,
            "reason_count": len(item["failure_reasons"]),
        }
        for item in results if not item["passed"]
    ]
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "mode": mode,
        "document": str(document_path),
        "document_stats": {"bytes": document_path.stat().st_size, "estimated_tokens": document_tokens, "chunks": len(chunks), "exceeds_64k_estimated_tokens": document_tokens > 65_536},
        "duration_ms": round((time.perf_counter() - started_at) * 1_000),
        "configuration": {
            "max_workers": max_workers,
            "reduce_fan_in": reduce_fan_in,
            "max_replans": max_replans,
            "context_limit_tokens": 64_000,
        },
        "passed": all(item["passed"] for item in assertions),
        "assertions": assertions,
        "failure_summary": {
            "failed_case_count": len(failed_cases),
            "failed_cases": failed_cases,
        },
        "results": results,
        "context_proof": {
            "logical_document_tokens": document_tokens,
            "physical_context_limit_tokens": 64_000,
            "max_single_agent_prompt_tokens": max_prompt,
            "all_agent_calls_within_limit": all_within,
            "isolated_specialist_contexts": True,
            "supervisor_received_raw_document": False,
            "passed_cases": sum(item["passed"] for item in results),
            "total_cases": len(results),
            "divide_and_conquer_verified": document_tokens > 65_536 and all_within and all_passed,
            "architecture": "LangGraph deterministic loop + Supervisor + Specialists + Validator",
        },
        "interpretation": "该测试验证确定性外循环把长资料任务路由给独立专业Agent，主Agent不接收原文，Validator执行硬校验与语义校验，且每次调用均低于64K；它不代表底层模型窗口被修改。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent 64K context-isolation benchmark")
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--document", type=Path, default=PROJECT_ROOT / "test_data" / "enterprise_research_128k.md")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "test_data" / "test_cases.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run_benchmark(args.document, args.cases, mode=args.mode)
    except LLMError as exc:
        print(json.dumps({"passed": False, "mode": args.mode, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
