from __future__ import annotations

import json
from pathlib import Path

from long_context_agent.tokens import estimate_tokens


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "test_data"
DOCUMENT_PATH = OUTPUT_DIR / "enterprise_research_128k.md"
CASES_PATH = OUTPUT_DIR / "test_cases.json"

TOTAL_SECTIONS = 1_100
MARKERS = {
    55: "【测试针 ALPHA】项目代号：青峦-7429。该代号是本次长上下文测试的唯一有效项目代号。",
    550: "【测试针 BETA】中期验收口令：银杏-5813。任何其他口令都是无效填充内容。",
    1_045: "【测试针 GAMMA】最终归档校验值：HF-90617-Z。该值用于确认文档末部信息被成功检索。",
}


def build_document() -> str:
    sections = [
        "# 企业项目调研长上下文测试文档",
        "本文档用于验证多 Agent 分治与上下文隔离。大部分内容为确定性填充记录，少量关键信息分布在开头、中部和末部。",
    ]
    for section in range(1, TOTAL_SECTIONS + 1):
        body = f"""## 第 {section:04d} 节 常规项目调研记录

本节记录编号为 R-{section:04d}，用于构造稳定、可重复的企业调研语料。常规内容包括流程责任、会议安排、风险等级、数据范围、审批状态和后续行动。

调研人员应记录来源、更新时间和责任部门，并在引用结论前回到原始材料核验。本节中的普通序号 {section:04d} 不代表业务结论。

如果资料之间存在冲突，应保留各方原文并标记冲突，不得通过摘要擅自覆盖较早记录。"""
        marker = MARKERS.get(section)
        if marker:
            body = f"{body}\n\n{marker}"
        sections.append(body)
    return "\n\n".join(sections) + "\n"


def generate(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    document_path = output_dir / DOCUMENT_PATH.name
    cases_path = output_dir / CASES_PATH.name
    document = build_document()
    document_path.write_text(document, encoding="utf-8")
    stats = {
        "document": document_path.name,
        "bytes": document_path.stat().st_size,
        "estimated_tokens": estimate_tokens(document),
        "physical_context_reference": 65_536,
        "exceeds_64kb_bytes": document_path.stat().st_size > 65_536,
        "exceeds_64k_estimated_tokens": estimate_tokens(document) > 65_536,
    }
    cases = {
        "document_stats": stats,
        "cases": [
            {
                "id": "needle_beginning",
                "question": "本次长上下文测试的唯一有效项目代号是什么？",
                "expected_terms": ["青峦-7429"],
            },
            {
                "id": "needle_middle",
                "question": "中期验收口令是什么？",
                "expected_terms": ["银杏-5813"],
            },
            {
                "id": "needle_end",
                "question": "用于确认文档末部信息的最终归档校验值是什么？",
                "expected_terms": ["HF-90617-Z"],
            },
            {
                "id": "multi_hop_all_positions",
                "question": "请同时给出唯一有效项目代号、中期验收口令和最终归档校验值。",
                "expected_terms": ["青峦-7429", "银杏-5813", "HF-90617-Z"],
            },
        ],
    }
    cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return document_path, cases_path, stats


if __name__ == "__main__":
    generated_document, generated_cases, generated_stats = generate()
    print(json.dumps({"document": str(generated_document), "cases": str(generated_cases), **generated_stats}, ensure_ascii=False, indent=2))
