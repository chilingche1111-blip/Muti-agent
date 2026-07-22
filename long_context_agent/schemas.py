from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    objective: str
    query: str
    agent_type: str = "fact_extractor"
    priority: int = 50
    input_budget: int = 12_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "query": self.query,
            "agent_type": self.agent_type,
            "priority": self.priority,
            "input_budget": self.input_budget,
        }


@dataclass(frozen=True)
class MultiAgentResult:
    answer: str
    citations: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    validation: dict[str, Any]
    context_metrics: dict[str, Any]
    stop_reason: str
