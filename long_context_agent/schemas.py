from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    objective: str
    query: str
    agent_name: str = "动态任务 Agent"
    agent_instruction: str = "独立完成当前子任务并返回可验证的结构化结论。"
    capabilities: tuple[str, ...] = ("reasoning",)
    tools: tuple[str, ...] = ("model_reasoning",)
    source_policy: str = "optional"
    priority: int = 50
    input_budget: int = 61_000
    depends_on: tuple[str, ...] = ()
    phase: int = 0
    required_targets: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "query": self.query,
            "agent_name": self.agent_name,
            "agent_instruction": self.agent_instruction,
            "capabilities": list(self.capabilities),
            "tools": list(self.tools),
            "source_policy": self.source_policy,
            "priority": self.priority,
            "input_budget": self.input_budget,
            "depends_on": list(self.depends_on),
            "phase": self.phase,
            "required_targets": list(self.required_targets),
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
