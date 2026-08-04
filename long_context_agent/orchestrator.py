from __future__ import annotations

import json
import math
import operator
import re
import threading
import uuid
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from .artifacts import ArtifactStore
from .document import DocumentIndex, SearchHit
from .schemas import MultiAgentResult, TaskSpec
from .tokens import estimate_messages_tokens, estimate_tokens


MODEL_CONTEXT_LIMIT = 64_000
OUTPUT_RESERVE = 2_000
SAFETY_MARGIN = 1_000
DEFAULT_AGENT_COUNT = 3
SOURCE_SHARD_BYTE_LIMIT = 64 * 1024
TARGET_CHUNKS_PER_AGENT = 128
TARGET_SOURCE_TOKENS_PER_AGENT = 45_000
EXHAUSTIVE_SCAN_TERMS = ("全部", "所有", "完整清单", "逐条", "逐项", "员工信息", "员工名单")
SOURCE_POLICIES = {"none", "optional", "required"}
BUILTIN_TOOLS = {"model_reasoning", "source_search", "web_sources"}
OUTPUT_MODES = {"finding", "artifact"}
FULL_DELIVERABLE_TERMS = (
    "完整", "全文", "全部正文", "整个", "原文", "成稿", "最终稿", "逐章", "逐节",
    "不要概括", "不要总结", "而不是概括", "直接输出",
)
DELIVERABLE_TYPES = (
    "小说", "故事", "剧本", "报告", "文章", "论文", "方案", "计划书", "说明书",
    "教程", "合同", "文案", "代码", "脚本", "邮件", "演讲稿",
)


class ChatModel(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2_000,
    ) -> str: ...


class GraphState(TypedDict, total=False):
    question: str
    tasks: list[dict[str, Any]]
    active_tasks: list[dict[str, Any]]
    findings: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    reduced: dict[str, Any]
    validation: dict[str, Any]
    answer: str
    iteration: int
    stop_reason: str
    planning_source: str
    intent: str
    agent_allocation: dict[str, Any]
    dependency_phase: int
    delivery_contract: dict[str, Any]


class DynamicWorkerState(TypedDict):
    question: str
    task: dict[str, Any]
    dependency_findings: list[dict[str, Any]]


def parse_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


class MultiAgentResearchSystem:
    """Deterministic LangGraph outer loop with runtime-generated Worker agents."""

    ROLE_BUDGETS = {
        "supervisor": MODEL_CONTEXT_LIMIT,
        "worker": MODEL_CONTEXT_LIMIT,
        "reducer": MODEL_CONTEXT_LIMIT,
        "validator": MODEL_CONTEXT_LIMIT,
        "finalizer": MODEL_CONTEXT_LIMIT,
    }

    def __init__(
        self,
        model: ChatModel,
        index: DocumentIndex,
        *,
        artifact_store: ArtifactStore | None = None,
        context_limit: int = MODEL_CONTEXT_LIMIT,
        max_workers: int = 8,
        default_agents: int = DEFAULT_AGENT_COUNT,
        shard_byte_limit: int = SOURCE_SHARD_BYTE_LIMIT,
        reduce_fan_in: int = 4,
        max_replans: int = 1,
        parallel_workers: int = 4,
        max_artifact_output_tokens: int = 12_000,
        target_artifact_characters: int = 1_200,
        available_tools: set[str] | None = None,
    ) -> None:
        if context_limit < 8_000:
            raise ValueError("模型上下文上限不能低于 8,000 Token")
        if max_workers < 1 or max_workers > 128:
            raise ValueError("动态子 Agent 数量必须在 1至128 之间")
        if default_agents < 1 or default_agents > 128:
            raise ValueError("默认动态 Agent 数量必须在 1至128 之间")
        if shard_byte_limit < 16 * 1024 or shard_byte_limit > 4 * 1024 * 1024:
            raise ValueError("单 Agent 分片阈值必须在 16 KB至4 MB 之间")
        if reduce_fan_in < 2 or reduce_fan_in > 8:
            raise ValueError("Reducer 扇入必须在 2至8 之间")
        if max_replans < 0 or max_replans > 3:
            raise ValueError("确定性外循环重规划次数必须在 0至3 之间")
        if parallel_workers < 1 or parallel_workers > 16:
            raise ValueError("并行模型请求数必须在 1至16 之间")
        if max_artifact_output_tokens < 2_000 or max_artifact_output_tokens > 24_000:
            raise ValueError("单 Agent 产物输出上限必须在 2,000至24,000 Token 之间")
        if target_artifact_characters < 400 or target_artifact_characters > 8_000:
            raise ValueError("目标分段长度必须在 400至8,000 字符之间")
        self.model = model
        self.index = index
        self.artifacts = artifact_store or ArtifactStore()
        self.context_limit = context_limit
        self.max_workers = max_workers
        self.default_agents = min(default_agents, max_workers)
        self.shard_byte_limit = shard_byte_limit
        self.reduce_fan_in = reduce_fan_in
        self.max_replans = max_replans
        self.parallel_workers = parallel_workers
        self.max_artifact_output_tokens = max_artifact_output_tokens
        self.target_artifact_characters = target_artifact_characters
        inferred_tools = {"model_reasoning"}
        if self.index.chunks:
            inferred_tools.add("source_search")
        self.available_tools = (
            {str(item) for item in available_tools if str(item) in BUILTIN_TOOLS}
            if available_tools is not None
            else inferred_tools
        )
        self.available_tools.add("model_reasoning")
        self._model_lock = threading.RLock()
        self._record_lock = threading.RLock()
        self._calls: list[dict[str, Any]] = []
        self._json_retries = 0
        self._planning_source = "unknown"
        self._intent = "unknown"
        self._retrieval_strategies: set[str] = set()
        self._allocation: dict[str, Any] = {}
        self._agent_strategy = ""
        self._conversation_context = ""
        self._delivery_contract: dict[str, Any] = {"response_mode": "synthesis"}
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("supervisor", self._supervisor)
        graph.add_node("worker", self._dynamic_worker)
        graph.add_node("dependency_scheduler", self._dependency_scheduler)
        graph.add_node("reducer", self._reduce_tree)
        graph.add_node("validator", self._validator)
        graph.add_node("supervisor_replan", self._supervisor_replan)
        graph.add_node("finalizer", self._finalize)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", self._dispatch_active_tasks, ["worker"])
        graph.add_edge("worker", "dependency_scheduler")
        graph.add_conditional_edges(
            "dependency_scheduler",
            self._route_dependencies,
            ["worker", "reducer"],
        )
        graph.add_edge("reducer", "validator")
        graph.add_conditional_edges(
            "validator",
            self._route_after_validation,
            ["supervisor_replan", "finalizer"],
        )
        graph.add_conditional_edges(
            "supervisor_replan",
            self._dispatch_after_replan,
            ["worker", "finalizer"],
        )
        graph.add_edge("finalizer", END)
        return graph.compile(checkpointer=MemorySaver())

    def answer(self, question: str, *, conversation_context: str = "") -> MultiAgentResult:
        if not question.strip():
            raise ValueError("问题不能为空")
        self._calls = []
        self._json_retries = 0
        self._planning_source = "unknown"
        self._intent = "unknown"
        self._retrieval_strategies = set()
        self._allocation = {}
        self._agent_strategy = ""
        self._conversation_context = str(conversation_context).strip()[:6_000]
        self._delivery_contract = self._infer_delivery_contract(
            question.strip(), self._conversation_context
        )
        self.artifacts.clear()
        state = self._graph.invoke(
            {
                "question": question.strip(), "findings": [], "trace": [], "iteration": 0,
                "delivery_contract": dict(self._delivery_contract),
            },
            config={
                "configurable": {"thread_id": uuid.uuid4().hex},
                "max_concurrency": self.parallel_workers,
            },
        )
        findings = self._deduplicate_findings(state.get("findings", []))
        evidence_ids = list(dict.fromkeys(
            evidence_id
            for finding in findings
            for evidence_id in finding.get("evidence_ids", [])
        ))
        citations = [
            citation
            for evidence_id in evidence_ids
            if (citation := self._citation(evidence_id)) is not None
        ]
        return MultiAgentResult(
            answer=state.get("answer", "未生成答案。"),
            citations=citations,
            tasks=state.get("tasks", []),
            findings=findings,
            trace=state.get("trace", []),
            validation=state.get("validation", {}),
            context_metrics=self._context_metrics(state.get("tasks", []), findings),
            stop_reason=state.get("stop_reason", "completed"),
        )

    def _supervisor(self, state: GraphState) -> dict[str, Any]:
        question = state["question"]
        source_stats = {
            "available": bool(self.index.chunks),
            "chunks": len(self.index.chunks),
            "indexed_bytes": self.index.total_indexed_bytes,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "ROLE:SUPERVISOR。你是动态 Agent 设计器和任务调度器。"
                    "不要从固定角色列表中选择 Agent；必须根据当前问题即时生成每个执行 Agent 的名称、"
                    "任务指令、能力和工具。不同任务可以生成完全不同的 Agent。"
                    "你不读取完整资料，只接收问题、对话摘要、可用工具和资料规模元数据。"
                    "将目标拆成边界清晰、可独立验证的子任务；只规划，不决定流程跳转或循环次数。"
                    "tools 只能使用本次明确提供的工具；capabilities 可以按问题自由命名。"
                    "source_policy 只能是 none、optional、required：不需要外部资料、可选参考资料、"
                    "或结论必须由外部资料支持。"
                    "任务存在先后依赖时，depends_on 只能引用排在当前任务之前的 task_XX；"
                    "没有依赖时返回空数组。创作类任务应先生成总纲，再并行生成章节，最后执行一致性检查。"
                    "同时判断用户最终需要的是综合回答还是完整产物。需要交付小说、报告、代码等完整内容时，"
                    "response_mode 必须为 full_artifact，正文任务的 output_mode 必须为 artifact；"
                    "要求总结、解释、比较、分析、问答时使用 synthesis/finding。不能只按内容名词判断："
                    "例如‘分析这部小说’应综合回答，‘写出整部小说’才交付完整产物；"
                    "‘总结报告’应摘要，‘撰写可直接提交的报告’应交付全文。"
                    "返回JSON：{\"problem_type\":...,\"strategy\":...,\"tasks\":[{"
                    "\"objective\":...,\"query\":...,\"agent_name\":...,"
                    "\"agent_instruction\":...,\"capabilities\":[...],\"tools\":[...],"
                    "\"source_policy\":...,\"priority\":1到100,\"depends_on\":[\"task_01\"],"
                    "\"output_mode\":\"finding|artifact\",\"include_in_final\":true/false,"
                    "\"sequence\":0,\"target_characters\":0,\"artifact_label\":...}],"
                    "\"delivery\":{\"response_mode\":\"synthesis|full_artifact\","
                    "\"artifact_type\":...,\"target_characters\":0,\"target_parts\":0}}。"
                    f"最多{self.max_workers}项。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"可用工具：{json.dumps(sorted(self.available_tools), ensure_ascii=False)}\n"
                    f"资料元数据：{json.dumps(source_stats, ensure_ascii=False)}\n"
                    f"默认并行下限：{self.default_agents}\n"
                    f"程序识别的交付约束：{json.dumps(self._delivery_contract, ensure_ascii=False)}\n"
                    + (
                        f"有界对话记忆（仅用于理解指代）：\n{self._conversation_context}\n"
                        if self._conversation_context else ""
                    )
                    + f"用户目标：{question}"
                ),
            },
        ]
        parsed, _ = self._call_json(
            "supervisor", messages, max_tokens=2_000, required_keys=("tasks",)
        )
        self._delivery_contract = self._resolve_delivery_contract(
            self._delivery_contract, parsed.get("delivery")
        )
        tasks = self._normalize_tasks(parsed.get("tasks"), question)
        tasks = self._apply_delivery_contract(tasks, question)
        planning_source = "dynamic_model_supervisor"
        intent = str(parsed.get("problem_type") or "general_problem")[:120]
        strategy = str(parsed.get("strategy") or "动态生成执行 Agent 并按子任务并行求解")[:500]
        tasks = self._allocate_agent_tasks(tasks, question)
        active_tasks = [task for task in tasks if not task.get("depends_on")]
        if not active_tasks and tasks:
            active_tasks = [tasks[0]]
        self._planning_source = planning_source
        self._intent = intent
        self._agent_strategy = strategy
        generated_names = list(dict.fromkeys(task["agent_name"] for task in tasks))
        return {
            "tasks": tasks,
            "active_tasks": active_tasks,
            "iteration": 0,
            "dependency_phase": 0,
            "planning_source": planning_source,
            "intent": intent,
            "agent_allocation": dict(self._allocation),
            "delivery_contract": dict(self._delivery_contract),
            "trace": [{
                "node": "supervisor",
                "role": "主 Agent",
                "status": "scheduled",
                "detail": (
                    f"问题类型={intent}；策略={strategy}；"
                    f"运行时生成 {len(generated_names)} 种 Agent 规格，"
                    f"分配 {self._allocation.get('allocated_agents', len(tasks))} 个隔离实例；"
                    f"首波执行 {len(active_tasks)} 个无依赖任务"
                ),
                "task_ids": [task["task_id"] for task in tasks],
                "generated_agents": generated_names,
            }],
        }

    def _infer_delivery_contract(self, question: str, conversation_context: str) -> dict[str, Any]:
        """Infer the response shape without hard-coding a novel-only workflow.

        The model may refine the plan, but explicit user requests for a complete
        deliverable are a deterministic contract and cannot be downgraded to a summary.
        """
        current = re.sub(r"\s+", "", question)
        prior = re.sub(r"\s+", "", conversation_context)
        combined = f"{current}\n{prior}"
        artifact_type = next((item for item in DELIVERABLE_TYPES if item in current), "")
        if not artifact_type and any(term in current for term in ("这个", "整个", "全文", "完整", "成稿")):
            artifact_type = next((item for item in DELIVERABLE_TYPES if item in prior), "")

        target_characters = self._parse_character_target(current)
        if not target_characters and artifact_type and any(
            term in current for term in ("这个", "整个", "继续", "全文", "完整", "成稿", "输出")
        ):
            target_characters = self._parse_character_target(prior)
        target_parts = self._parse_part_target(current, artifact_type)
        if not target_parts and artifact_type and any(
            term in current for term in ("这个", "整个", "继续", "全文", "完整", "成稿", "输出")
        ):
            target_parts = self._parse_part_target(prior, artifact_type)

        full_requested = bool(
            artifact_type
            and (
                target_characters
                or target_parts
                or any(term in current for term in FULL_DELIVERABLE_TERMS)
                or re.search(r"(?:写|生成|创作|输出|制作|撰写).{0,12}" + re.escape(artifact_type), current)
            )
        )
        summary_requested = any(term in current for term in ("总结", "概括", "摘要")) and not any(
            term in current for term in ("不要总结", "不要概括", "不是概括", "而不是概括")
        )
        if summary_requested and not target_characters and not target_parts:
            full_requested = False

        if full_requested and not target_parts:
            if target_characters:
                target_parts = max(
                    self.default_agents - 1,
                    math.ceil(target_characters / self.target_artifact_characters),
                )
            else:
                target_parts = max(1, self.default_agents - 1)
        if full_requested:
            reserved_control_agents = 2 if self.max_workers >= 3 else max(0, self.max_workers - 1)
            target_parts = max(1, min(target_parts, max(1, self.max_workers - reserved_control_agents)))

        return {
            "response_mode": "full_artifact" if full_requested else "synthesis",
            "artifact_type": artifact_type or "通用内容",
            "target_characters": max(0, target_characters),
            "target_parts": max(0, target_parts),
            "assembly": "ordered_lossless" if full_requested else "model_synthesis",
            "summary_may_replace_deliverable": False if full_requested else True,
            "user_preference": (
                "full_artifact" if full_requested else "synthesis" if summary_requested else "auto"
            ),
            "decision_source": (
                "explicit_user_full" if full_requested else "explicit_user_summary" if summary_requested
                else "pending_supervisor"
            ),
        }

    def _resolve_delivery_contract(
        self,
        inferred: dict[str, Any],
        model_delivery: Any,
    ) -> dict[str, Any]:
        """Resolve response shape with explicit user intent above Supervisor judgment."""
        contract = dict(inferred)
        preference = str(contract.get("user_preference", "auto"))
        if preference in {"full_artifact", "synthesis"}:
            return contract

        proposed = model_delivery if isinstance(model_delivery, dict) else {}
        proposed_mode = str(proposed.get("response_mode", "synthesis")).strip().casefold()
        if proposed_mode not in {"synthesis", "full_artifact"}:
            proposed_mode = "synthesis"
        contract["response_mode"] = proposed_mode
        contract["decision_source"] = "dynamic_supervisor"
        if proposed_mode == "full_artifact":
            proposed_type = str(proposed.get("artifact_type", "")).strip()[:80]
            if proposed_type:
                contract["artifact_type"] = proposed_type
            for key in ("target_characters", "target_parts"):
                if int(contract.get(key, 0) or 0) > 0:
                    continue
                try:
                    contract[key] = max(0, int(proposed.get(key, 0) or 0))
                except (TypeError, ValueError):
                    contract[key] = 0
            if not contract.get("target_parts"):
                target_characters = int(contract.get("target_characters", 0) or 0)
                contract["target_parts"] = (
                    max(self.default_agents - 1, math.ceil(target_characters / self.target_artifact_characters))
                    if target_characters else max(1, self.default_agents - 1)
                )
            reserved = 2 if self.max_workers >= 3 else max(0, self.max_workers - 1)
            contract["target_parts"] = max(
                1, min(int(contract["target_parts"]), max(1, self.max_workers - reserved))
            )
            contract["assembly"] = "ordered_lossless"
            contract["summary_may_replace_deliverable"] = False
        else:
            contract["assembly"] = "model_synthesis"
            contract["summary_may_replace_deliverable"] = True
        return contract

    @staticmethod
    def _parse_character_target(text: str) -> int:
        ten_thousands = re.search(r"(\d+(?:\.\d+)?)\s*万\s*(?:字|字符)", text)
        if ten_thousands:
            return min(2_000_000, max(1, round(float(ten_thousands.group(1)) * 10_000)))
        exact = re.search(r"(?<!\d)(\d[\d,，]*)\s*(?:字|字符)", text)
        if exact:
            return min(2_000_000, max(1, int(exact.group(1).replace(",", "").replace("，", ""))))
        return 0

    @staticmethod
    def _parse_part_target(text: str, artifact_type: str) -> int:
        units = "章|章节" if artifact_type in {"小说", "故事"} else "章|章节|部分|节|篇|模块"
        match = re.search(rf"(?<!\d)(\d{{1,3}})\s*(?:个)?(?:{units})", text)
        return int(match.group(1)) if match else 0

    def _apply_delivery_contract(
        self,
        tasks: list[dict[str, Any]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Turn a full-deliverable request into runtime-generated ordered artifacts."""
        if self._delivery_contract.get("response_mode") != "full_artifact":
            return tasks

        artifact_type = str(self._delivery_contract.get("artifact_type") or "内容")
        part_count = max(1, int(self._delivery_contract.get("target_parts", 1)))
        target_total = max(0, int(self._delivery_contract.get("target_characters", 0)))
        outline_template = next((
            dict(task) for task in tasks
            if any(term in f"{task.get('objective', '')}{task.get('agent_name', '')}" for term in ("大纲", "总纲", "结构", "规划"))
        ), None)
        content_template = next((
            dict(task) for task in tasks
            if str(task.get("output_mode", "")).casefold() == "artifact"
            or any(term in str(task.get("objective", "")) for term in ("撰写", "创作", "正文", "章节", "实现"))
        ), None)

        outline = outline_template or {
            "objective": f"为{artifact_type}生成完整结构与各部分约束",
            "query": f"根据用户目标设计{artifact_type}整体结构，明确每一部分的内容、衔接和一致性约束。",
            "agent_name": f"{artifact_type}结构规划 Agent",
            "agent_instruction": f"为当前{artifact_type}即时生成可执行结构，逐部分说明目标、承接关系与必须保持的一致性。",
            "capabilities": ["结构规划", "一致性约束"],
            "tools": ["model_reasoning"],
            "source_policy": "none" if not self.index.chunks else "optional",
            "priority": 100,
        }
        outline.update({
            "task_id": "task_01",
            "base_task_id": str(outline.get("task_id", "task_01")),
            "depends_on": [],
            "phase": 0,
            "output_mode": "finding",
            "include_in_final": False,
            "sequence": 0,
            "target_characters": 0,
            "artifact_label": f"{artifact_type}结构",
            "input_budget": min(
                self.ROLE_BUDGETS["worker"], self.context_limit - OUTPUT_RESERVE - SAFETY_MARGIN
            ),
            "required_targets": [],
        })

        result = [outline] if self.max_workers > 1 else []
        outline_dependency = ["task_01"] if result else []
        first_content_task_number = 2 if result else 1
        if artifact_type in {"小说", "故事"}:
            label = lambda index: f"第{index}章"
        else:
            label = lambda index: f"第{index}部分"
        base_characters, remainder = divmod(target_total, part_count) if target_total else (0, 0)
        for index in range(1, part_count + 1):
            section_label = label(index)
            minimum = base_characters + (1 if index <= remainder else 0)
            template = dict(content_template or {})
            capabilities = template.get("capabilities") or ["内容生成", "上下文衔接", "自检"]
            template.update({
                "task_id": f"task_{index + first_content_task_number - 1:02d}",
                "base_task_id": str(template.get("task_id", f"generated_part_{index:02d}")),
                "objective": f"完成{artifact_type}{section_label}的完整正文",
                "query": (
                    f"用户总目标：{question}\n依据整体结构，生成{section_label}完整正文。"
                    + (f"正文不得少于 {minimum} 个字符。" if minimum else "交付可直接使用的完整正文。")
                )[:1_000],
                "agent_name": f"{artifact_type}{section_label}生成 Agent",
                "agent_instruction": (
                    f"独立完成{section_label}的可直接交付正文；遵守结构 Agent 给出的设定和衔接约束；"
                    "不得返回摘要、完成报告、写作说明或省略号占位。"
                ),
                "capabilities": list(capabilities),
                "tools": ["model_reasoning"],
                "source_policy": "none" if not self.index.chunks else str(template.get("source_policy", "optional")),
                "priority": max(1, 100 - index),
                "input_budget": min(
                    self.ROLE_BUDGETS["worker"], self.context_limit - OUTPUT_RESERVE - SAFETY_MARGIN
                ),
                "depends_on": outline_dependency,
                "phase": 1 if outline_dependency else 0,
                "required_targets": [],
                "output_mode": "artifact",
                "include_in_final": True,
                "sequence": index,
                "target_characters": minimum,
                "artifact_label": section_label,
            })
            result.append(template)
        if len(result) < self.max_workers and result and outline_dependency:
            review_template = next((
                dict(task) for task in tasks
                if any(term in f"{task.get('objective', '')}{task.get('agent_name', '')}" for term in ("一致性", "审查", "校验", "检查"))
            ), {})
            content_task_ids = [
                str(task["task_id"]) for task in result if task.get("output_mode") == "artifact"
            ]
            review_template.update({
                "task_id": f"task_{len(result) + 1:02d}",
                "base_task_id": str(review_template.get("task_id", "generated_review")),
                "objective": f"检查{artifact_type}各部分的连续性与交付完整性",
                "query": f"检查{artifact_type}各部分是否遵守整体结构，指出矛盾、断裂、缺漏和未完成内容。",
                "agent_name": f"{artifact_type}整体审校 Agent",
                "agent_instruction": "根据结构摘要和各部分的有界摘要执行整体审校，不改写或替代原始正文。",
                "capabilities": ["一致性审查", "完整性检查"],
                "tools": ["model_reasoning"],
                "source_policy": "none",
                "priority": 1,
                "input_budget": min(
                    self.ROLE_BUDGETS["worker"], self.context_limit - OUTPUT_RESERVE - SAFETY_MARGIN
                ),
                "depends_on": content_task_ids,
                "phase": 2,
                "required_targets": [],
                "output_mode": "finding",
                "include_in_final": False,
                "sequence": 0,
                "target_characters": 0,
                "artifact_label": f"{artifact_type}整体审校",
            })
            result.append(review_template)
        return result[: self.max_workers]

    def _normalize_tasks(self, raw_tasks: Any, question: str) -> list[dict[str, Any]]:
        tasks: list[TaskSpec] = []
        seen: set[tuple[str, str]] = set()
        if isinstance(raw_tasks, list):
            for raw in raw_tasks:
                if len(tasks) >= self.max_workers or not isinstance(raw, dict):
                    break
                objective = str(raw.get("objective", "")).strip()[:500]
                query = str(raw.get("query", objective)).strip()[:500]
                agent_name = str(raw.get("agent_name") or f"{objective[:24]} Agent").strip()[:80]
                instruction = str(raw.get("agent_instruction") or (
                    f"独立完成子任务：{objective}。说明采用的推理依据并返回结构化结论。"
                )).strip()[:1_000]
                capabilities = self._string_list(raw.get("capabilities"), fallback=("reasoning",))
                requested_tools = self._string_list(raw.get("tools"), fallback=("model_reasoning",))
                tools = tuple(item for item in requested_tools if item in self.available_tools)
                if "model_reasoning" not in tools:
                    tools = ("model_reasoning", *tools)
                source_policy = str(raw.get("source_policy", "optional")).strip().casefold()
                if source_policy not in SOURCE_POLICIES:
                    source_policy = "optional"
                if source_policy != "none" and "source_search" in self.available_tools and self.index.chunks:
                    tools = tuple(dict.fromkeys((*tools, "source_search")))
                if not self.index.chunks:
                    source_policy = "none"
                key = (objective.casefold(), query.casefold())
                if not objective or not query or key in seen:
                    continue
                seen.add(key)
                try:
                    priority = max(1, min(100, int(raw.get("priority", 50))))
                except (TypeError, ValueError):
                    priority = 50
                known_task_ids = {item.task_id for item in tasks}
                requested_dependencies = self._string_list(raw.get("depends_on"), fallback=())
                depends_on = tuple(
                    item for item in requested_dependencies
                    if item in known_task_ids
                )
                phases = {item.task_id: item.phase for item in tasks}
                phase = 1 + max((phases[item] for item in depends_on), default=-1)
                output_mode = str(raw.get("output_mode", "finding")).strip().casefold()
                if output_mode not in OUTPUT_MODES:
                    output_mode = "finding"
                try:
                    sequence = max(0, int(raw.get("sequence", 0)))
                    target_characters = max(0, int(raw.get("target_characters", 0)))
                except (TypeError, ValueError):
                    sequence, target_characters = 0, 0
                tasks.append(TaskSpec(
                    task_id=f"task_{len(tasks) + 1:02d}",
                    objective=objective,
                    query=query,
                    agent_name=agent_name,
                    agent_instruction=instruction,
                    capabilities=capabilities,
                    tools=tools,
                    source_policy=source_policy,
                    priority=priority,
                    input_budget=min(
                        self.ROLE_BUDGETS["worker"],
                        self.context_limit - OUTPUT_RESERVE - SAFETY_MARGIN,
                    ),
                    depends_on=depends_on,
                    phase=phase,
                    required_targets=(objective,) if source_policy == "required" else (),
                    output_mode=output_mode,
                    include_in_final=bool(raw.get("include_in_final", output_mode == "artifact")),
                    sequence=sequence,
                    target_characters=target_characters,
                    artifact_label=str(raw.get("artifact_label") or objective)[:120],
                ))
        if not tasks:
            tasks = [TaskSpec(
                task_id="task_01",
                objective=question[:500],
                query=question[:500],
                agent_name="问题求解 Agent",
                agent_instruction=f"独立分析并解决用户问题：{question[:500]}",
                capabilities=("reasoning", "self_check"),
                tools=tuple(sorted(self.available_tools)),
                source_policy="optional" if self.index.chunks else "none",
                input_budget=min(
                    self.ROLE_BUDGETS["worker"],
                    self.context_limit - OUTPUT_RESERVE - SAFETY_MARGIN,
                ),
                required_targets=(question[:500],) if self.index.chunks else (),
            )]
        return [task.as_dict() for task in tasks]

    @staticmethod
    def _string_list(value: Any, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
        if not isinstance(value, list):
            return fallback
        cleaned = tuple(dict.fromkeys(str(item).strip()[:80] for item in value if str(item).strip()))
        return cleaned[:8] or fallback

    def _allocate_agent_tasks(
        self,
        base_tasks: list[dict[str, Any]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Scale workers from the plan and only use source size for source-required work."""
        chunks = self.index.all_chunks()
        indexed_bytes = self.index.total_indexed_bytes
        indexed_tokens = sum(chunk.estimated_tokens for chunk in chunks)
        compact = re.sub(r"\s+", "", question)
        exhaustive_scan = any(term in compact for term in EXHAUSTIVE_SCAN_TERMS)
        source_tasks = [
            task for task in base_tasks
            if task.get("source_policy") == "required" and "source_search" in task.get("tools", [])
        ]
        source_scaling = bool(source_tasks and chunks)
        target_shard_bytes = self.shard_byte_limit // 2 if exhaustive_scan else self.shard_byte_limit
        target_shard_tokens = min(
            TARGET_SOURCE_TOKENS_PER_AGENT,
            self.context_limit - OUTPUT_RESERVE - SAFETY_MARGIN - 8_000,
        )
        byte_required = max(1, math.ceil(indexed_bytes / target_shard_bytes)) if source_scaling else 1
        token_required = max(1, math.ceil(indexed_tokens / target_shard_tokens)) if source_scaling else 1
        chunk_required = max(1, math.ceil(len(chunks) / TARGET_CHUNKS_PER_AGENT)) if source_scaling else 1
        complexity_markers = sum(
            compact.count(marker)
            for marker in ("同时", "分别", "逐项", "以及", "并且", "、", "；", "比较", "风险")
        )
        complexity_required = min(self.max_workers, 1 + complexity_markers + len(compact) // 500)
        desired = max(
            self.default_agents,
            len(base_tasks),
            byte_required,
            token_required,
            chunk_required,
            complexity_required,
        )
        allocated = min(self.max_workers, desired)

        reasons: list[str] = [f"默认至少 {self.default_agents} 个"]
        if source_scaling and byte_required > self.default_agents:
            reasons.append(
                f"索引文本 {indexed_bytes} Bytes 按 {target_shard_bytes} Bytes 目标分片需要 {byte_required} 个"
            )
        if source_scaling and token_required > self.default_agents:
            reasons.append(f"索引文本约 {indexed_tokens} Token 需要 {token_required} 个安全分片")
        if source_scaling and chunk_required > self.default_agents:
            reasons.append(f"{len(chunks)} 个检索块需要 {chunk_required} 个")
        if complexity_required > self.default_agents:
            reasons.append(f"任务复杂度需要 {complexity_required} 个")
        if desired > self.max_workers:
            reasons.append(f"受最大 Worker 数 {self.max_workers} 限制")

        allocated_tasks: list[dict[str, Any]] = []
        chunk_count = len(chunks)
        shard_ranges = self._balanced_token_shards(chunks, allocated if source_scaling else 1)
        combined_targets = list(dict.fromkeys(
            str(task.get("objective", "")).strip()
            for task in source_tasks
            if str(task.get("objective", "")).strip()
        ))
        combined_source_query = "\n".join(dict.fromkeys(
            str(task.get("query", "")).strip()
            for task in source_tasks
            if str(task.get("query", "")).strip()
        ))
        for slot in range(allocated):
            template = dict(base_tasks[slot % len(base_tasks)])
            if not source_scaling or chunk_count == 0:
                chunk_start = chunk_end = 0
            else:
                chunk_start, chunk_end = shard_ranges[slot]
            shard_chunks = chunks[chunk_start:chunk_end]
            combined_query = combined_source_query if source_scaling else str(template.get("query", "")).strip()
            if question.casefold() not in combined_query.casefold():
                combined_query = f"{combined_query}\n总体目标：{question}".strip()
            template.update({
                "task_id": f"task_{slot + 1:02d}",
                "base_task_id": template.get("task_id", ""),
                "query": combined_query[:1_000],
                "agent_instance_id": f"dynamic_worker_{slot + 1:02d}",
                "shard_index": slot + 1,
                "shard_count": allocated,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "shard_chunks": len(shard_chunks),
                "shard_bytes": sum(chunk.byte_size for chunk in shard_chunks),
                "shard_tokens": sum(chunk.estimated_tokens for chunk in shard_chunks),
                "coverage_required": source_scaling,
                "required_targets": combined_targets if source_scaling else template.get("required_targets", []),
                "target_queries": [
                    {"target": task.get("objective", ""), "query": task.get("query", "")}
                    for task in source_tasks
                ] if source_scaling else [],
            })
            if source_scaling:
                template["source_policy"] = "required"
                template["tools"] = list(dict.fromkeys([*template.get("tools", []), "source_search"]))
                template["depends_on"] = []
                template["phase"] = 0
                template["agent_instruction"] = (
                    f"{template.get('agent_instruction', '')} 完整扫描分配到的连续分片，并同时检查："
                    f"{'；'.join(combined_targets)}。不得用局部未命中推断整份资料不存在。"
                )[:1_000]
            allocated_tasks.append(template)

        if not source_scaling:
            generated_by_base: dict[str, list[str]] = {}
            for task in allocated_tasks:
                generated_by_base.setdefault(str(task.get("base_task_id", "")), []).append(task["task_id"])
            for task in allocated_tasks:
                dependency_ids = [
                    generated_id
                    for base_dependency in task.get("depends_on", [])
                    for generated_id in generated_by_base.get(str(base_dependency), [])
                ]
                task["depends_on"] = list(dict.fromkeys(dependency_ids))

        self._allocation = {
            "default_agents": self.default_agents,
            "desired_agents": desired,
            "allocated_agents": allocated,
            "max_agents": self.max_workers,
            "source_indexed_bytes": indexed_bytes,
            "source_indexed_tokens": indexed_tokens,
            "largest_source_indexed_bytes": self.index.largest_source_indexed_bytes,
            "shard_byte_limit": self.shard_byte_limit,
            "target_shard_bytes": target_shard_bytes,
            "target_shard_tokens": target_shard_tokens,
            "source_exceeds_shard_limit": self.index.largest_source_indexed_bytes > self.shard_byte_limit,
            "multi_agent_sharding_active": allocated > 1,
            "exhaustive_scan": exhaustive_scan,
            "source_scaling_active": source_scaling,
            "required_targets": combined_targets,
            "allocation_reason": "；".join(reasons),
            "parallel_workers": self.parallel_workers,
            "max_artifact_output_tokens": self.max_artifact_output_tokens,
            "target_artifact_characters": self.target_artifact_characters,
        }
        return allocated_tasks

    @staticmethod
    def _balanced_token_shards(chunks: list[Any], count: int) -> list[tuple[int, int]]:
        if not chunks or count <= 0:
            return [(0, 0)] * max(1, count)
        requested_count = count
        count = min(count, len(chunks))
        ranges: list[tuple[int, int]] = []
        start = 0
        remaining_tokens = sum(max(1, chunk.estimated_tokens) for chunk in chunks)
        for slot in range(count):
            remaining_slots = count - slot
            if remaining_slots == 1:
                end = len(chunks)
            else:
                target = max(1, math.ceil(remaining_tokens / remaining_slots))
                used = 0
                end = start
                last_allowed = len(chunks) - (remaining_slots - 1)
                while end < last_allowed and (used < target or end == start):
                    used += max(1, chunks[end].estimated_tokens)
                    end += 1
            shard_tokens = sum(max(1, chunk.estimated_tokens) for chunk in chunks[start:end])
            ranges.append((start, end))
            remaining_tokens -= shard_tokens
            start = end
        while len(ranges) < requested_count:
            ranges.append(ranges[len(ranges) % count])
        return ranges

    @staticmethod
    def _dispatch_active_tasks(state: GraphState):
        findings_by_task = {
            str(item.get("task_id", "")): item
            for item in state.get("findings", [])
        }
        return [
            Send("worker", {
                "question": state["question"],
                "task": task,
                "dependency_findings": MultiAgentResearchSystem._bounded_finding_summaries([
                    findings_by_task[dependency]
                    for dependency in task.get("depends_on", []) if dependency in findings_by_task
                ]),
            })
            for task in state.get("active_tasks", [])
        ]

    def _dependency_scheduler(self, state: GraphState) -> dict[str, Any]:
        completed = {
            str(item.get("task_id", ""))
            for item in self._deduplicate_findings(state.get("findings", []))
        }
        pending = [task for task in state.get("tasks", []) if task["task_id"] not in completed]
        ready = [
            task for task in pending
            if set(task.get("depends_on", [])).issubset(completed)
        ]
        phase = state.get("dependency_phase", 0) + (1 if ready else 0)
        detail = (
            f"依赖波次 {phase}：调度 {len(ready)} 个就绪任务，剩余 {len(pending) - len(ready)} 个"
            if ready else f"依赖执行完成：{len(completed)}/{len(state.get('tasks', []))} 个任务已完成"
        )
        return {
            "active_tasks": ready,
            "dependency_phase": phase,
            "trace": [{
                "node": "dependency_scheduler",
                "role": "依赖调度器",
                "status": "scheduled" if ready else "completed",
                "detail": detail,
                "task_ids": [task["task_id"] for task in ready],
            }],
        }

    @staticmethod
    def _route_dependencies(state: GraphState):
        if state.get("active_tasks"):
            return MultiAgentResearchSystem._dispatch_active_tasks(state)
        return "reducer"

    def _dynamic_worker(self, state: DynamicWorkerState) -> dict[str, Any]:
        task = state["task"]
        hits, retrieval_strategy, retrieval_meta = self._retrieve_hits(task)
        with self._record_lock:
            self._retrieval_strategies.add(retrieval_strategy)
        # input_budget is the maximum safe prompt budget, not a target. Leave room for
        # the generated AgentSpec, current question and bounded conversation context.
        evidence_budget = int(task.get("input_budget", 61_000)) - 4_000
        evidence_ids, packed, pack_meta = self._pack_evidence(
            hits,
            budget=max(1_000, evidence_budget),
            task=task,
        )
        assigned_chunks = int(task.get("shard_chunks", 0))
        coverage_required = bool(task.get("coverage_required", False))
        scanned_chunks = int(pack_meta["packed_chunks"]) if coverage_required else 0
        coverage_complete = bool(
            not coverage_required
            or (not pack_meta["truncated"] and scanned_chunks >= assigned_chunks)
        )
        coverage = {
            "required": coverage_required,
            "complete": coverage_complete,
            "shard_index": int(task.get("shard_index", 1) or 1),
            "shard_count": int(task.get("shard_count", 1) or 1),
            "chunk_start": int(task.get("chunk_start", 0) or 0),
            "chunk_end": int(task.get("chunk_end", 0) or 0),
            "assigned_chunks": assigned_chunks,
            "scanned_chunks": scanned_chunks,
            "assigned_tokens": int(task.get("shard_tokens", 0) or 0),
            "packed_tokens": int(pack_meta["packed_tokens"]),
            "candidate_matches": int(retrieval_meta.get("candidate_matches", 0)),
            "target_candidates": retrieval_meta.get("target_candidates", {}),
            "retrieval_strategy": retrieval_strategy,
        }
        dependency_block = (
            "依赖任务的结构化结果：\n"
            f"{json.dumps(state.get('dependency_findings', []), ensure_ascii=False)}\n"
            if state.get("dependency_findings") else ""
        )
        conversation_block = (
            "有界对话记忆（只用于理解当前问题，不可作为事实证据）：\n"
            f"{self._conversation_context}\n"
            if self._conversation_context else ""
        )
        deliverable_artifact_id = ""
        if task.get("output_mode") == "artifact":
            target_characters = max(0, int(task.get("target_characters", 0) or 0))
            requested_characters = math.ceil(target_characters * 1.15) if target_characters else 0
            artifact_messages = [
                {
                    "role": "system",
                    "content": (
                        "ROLE:DYNAMIC_WORKER。你是 Supervisor 根据当前交付任务即时生成的内容 Agent。"
                        f"Agent 名称：{task['agent_name']}。任务指令：{task['agent_instruction']}。"
                        "本次响应会被程序作为最终产物原文保存并按顺序无损组装。"
                        "只输出可直接交付的正文，不输出 JSON，不写完成情况、摘要、解释、字数统计或 Validator 报告；"
                        "不得用‘略’、省略号或提纲代替正文。可以使用 Markdown 标题和正常段落。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"总体请求：{state['question']}\n"
                        f"当前部分：{task.get('artifact_label') or task['objective']}\n"
                        f"当前目标：{task['objective']}\n工作要求：{task['query']}\n"
                        + (f"请生成至少 {requested_characters} 个字符，确保硬性下限为 {target_characters} 个字符。\n" if target_characters else "")
                        + dependency_block
                        + conversation_block
                        + f"可选外部资料：\n{packed or '无；使用模型原生能力完成。'}"
                    ),
                },
            ]
            output_tokens = min(
                self.max_artifact_output_tokens,
                max(2_400, math.ceil(max(1, requested_characters) * 1.8)),
            )
            deliverable_text = self._call(
                "worker", artifact_messages, max_tokens=output_tokens
            ).strip()
            deliverable_artifact_id = self.artifacts.put(
                "deliverable",
                deliverable_text,
                {
                    "task_id": task["task_id"],
                    "agent_name": task["agent_name"],
                    "sequence": int(task.get("sequence", 0) or 0),
                    "label": str(task.get("artifact_label", "")),
                    "target_characters": target_characters,
                },
            )
            parsed = {
                "summary": (
                    f"{task.get('artifact_label') or task['objective']}正文片段："
                    f"{deliverable_text[:1_200]}"
                ),
                "facts": [],
                "claims": [],
                "uncertainties": [],
                "contradictions": [],
                "evidence_ids": [],
                "confidence": 0.9,
            }
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "ROLE:DYNAMIC_WORKER。你不是预制角色，而是 Supervisor 为当前问题即时创建的执行 Agent。"
                        f"Agent 名称：{task['agent_name']}。任务指令：{task['agent_instruction']}。"
                        f"能力：{json.dumps(task.get('capabilities', []), ensure_ascii=False)}。"
                        f"允许工具：{json.dumps(task.get('tools', []), ensure_ascii=False)}。"
                        f"来源策略：{task.get('source_policy', 'none')}。"
                        "你只处理当前子任务，不知道其他 Worker 的消息，也不能调用未授权工具。"
                        "来源策略为 required 时，事实结论必须引用给定 evidence_id；为 none 时可使用模型通用知识。"
                        "返回JSON：{\"summary\":...,\"facts\":[...],\"claims\":[...],"
                        "\"uncertainties\":[...],\"contradictions\":[...],"
                        "\"evidence_ids\":[...],\"confidence\":0到1}。不得编造证据ID。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"task_id：{task['task_id']}\n目标：{task['objective']}\n"
                        f"工作查询：{task['query']}\n"
                        + dependency_block
                        + conversation_block
                        + f"\n本任务可见的外部资料：\n{packed or '未提供外部资料；请按来源策略使用模型能力完成任务。'}"
                    ),
                },
            ]
            parsed, _ = self._call_json(
                "worker", messages, max_tokens=1_800,
                required_keys=("summary", "evidence_ids"),
            )
        valid_ids = [
            item for item in parsed.get("evidence_ids", [])
            if isinstance(item, str) and item in evidence_ids
        ]
        claims = [str(item)[:800] for item in parsed.get("claims", []) if str(item).strip()]
        facts = [str(item)[:800] for item in parsed.get("facts", []) if str(item).strip()]
        if not facts:
            facts = list(claims)
        uncertainties = [str(item)[:800] for item in parsed.get("uncertainties", []) if str(item).strip()]
        contradictions = [str(item)[:800] for item in parsed.get("contradictions", []) if str(item).strip()]
        summary = str(parsed.get("summary", "")).strip()
        if not summary:
            summary = (
                "没有足够的外部资料支持该子任务。"
                if task.get("source_policy") == "required" and not hits
                else (hits[0].chunk.text[:800] if hits else "当前 Agent 未形成有效结论。")
            )
        finding = {
            "task_id": task["task_id"],
            "agent_name": task["agent_name"],
            "agent_instruction": task["agent_instruction"],
            "capabilities": task.get("capabilities", []),
            "tools": task.get("tools", []),
            "source_policy": task.get("source_policy", "none"),
            "objective": task["objective"],
            "summary": summary[:1_500],
            "facts": facts[:12],
            "claims": claims[:8],
            "uncertainties": uncertainties[:8],
            "contradictions": contradictions[:8],
            "evidence_ids": valid_ids,
            "confidence": self._confidence(parsed.get("confidence")),
            "retrieval_strategy": retrieval_strategy,
            "answer_basis": "external_sources" if valid_ids else "model_reasoning",
            "agent_instance_id": task.get("agent_instance_id"),
            "shard_index": task.get("shard_index"),
            "shard_count": task.get("shard_count"),
            "shard_bytes": task.get("shard_bytes"),
            "shard_tokens": task.get("shard_tokens"),
            "depends_on": task.get("depends_on", []),
            "phase": task.get("phase", 0),
            "required_targets": task.get("required_targets", []),
            "coverage": coverage,
            "negative_claim": self._looks_negative(summary),
            "output_mode": task.get("output_mode", "finding"),
            "include_in_final": bool(task.get("include_in_final", False)),
            "sequence": int(task.get("sequence", 0) or 0),
            "target_characters": int(task.get("target_characters", 0) or 0),
            "artifact_label": str(task.get("artifact_label", "")),
            "deliverable_artifact_id": deliverable_artifact_id,
            "deliverable_characters": (
                len(self.artifacts.get(deliverable_artifact_id).content)
                if deliverable_artifact_id and self.artifacts.get(deliverable_artifact_id) else 0
            ),
        }
        finding["artifact_id"] = self.artifacts.put(
            "finding", json.dumps(finding, ensure_ascii=False),
            {"task_id": task["task_id"], "agent_name": task["agent_name"]},
        )
        return {
            "findings": [finding],
            "trace": [{
                "node": "worker",
                "role": task["agent_name"],
                "task_id": task["task_id"],
                "status": "completed",
                "detail": (
                    f"独立执行完成；分片={task.get('shard_index', 1)}/{task.get('shard_count', 1)}；"
                    f"工具={','.join(task.get('tools', []))}；分片大小={task.get('shard_bytes', 0)} Bytes；"
                    f"资料策略={retrieval_strategy}；"
                    f"覆盖={scanned_chunks}/{assigned_chunks}；引用 {len(valid_ids)} 个证据对象"
                    + (
                        f"；完整产物={finding['deliverable_characters']} 字符，已写入独立 Artifact"
                        if deliverable_artifact_id else ""
                    )
                ),
                "artifact_id": finding["artifact_id"],
            }],
        }

    def _retrieve_hits(
        self,
        task: dict[str, Any],
    ) -> tuple[list[SearchHit], str, dict[str, Any]]:
        """Use bounded external context only when the generated Agent requests that tool."""
        if "source_search" not in task.get("tools", []) or not self.index.chunks:
            return [], "model_reasoning_only", {"candidate_matches": 0, "target_candidates": {}}
        query = str(task.get("query", ""))
        referential_terms = ("这个", "该内容", "上述", "前面", "刚才", "它", "其", "这些", "他们")
        if self._conversation_context and any(term in query for term in referential_terms):
            query = f"{query}\n{self._conversation_context}"
        chunk_start = max(0, int(task.get("chunk_start", 0)))
        chunk_end = min(
            len(self.index.chunks),
            int(task.get("chunk_end", len(self.index.chunks))),
        )
        shard_chunks = self.index.all_chunks()[chunk_start:chunk_end]
        compact = re.sub(r"\s+", "", f"{task.get('objective', '')}{query}".casefold())
        target_candidates: dict[str, int] = {}
        candidate_hits: list[SearchHit] = []
        for target in task.get("target_queries", []):
            target_name = str(target.get("target", "")).strip()
            target_query = str(target.get("query", target_name)).strip()
            if not target_name or not target_query:
                continue
            matches = self.index.search(
                target_query,
                limit=5,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
            direct_terms = [
                re.sub(r"\s+", "", value).casefold()
                for value in (target_query, target_name)
                if len(re.sub(r"\s+", "", value)) >= 2
            ]
            direct_matches = [
                hit for hit in matches
                if any(term in re.sub(r"\s+", "", hit.chunk.text).casefold() for term in direct_terms)
            ]
            target_candidates[target_name] = len(direct_matches)
            candidate_hits.extend(direct_matches)

        if task.get("source_policy") == "required" or task.get("coverage_required"):
            combined: list[SearchHit] = []
            seen: set[str] = set()
            for hit in [
                *candidate_hits,
                *(SearchHit(chunk=chunk, score=1.0, source="shard_full_scan") for chunk in shard_chunks),
            ]:
                if hit.chunk.chunk_id in seen:
                    continue
                seen.add(hit.chunk.chunk_id)
                combined.append(hit)
            return combined, "sharded_full_coverage", {
                "candidate_matches": len({hit.chunk.chunk_id for hit in candidate_hits}),
                "target_candidates": target_candidates,
            }
        lexical_hits = self.index.search(
            query,
            limit=7,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        broad_terms = ("总结", "概括", "全文", "主要内容", "核心内容", "要点", "文档开头", "文档末尾")
        capabilities = " ".join(str(item).casefold() for item in task.get("capabilities", []))
        needs_coverage = any(term in capabilities for term in ("synthesis", "summary", "归纳", "总结")) or any(
            term in compact for term in broad_terms
        )
        if not needs_coverage:
            if lexical_hits:
                return lexical_hits, "sharded_hybrid_exact", {
                    "candidate_matches": len(lexical_hits), "target_candidates": target_candidates,
                }
            fallback = [
                SearchHit(chunk=chunk, score=0.01, source="shard_fallback")
                for chunk in shard_chunks[:3]
            ]
            return fallback, "sharded_positional_fallback", {
                "candidate_matches": 0, "target_candidates": target_candidates,
            }

        chunks = shard_chunks
        coverage_hits: list[SearchHit] = []
        if chunks:
            sample_count = min(5, len(chunks))
            positions = {
                round(index * (len(chunks) - 1) / max(1, sample_count - 1))
                for index in range(sample_count)
            }
            if "末尾" in compact or "结尾" in compact:
                ordered_positions = sorted(positions, reverse=True)
            elif "开头" in compact or "背景" in compact:
                ordered_positions = sorted(positions)
            else:
                midpoint = (len(chunks) - 1) / 2
                middle_first = sorted(positions, key=lambda position: abs(position - midpoint))
                anchors = [middle_first[0], min(positions), max(positions)]
                ordered_positions = list(dict.fromkeys([*anchors, *middle_first]))
            coverage_hits = [
                SearchHit(chunk=chunks[position], score=0.0, source="positional_coverage")
                for position in ordered_positions
            ]

        # Put coverage first so a long lexical hit list cannot crowd the document tail out.
        combined: list[SearchHit] = []
        seen: set[str] = set()
        for hit in [*coverage_hits, *lexical_hits]:
            if hit.chunk.chunk_id in seen:
                continue
            seen.add(hit.chunk.chunk_id)
            combined.append(hit)
        if not combined and chunks:
            combined = [SearchHit(chunk=chunks[0], score=0.01, source="shard_fallback")]
        return combined[:10], "sharded_hybrid_plus_positional_coverage", {
            "candidate_matches": len(lexical_hits), "target_candidates": target_candidates,
        }

    def _pack_evidence(
        self,
        hits: list[SearchHit],
        *,
        budget: int,
        task: dict[str, Any],
    ) -> tuple[list[str], str, dict[str, Any]]:
        artifact_ids: list[str] = []
        blocks: list[str] = []
        used = 0
        packed_chunk_ids: list[str] = []
        truncated = False
        for hit in hits:
            text = hit.chunk.text.strip()
            cost = estimate_tokens(text) + 80
            if blocks and used + cost > budget:
                truncated = True
                break
            source = self.index.source_for(hit.chunk.document_name)
            indexed_bytes = int(source.get("indexed_bytes", 0))
            artifact_id = self.artifacts.put("evidence", text, {
                "chunk_id": hit.chunk.chunk_id,
                "document": hit.chunk.document_name,
                "section": hit.chunk.section_title,
                "score": round(hit.score, 6),
                "retrieval_source": hit.source,
                "document_bytes": int(source.get("source_bytes", 0)),
                "indexed_bytes": indexed_bytes,
                "chunk_bytes": hit.chunk.byte_size,
                "source_exceeds_shard_limit": indexed_bytes > self.shard_byte_limit,
                "shard_byte_limit": self.shard_byte_limit,
                "task_id": task.get("task_id"),
                "agent_name": task.get("agent_name"),
                "agent_instance_id": task.get("agent_instance_id"),
                "shard_index": task.get("shard_index", 1),
                "shard_count": task.get("shard_count", 1),
                "shard_bytes": task.get("shard_bytes", 0),
            })
            artifact_ids.append(artifact_id)
            blocks.append(f"[artifact_id={artifact_id}]\n{text}")
            packed_chunk_ids.append(hit.chunk.chunk_id)
            used += cost
        return artifact_ids, "\n\n".join(blocks), {
            "packed_chunks": len(packed_chunk_ids),
            "packed_chunk_ids": packed_chunk_ids,
            "packed_tokens": used,
            "truncated": truncated or len(packed_chunk_ids) < len(hits),
        }

    def _reduce_tree(self, state: GraphState) -> dict[str, Any]:
        deduplicated = self._deduplicate_findings(state["findings"])
        if self._delivery_contract.get("response_mode") == "full_artifact":
            summaries = self._bounded_finding_summaries(deduplicated)
            reduced = {
                "summary": (
                    f"完整产物正文保存在 {sum(bool(item.get('deliverable_artifact_id')) for item in deduplicated)} "
                    "个独立 Artifact 中；Reducer 仅合并有界审校账本。"
                ),
                "facts": list(dict.fromkeys(
                    str(value)[:800] for item in summaries for value in item.get("facts", [])
                    if str(value).strip()
                ))[:80],
                "claims": list(dict.fromkeys(
                    str(value)[:800] for item in summaries for value in item.get("claims", [])
                    if str(value).strip()
                ))[:80],
                "unresolved_questions": list(dict.fromkeys(
                    str(value)[:800] for item in summaries for value in item.get("uncertainties", [])
                    if str(value).strip()
                ))[:40],
                "contradictions": list(dict.fromkeys(
                    str(value)[:800] for item in summaries for value in item.get("contradictions", [])
                    if str(value).strip()
                ))[:40],
                "evidence_ids": list(dict.fromkeys(
                    evidence_id for item in summaries for evidence_id in item.get("evidence_ids", [])
                    if isinstance(evidence_id, str)
                )),
                "deliverable_artifact_ids": [
                    str(item.get("deliverable_artifact_id")) for item in deduplicated
                    if item.get("deliverable_artifact_id")
                ],
                "coverage": self._merge_coverage([item.get("coverage", {}) for item in summaries]),
            }
            reduced["artifact_id"] = self.artifacts.put(
                "reduction", json.dumps(reduced, ensure_ascii=False), {"mode": "deterministic_deliverable_ledger"}
            )
            return {
                "reduced": reduced,
                "trace": [{
                    "node": "reducer", "role": "确定性产物账本", "status": "completed",
                    "detail": "完整正文不进入 Reducer 提示词；仅合并有界摘要、矛盾和 Artifact ID",
                }],
            }
        current = [self._finding_summary(item) for item in deduplicated]
        trace: list[dict[str, Any]] = []
        level = 0
        while len(current) > 1:
            level += 1
            next_level: list[dict[str, Any]] = []
            for offset in range(0, len(current), self.reduce_fan_in):
                group = current[offset : offset + self.reduce_fan_in]
                messages = [
                    {"role": "system", "content": (
                        "ROLE:REDUCER。合并一小组动态 Worker 的结构化结论，不读取完整原文。"
                        "不得删除输入中的事实、未解决项、矛盾或覆盖状态。"
                        "返回JSON：{\"summary\":...,\"facts\":[...],\"claims\":[...],"
                        "\"unresolved_questions\":[...],\"contradictions\":[...],"
                        "\"evidence_ids\":[...]}。"
                    )},
                    {"role": "user", "content": json.dumps(group, ensure_ascii=False)},
                ]
                parsed, _ = self._call_json(
                    "reducer",
                    messages,
                    max_tokens=1_800,
                    required_keys=("summary", "evidence_ids"),
                )
                input_facts = list(dict.fromkeys(
                    str(value)[:800]
                    for item in group
                    for value in [*item.get("facts", []), *item.get("claims", [])]
                    if str(value).strip()
                ))
                input_claims = list(dict.fromkeys(
                    str(value)[:800]
                    for item in group
                    for value in item.get("claims", [])
                    if str(value).strip()
                ))
                input_evidence = list(dict.fromkeys(
                    evidence_id
                    for item in group
                    for evidence_id in item.get("evidence_ids", [])
                    if isinstance(evidence_id, str)
                ))
                input_unresolved = list(dict.fromkeys(
                    str(value)[:800]
                    for item in group
                    for value in [*item.get("uncertainties", []), *item.get("unresolved_questions", [])]
                    if str(value).strip()
                ))
                input_contradictions = list(dict.fromkeys(
                    str(value)[:800]
                    for item in group
                    for value in item.get("contradictions", [])
                    if str(value).strip()
                ))
                reduced = {
                    "summary": str(parsed.get("summary") or "；".join(item["summary"] for item in group))[:3_000],
                    "facts": input_facts[:80],
                    "claims": list(dict.fromkeys([
                        *input_claims,
                        *[str(item)[:800] for item in parsed.get("claims", []) if str(item).strip()],
                    ]))[:80],
                    "unresolved_questions": list(dict.fromkeys([
                        *input_unresolved,
                        *[str(item)[:800] for item in parsed.get("unresolved_questions", []) if str(item).strip()],
                    ]))[:40],
                    "contradictions": list(dict.fromkeys([
                        *input_contradictions,
                        *[str(item)[:800] for item in parsed.get("contradictions", []) if str(item).strip()],
                    ]))[:40],
                    # Evidence closure is deterministic: the model cannot add or silently drop IDs.
                    "evidence_ids": input_evidence,
                    "coverage": self._merge_coverage([item.get("coverage", {}) for item in group]),
                }
                reduced["artifact_id"] = self.artifacts.put(
                    "reduction", json.dumps(reduced, ensure_ascii=False), {"level": level}
                )
                next_level.append(reduced)
            trace.append({
                "node": "reducer", "role": "reducer", "status": "completed",
                "detail": f"第 {level} 层：{len(current)} 个输入归并为 {len(next_level)} 个结果",
            })
            current = next_level
        reduced = current[0] if current else {
            "summary": "没有动态 Agent 结论。", "facts": [], "claims": [],
            "unresolved_questions": [], "contradictions": [], "evidence_ids": [],
            "coverage": {},
        }
        return {"reduced": reduced, "trace": trace}

    def _validator(self, state: GraphState) -> dict[str, Any]:
        tasks = state.get("tasks", [])
        findings = self._deduplicate_findings(state.get("findings", []))
        hard = self._hard_validation(tasks, findings, state.get("reduced", {}))
        full_artifact = bool(hard.get("deliverable", {}).get("required"))
        if full_artifact:
            control_tasks = [task for task in tasks if not task.get("include_in_final")]
            control_findings = [item for item in findings if not item.get("include_in_final")]
            task_payload = [
                *self._bounded_task_summaries(control_tasks),
                {
                    "task_id": "deliverable_parts_aggregate",
                    "objective": "完整产物的全部有序正文部分",
                    "output_mode": "artifact",
                    "part_count": hard["deliverable"].get("expected_parts", 0),
                    "target_characters": hard["deliverable"].get("target_characters", 0),
                    "actual_characters": hard["deliverable"].get("actual_characters", 0),
                },
            ]
            finding_payload = self._bounded_finding_summaries(
                control_findings, total_character_budget=24_000
            )
            hard_payload = dict(hard)
            hard_payload["deliverable"] = {
                key: value for key, value in hard.get("deliverable", {}).items()
                if key != "parts"
            }
        else:
            task_payload = self._bounded_task_summaries(tasks)
            finding_payload = self._bounded_finding_summaries(findings)
            hard_payload = hard
        messages = [
            {"role": "system", "content": (
                "ROLE:VALIDATOR。你只做语义质量检查，不能改变工作流或放宽程序规则。"
                "没有外部资料的通用推理任务可以不含 evidence_id；只有 source_policy=required 的任务必须有证据。"
                "必须逐项检查 required_targets；当结论声称未找到、不存在或无法提供时，"
                "只有 coverage.percent=100 且全部分片完成才允许通过。"
                "返回JSON：{\"semantic_pass\":true/false,\"missing_task_ids\":[...],"
                "\"missing_required_targets\":[...],\"unsupported_negative_claims\":[...],"
                "\"contradictions\":[...],\"notes\":...}。"
            )},
            {"role": "user", "content": json.dumps({
                "tasks": task_payload,
                "findings": finding_payload,
                "hard_validation": hard_payload,
                "coverage": hard.get("coverage", {}),
            }, ensure_ascii=False)},
        ]
        parsed, response_valid = self._call_json(
            "validator",
            messages,
            max_tokens=1_000,
            required_keys=("semantic_pass", "missing_task_ids", "contradictions"),
        )
        known_ids = {task["task_id"] for task in tasks}
        semantic_missing = [
            item for item in parsed.get("missing_task_ids", [])
            if isinstance(item, str) and item in known_ids
        ]
        contradictions = [str(item)[:600] for item in parsed.get("contradictions", [])]
        missing_targets = [str(item)[:300] for item in parsed.get("missing_required_targets", []) if str(item).strip()]
        unsupported_negatives = [str(item)[:600] for item in parsed.get("unsupported_negative_claims", []) if str(item).strip()]
        semantic_pass = (
            bool(parsed.get("semantic_pass", False))
            and not semantic_missing
            and not contradictions
            and not missing_targets
            and not unsupported_negatives
        )
        semantic_failure_codes: list[str] = []
        if not response_valid:
            semantic_failure_codes.append("validator_invalid_json")
        if semantic_missing:
            semantic_failure_codes.append("validator_missing_tasks")
        if contradictions:
            semantic_failure_codes.append("validator_contradictions")
        if missing_targets:
            semantic_failure_codes.append("missing_required_entity")
        if unsupported_negatives:
            semantic_failure_codes.append("unsupported_negative_claim")
        if response_valid and not semantic_pass and not semantic_missing and not contradictions:
            semantic_failure_codes.append("validator_semantic_rejected")
        retryable = sorted(set(
            hard["missing_task_ids"]
            + hard.get("retryable_task_ids", [])
            + semantic_missing
        ))
        approved = bool(hard["passed"] and semantic_pass)
        validation = {
            "approved": approved,
            "decision_source": "deterministic_policy",
            "hard_checks": hard,
            "semantic_checks": {
                "passed": semantic_pass,
                "response_valid": response_valid,
                "failure_codes": semantic_failure_codes,
                "missing_task_ids": semantic_missing,
                "contradictions": contradictions,
                "missing_required_targets": missing_targets,
                "unsupported_negative_claims": unsupported_negatives,
                "notes": str(parsed.get("notes") or (
                    "Validator 未返回可解析的 JSON。"
                    if not response_valid else "语义检查未返回有效说明。"
                ))[:1_000],
            },
            "retryable_task_ids": retryable,
            "iteration": state.get("iteration", 0),
        }
        return {
            "validation": validation,
            "trace": [{
                "node": "validator", "role": "Validator", "status": "passed" if approved else "rejected",
                "detail": f"硬校验={'通过' if hard['passed'] else '失败'}；语义校验={'通过' if semantic_pass else '失败'}；可重试 {len(retryable)} 项",
            }],
        }

    def _hard_validation(
        self,
        tasks: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        reduced: dict[str, Any],
    ) -> dict[str, Any]:
        task_ids = [str(task.get("task_id", "")) for task in tasks]
        known = set(task_ids)
        tasks_by_id = {str(task.get("task_id", "")): task for task in tasks}
        findings_by_task = {str(item.get("task_id", "")): item for item in findings}
        missing = sorted(known - set(findings_by_task))
        invalid: list[str] = []
        for task_id, finding in findings_by_task.items():
            if task_id not in known:
                invalid.append(f"unknown_task:{task_id}")
                continue
            task = tasks_by_id.get(task_id, {})
            if not str(finding.get("agent_name", "")).strip():
                invalid.append(f"missing_dynamic_agent_name:{task_id}")
            if not str(task.get("agent_instruction", "")).strip():
                invalid.append(f"missing_dynamic_agent_instruction:{task_id}")
            if not str(finding.get("summary", "")).strip():
                invalid.append(f"empty_summary:{task_id}")
            evidence_ids = finding.get("evidence_ids", [])
            if task.get("source_policy") == "required" and not evidence_ids:
                missing.append(task_id)
            coverage = finding.get("coverage", {})
            if task.get("coverage_required") and not coverage.get("complete", False):
                invalid.append(f"insufficient_coverage:{task_id}")
                missing.append(task_id)
            if (
                finding.get("negative_claim")
                and int(coverage.get("candidate_matches", 0)) > 0
                and not finding.get("facts")
            ):
                invalid.append(f"candidate_ignored:{task_id}")
                missing.append(task_id)
            for evidence_id in evidence_ids:
                artifact = self.artifacts.get(str(evidence_id))
                if artifact is None or artifact.kind != "evidence":
                    invalid.append(f"invalid_evidence:{task_id}")
            artifact = self.artifacts.get(str(finding.get("artifact_id", "")))
            if artifact is None or artifact.kind != "finding":
                invalid.append(f"invalid_finding_artifact:{task_id}")
        reduced_ids = set(reduced.get("evidence_ids", []))
        finding_ids = {item for finding in findings for item in finding.get("evidence_ids", [])}
        if not reduced_ids.issubset(finding_ids):
            invalid.append("reducer_introduced_unknown_evidence")
        within_budget = all(
            int(call.get("accounted_total_tokens") or (
                int(call.get("actual_prompt_tokens") or call["estimated_prompt_tokens"])
                + int(call.get("reserved_output_tokens", OUTPUT_RESERVE))
                + SAFETY_MARGIN
            )) <= self.context_limit
            for call in self._calls
        )
        if not within_budget:
            invalid.append("context_budget_exceeded")
        coverage_report = self._build_coverage_report(tasks, findings)
        if coverage_report["required"] and not coverage_report["complete"]:
            invalid.append("insufficient_coverage")
        unsupported_negative = bool(
            coverage_report["negative_claim_count"]
            and coverage_report["coverage_percent"] < 100
        )
        if unsupported_negative:
            invalid.append("unsupported_negative_claim")
        unresolved_targets = [
            item["target"] for item in coverage_report.get("targets", [])
            if item.get("status") == "unresolved"
        ]
        if unresolved_targets:
            invalid.append("missing_required_entity")
        deliverable = self._build_deliverable_report(tasks, findings)
        if deliverable["required"] and not deliverable["complete"]:
            invalid.extend(
                f"deliverable_{code}"
                for code in deliverable.get("failure_codes", [])
            )
            missing.extend(deliverable.get("retryable_task_ids", []))
        missing = sorted(set(missing))
        checks = {
            "unique_task_ids": len(task_ids) == len(set(task_ids)) and all(task_ids),
            "all_tasks_have_findings": not missing,
            "artifacts_and_evidence_valid": not invalid,
            "reducer_evidence_closed": "reducer_introduced_unknown_evidence" not in invalid,
            "all_calls_within_context_limit": within_budget,
            "source_coverage_complete": not coverage_report["required"] or coverage_report["complete"],
            "negative_claims_supported": not unsupported_negative,
            "required_targets_resolved": not unresolved_targets,
            "deliverable_complete": not deliverable["required"] or deliverable["complete"],
        }
        return {
            "passed": all(bool(value) for value in checks.values()),
            "checks": checks,
            "missing_task_ids": missing,
            "violations": invalid,
            "retryable_task_ids": sorted(set([*missing, *deliverable.get("retryable_task_ids", [])])),
            "unresolved_targets": unresolved_targets,
            "coverage": coverage_report,
            "deliverable": deliverable,
        }

    def _build_deliverable_report(
        self,
        tasks: list[dict[str, Any]],
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        artifact_tasks = [
            task for task in tasks
            if task.get("output_mode") == "artifact" and task.get("include_in_final")
        ]
        if not artifact_tasks:
            return {
                "required": False, "complete": True, "response_mode": "synthesis",
                "expected_parts": 0, "completed_parts": 0, "target_characters": 0,
                "actual_characters": 0, "parts": [], "failure_codes": [],
                "retryable_task_ids": [],
            }
        findings_by_task = {str(item.get("task_id", "")): item for item in findings}
        parts: list[dict[str, Any]] = []
        retryable: list[str] = []
        failure_codes: list[str] = []
        seen_sequences: set[int] = set()
        for task in sorted(artifact_tasks, key=lambda item: int(item.get("sequence", 0) or 0)):
            task_id = str(task.get("task_id", ""))
            finding = findings_by_task.get(task_id, {})
            artifact_id = str(finding.get("deliverable_artifact_id", ""))
            artifact = self.artifacts.get(artifact_id)
            sequence = int(task.get("sequence", 0) or 0)
            target = int(task.get("target_characters", 0) or 0)
            actual = len(artifact.content) if artifact and artifact.kind == "deliverable" else 0
            valid = bool(artifact and artifact.kind == "deliverable" and artifact.content.strip())
            short = bool(valid and target and actual < target)
            if not valid:
                failure_codes.append("artifact_missing")
                retryable.append(task_id)
            if short:
                failure_codes.append("part_too_short")
                retryable.append(task_id)
            if sequence <= 0 or sequence in seen_sequences:
                failure_codes.append("sequence_invalid")
            seen_sequences.add(sequence)
            parts.append({
                "task_id": task_id,
                "sequence": sequence,
                "label": str(task.get("artifact_label", "")),
                "artifact_id": artifact_id,
                "target_characters": target,
                "actual_characters": actual,
                "valid": valid,
                "short": short,
            })
        target_total = int(self._delivery_contract.get("target_characters", 0) or 0)
        actual_total = sum(item["actual_characters"] for item in parts)
        if target_total and actual_total < target_total:
            failure_codes.append("total_too_short")
            retryable.extend(
                item["task_id"] for item in parts
                if item["actual_characters"] < item["target_characters"]
            )
        expected_sequences = set(range(1, len(artifact_tasks) + 1))
        if seen_sequences != expected_sequences:
            failure_codes.append("sequence_incomplete")
        failure_codes = list(dict.fromkeys(failure_codes))
        retryable = list(dict.fromkeys(item for item in retryable if item))
        return {
            "required": True,
            "complete": not failure_codes,
            "response_mode": "full_artifact",
            "artifact_type": self._delivery_contract.get("artifact_type", "通用内容"),
            "expected_parts": len(artifact_tasks),
            "completed_parts": sum(item["valid"] for item in parts),
            "target_characters": target_total,
            "actual_characters": actual_total,
            "assembled_outside_model_window": True,
            "summary_may_replace_deliverable": False,
            "parts": parts,
            "failure_codes": failure_codes,
            "retryable_task_ids": retryable,
        }

    def _build_coverage_report(
        self,
        tasks: list[dict[str, Any]],
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        required_tasks = [task for task in tasks if task.get("coverage_required")]
        findings_by_task = {str(item.get("task_id", "")): item for item in findings}
        covered_indices: set[int] = set()
        shards: list[dict[str, Any]] = []
        target_candidates: dict[str, int] = {}
        target_support: dict[str, int] = {}
        negative_count = 0
        for task in required_tasks:
            finding = findings_by_task.get(str(task.get("task_id", "")), {})
            coverage = finding.get("coverage", {})
            complete = bool(coverage.get("complete", False))
            start = int(task.get("chunk_start", 0) or 0)
            end = int(task.get("chunk_end", start) or start)
            if complete:
                covered_indices.update(range(start, end))
            negative = bool(finding.get("negative_claim"))
            negative_count += int(negative)
            facts = [str(item) for item in finding.get("facts", []) if str(item).strip()]
            for target, count in coverage.get("target_candidates", {}).items():
                target_candidates[str(target)] = target_candidates.get(str(target), 0) + int(count or 0)
                if int(count or 0) > 0 and facts and not negative:
                    target_support[str(target)] = target_support.get(str(target), 0) + 1
            shards.append({
                "task_id": task.get("task_id"),
                "shard_index": int(task.get("shard_index", 1) or 1),
                "shard_count": int(task.get("shard_count", 1) or 1),
                "complete": complete,
                "assigned_chunks": int(task.get("shard_chunks", 0) or 0),
                "scanned_chunks": int(coverage.get("scanned_chunks", 0) or 0),
                "assigned_tokens": int(task.get("shard_tokens", 0) or 0),
                "packed_tokens": int(coverage.get("packed_tokens", 0) or 0),
                "candidate_matches": int(coverage.get("candidate_matches", 0) or 0),
            })
        total_chunks = len(self.index.chunks) if required_tasks else 0
        percent = round(len(covered_indices) / max(1, total_chunks) * 100, 2) if required_tasks else 100.0
        targets = []
        for target in self._allocation.get("required_targets", []):
            candidates = target_candidates.get(str(target), 0)
            supporting = target_support.get(str(target), 0)
            if supporting:
                status = "resolved"
            elif candidates and percent >= 100:
                status = "unresolved"
            elif percent >= 100:
                status = "not_found_after_full_coverage"
            else:
                status = "coverage_incomplete"
            targets.append({
                "target": str(target),
                "candidate_matches": candidates,
                "supporting_findings": supporting,
                "status": status,
            })
        return {
            "required": bool(required_tasks),
            "complete": bool(not required_tasks or (percent >= 100 and all(item["complete"] for item in shards))),
            "coverage_percent": percent,
            "document_chunks": total_chunks,
            "covered_chunks": len(covered_indices),
            "completed_shards": sum(item["complete"] for item in shards),
            "total_shards": len(shards),
            "negative_claim_count": negative_count,
            "targets": targets,
            "shards": sorted(shards, key=lambda item: item["shard_index"]),
        }

    def _route_after_validation(self, state: GraphState) -> str:
        validation = state.get("validation", {})
        if validation.get("approved"):
            return "finalizer"
        retryable = validation.get("retryable_task_ids", [])
        if retryable and state.get("iteration", 0) < self.max_replans:
            return "supervisor_replan"
        return "finalizer"

    def _supervisor_replan(self, state: GraphState) -> dict[str, Any]:
        retry_ids = set(state.get("validation", {}).get("retryable_task_ids", []))
        active = [task for task in state.get("tasks", []) if task["task_id"] in retry_ids]
        iteration = state.get("iteration", 0) + 1
        return {
            "active_tasks": active,
            "iteration": iteration,
            "trace": [{
                "node": "supervisor_replan", "role": "主 Agent", "status": "scheduled",
                "detail": f"确定性外循环第 {iteration} 次，仅重新调度 {len(active)} 个失败任务",
            }],
        }

    @staticmethod
    def _dispatch_after_replan(state: GraphState):
        active = state.get("active_tasks", [])
        if not active:
            return "finalizer"
        return MultiAgentResearchSystem._dispatch_active_tasks(state)

    def _finalize(self, state: GraphState) -> dict[str, Any]:
        validation = state.get("validation", {})
        deliverable = validation.get("hard_checks", {}).get("deliverable", {})
        if deliverable.get("required"):
            findings_by_task = {
                str(item.get("task_id", "")): item
                for item in self._deduplicate_findings(state.get("findings", []))
            }
            assembled: list[str] = []
            for part in sorted(deliverable.get("parts", []), key=lambda item: int(item.get("sequence", 0))):
                finding = findings_by_task.get(str(part.get("task_id", "")), {})
                artifact = self.artifacts.get(str(finding.get("deliverable_artifact_id", "")))
                if artifact is None or artifact.kind != "deliverable" or not artifact.content.strip():
                    continue
                content = artifact.content.strip()
                label = str(part.get("label", "")).strip()
                if label and label not in content[:160]:
                    content = f"## {label}\n\n{content}"
                assembled.append(content)
            answer = "\n\n".join(assembled).strip()
            if not validation.get("approved"):
                failures = "、".join(deliverable.get("failure_codes", [])) or "语义校验未通过"
                answer = (
                    f"> 交付校验未完全通过（{failures}）。以下仍返回已生成的全部原文，未用摘要替代。\n\n"
                    f"{answer}"
                ).strip()
            return {
                "answer": answer or "没有生成可交付的正文。",
                "stop_reason": "validated_full_artifact" if validation.get("approved") else "partial_artifact_returned",
                "trace": [{
                    "node": "finalizer", "role": "确定性产物组装器", "status": "completed",
                    "detail": (
                        f"按顺序无损组装 {len(assembled)}/{deliverable.get('expected_parts', 0)} 个正文 Artifact；"
                        f"最终返回 {len(answer)} 字符；未调用模型概括或改写正文"
                    ),
                }],
            }
        messages = [
            {"role": "system", "content": (
                "ROLE:FINALIZER。根据动态 Agent 的归并结果和 Validator 报告直接回答用户问题。"
                "对于带 evidence_id 的外部事实必须忠于证据；对于 source_policy=none 的任务可以使用"
                "模型通用知识。不要把没有外部来源的通用答案错误描述为资料缺失。"
                "必须逐项回答 required_targets；如果 Validator 标记覆盖不完整、目标未解决或负向结论"
                "缺乏全覆盖证明，必须明确说明未通过原因，不能声称资料中不存在。"
                "验证未通过时说明具体缺口后，仍应回答现有结果能够支持的部分。"
            )},
            {"role": "user", "content": (
                (
                    f"有界对话记忆（仅用于理解指代）：\n{self._conversation_context}\n\n"
                    if self._conversation_context else ""
                )
                + f"用户问题：{state['question']}\n\n归并结论："
                f"{json.dumps(state.get('reduced', {}), ensure_ascii=False)}\n\nValidator："
                f"{json.dumps(validation, ensure_ascii=False)}"
            )},
        ]
        answer = self._call("finalizer", messages, max_tokens=2_000).strip()
        return {
            "answer": answer,
            "stop_reason": "validated" if validation.get("approved") else "validation_failed_closed",
            "trace": [{
                "node": "finalizer", "role": "Finalizer", "status": "completed",
                "detail": "汇总动态 Agent 结果；外部事实受证据约束，通用任务保留模型原生能力",
            }],
        }

    def _call(self, role: str, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        counter = getattr(self.model, "count_messages_tokens", None)
        if callable(counter):
            preflight = int(counter(messages))
            preflight_source = "model_tokenizer"
        else:
            preflight = estimate_messages_tokens(messages)
            preflight_source = "conservative_estimate"
        reserved_output = max(OUTPUT_RESERVE, int(max_tokens))
        role_budget = min(self.ROLE_BUDGETS[role], self.context_limit - reserved_output - SAFETY_MARGIN)
        if preflight > role_budget:
            raise ValueError(f"{role} Agent 输入约 {preflight} Token，超过其 {role_budget} Token 独立预算")
        if getattr(self.model, "supports_parallel_requests", False):
            response = self.model.chat(messages, temperature=0.1, max_tokens=max_tokens)
            usage = getattr(self.model, "last_usage", None)
        else:
            with self._model_lock:
                response = self.model.chat(messages, temperature=0.1, max_tokens=max_tokens)
                usage = getattr(self.model, "last_usage", None)
        actual = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
        measured_prompt = int(actual if actual is not None else preflight)
        with self._record_lock:
            self._calls.append({
                "role": role,
                "estimated_prompt_tokens": preflight,
                "actual_prompt_tokens": actual,
                "accounting_source": "api" if actual is not None else "estimate",
                "preflight_source": preflight_source,
                "role_budget": role_budget,
                "requested_output_tokens": int(max_tokens),
                "reserved_output_tokens": reserved_output,
                "actual_completion_tokens": completion,
                "safety_margin_tokens": SAFETY_MARGIN,
                "accounted_total_tokens": measured_prompt + reserved_output + SAFETY_MARGIN,
                "window_utilization_percent": round(
                    (measured_prompt + reserved_output + SAFETY_MARGIN) / self.context_limit * 100,
                    2,
                ),
            })
        return response

    def _call_json(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        required_keys: tuple[str, ...],
    ) -> tuple[dict[str, Any], bool]:
        """Parse structured output and make one bounded repair attempt when necessary."""
        raw = self._call(role, messages, max_tokens=max_tokens)
        parsed = parse_json_object(raw)
        if parsed is not None and all(key in parsed for key in required_keys):
            return parsed, True

        with self._record_lock:
            self._json_retries += 1
        original_request = messages[-1]["content"]
        retry_messages = [
            messages[0],
            {
                "role": "user",
                "content": (
                    f"{original_request}\n\n"
                    "上次响应无法通过 JSON 契约校验。请重新完成同一任务，只返回一个 JSON 对象，"
                    f"必须包含字段：{', '.join(required_keys)}。不要使用 Markdown。\n"
                    f"上次响应片段：{raw[:1_000]}"
                ),
            },
        ]
        repaired_raw = self._call(role, retry_messages, max_tokens=max_tokens)
        repaired = parse_json_object(repaired_raw)
        valid = repaired is not None and all(key in repaired for key in required_keys)
        return (repaired or {}), valid

    def _context_metrics(
        self,
        tasks: list[dict[str, Any]],
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        def measured(call: dict[str, Any]) -> int:
            return int(call.get("actual_prompt_tokens") or call["estimated_prompt_tokens"])

        max_prompt = max((measured(call) for call in self._calls), default=0)
        max_accounted_total = max(
            (int(call.get("accounted_total_tokens", 0)) for call in self._calls),
            default=0,
        )
        by_role: dict[str, dict[str, int]] = {}
        for call in self._calls:
            bucket = by_role.setdefault(call["role"], {
                "calls": 0, "max_prompt_tokens": 0, "max_accounted_total_tokens": 0,
            })
            bucket["calls"] += 1
            bucket["max_prompt_tokens"] = max(bucket["max_prompt_tokens"], measured(call))
            bucket["max_accounted_total_tokens"] = max(
                bucket["max_accounted_total_tokens"], int(call.get("accounted_total_tokens", 0)),
            )
        dynamic_agent_counts: dict[str, int] = {}
        for task in tasks:
            agent_name = str(task.get("agent_name") or "未命名 Agent")
            dynamic_agent_counts[agent_name] = dynamic_agent_counts.get(agent_name, 0) + 1
        return {
            "architecture": "LangGraph deterministic loop + Dynamic Agent Factory + Validator",
            "control_plane": "deterministic_code",
            "context_limit_tokens": self.context_limit,
            "hard_safe_input_tokens": self.context_limit - OUTPUT_RESERVE - SAFETY_MARGIN,
            "max_single_agent_prompt_tokens": max_prompt,
            "max_window_utilization_percent": round(max_prompt / self.context_limit * 100, 2),
            "max_accounted_total_tokens": max_accounted_total,
            "max_total_window_utilization_percent": round(
                max_accounted_total / self.context_limit * 100, 2
            ),
            "all_agent_calls_within_limit": all(
                int(call.get("accounted_total_tokens") or (
                    measured(call) + int(call.get("reserved_output_tokens", OUTPUT_RESERVE)) + SAFETY_MARGIN
                )) <= self.context_limit
                for call in self._calls
            ),
            "isolated_worker_contexts": True,
            "supervisor_received_raw_document": False,
            "task_count": len(tasks),
            "dynamic_agent_counts": dynamic_agent_counts,
            "generated_agent_profiles": [
                {
                    "task_id": task.get("task_id"),
                    "agent_name": task.get("agent_name"),
                    "capabilities": task.get("capabilities", []),
                    "tools": task.get("tools", []),
                    "source_policy": task.get("source_policy", "none"),
                }
                for task in tasks
            ],
            "model_calls": len(self._calls),
            "planning_source": self._planning_source,
            "intent": self._intent,
            "agent_strategy": self._agent_strategy,
            "runtime_settings": {
                "default_agents": self.default_agents,
                "max_agents": self.max_workers,
                "parallel_workers": self.parallel_workers,
                "reduce_fan_in": self.reduce_fan_in,
                "max_replans": self.max_replans,
                "max_artifact_output_tokens": self.max_artifact_output_tokens,
                "target_artifact_characters": self.target_artifact_characters,
            },
            "available_tools": sorted(self.available_tools),
            "structured_output_retries": self._json_retries,
            "retrieval_strategies": sorted(self._retrieval_strategies),
            "agent_allocation": dict(self._allocation),
            "dependency_phases": max((int(task.get("phase", 0)) for task in tasks), default=0) + 1,
            "coverage_report": self._build_coverage_report(tasks, findings),
            "delivery_contract": dict(self._delivery_contract),
            "deliverable_report": self._build_deliverable_report(tasks, findings),
            "token_accounting": {
                "mode": (
                    "api_usage" if self._calls and all(call.get("actual_prompt_tokens") is not None for call in self._calls)
                    else "mixed" if any(call.get("actual_prompt_tokens") is not None for call in self._calls)
                    else "estimated"
                ),
                "api_measured_calls": sum(call.get("actual_prompt_tokens") is not None for call in self._calls),
                "estimated_calls": sum(call.get("actual_prompt_tokens") is None for call in self._calls),
                "output_reserve_tokens": OUTPUT_RESERVE,
                "safety_margin_tokens": SAFETY_MARGIN,
            },
            "conversation_memory": {
                "enabled": bool(self._conversation_context),
                "characters": len(self._conversation_context),
                "estimated_tokens": estimate_tokens(self._conversation_context),
                "stored_outside_model_window": True,
            },
            "by_role": by_role,
            "calls": self._calls,
        }

    def _citation(self, artifact_id: str) -> dict[str, Any] | None:
        artifact = self.artifacts.get(artifact_id)
        if artifact is None or artifact.kind != "evidence":
            return None
        return {
            "artifact_id": artifact_id,
            "chunk_id": artifact.metadata.get("chunk_id"),
            "document": artifact.metadata.get("document"),
            "section": artifact.metadata.get("section"),
            "excerpt": artifact.content[:500],
            "document_bytes": artifact.metadata.get("document_bytes", 0),
            "indexed_bytes": artifact.metadata.get("indexed_bytes", 0),
            "chunk_bytes": artifact.metadata.get("chunk_bytes", len(artifact.content.encode("utf-8"))),
            "source_exceeds_shard_limit": artifact.metadata.get("source_exceeds_shard_limit", False),
            "shard_byte_limit": artifact.metadata.get("shard_byte_limit", self.shard_byte_limit),
            "task_id": artifact.metadata.get("task_id"),
            "agent_name": artifact.metadata.get("agent_name"),
            "agent_instance_id": artifact.metadata.get("agent_instance_id"),
            "shard_index": artifact.metadata.get("shard_index", 1),
            "shard_count": artifact.metadata.get("shard_count", 1),
            "shard_bytes": artifact.metadata.get("shard_bytes", 0),
        }

    @staticmethod
    def _finding_summary(finding: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": finding.get("task_id"),
            "agent_name": finding.get("agent_name"),
            "capabilities": finding.get("capabilities", [])[:8],
            "tools": finding.get("tools", [])[:8],
            "source_policy": finding.get("source_policy", "none"),
            "summary": str(finding.get("summary", ""))[:1_500],
            "facts": finding.get("facts", [])[:12],
            "claims": finding.get("claims", [])[:8],
            "uncertainties": finding.get("uncertainties", [])[:8],
            "contradictions": finding.get("contradictions", [])[:8],
            "evidence_ids": finding.get("evidence_ids", [])[:12],
            "depends_on": finding.get("depends_on", [])[:16],
            "phase": finding.get("phase", 0),
            "required_targets": finding.get("required_targets", [])[:16],
            "output_mode": finding.get("output_mode", "finding"),
            "include_in_final": bool(finding.get("include_in_final", False)),
            "sequence": int(finding.get("sequence", 0) or 0),
            "artifact_label": str(finding.get("artifact_label", ""))[:120],
            "deliverable_artifact_id": finding.get("deliverable_artifact_id", ""),
            "target_characters": int(finding.get("target_characters", 0) or 0),
            "deliverable_characters": int(finding.get("deliverable_characters", 0) or 0),
            "coverage": {
                key: value for key, value in finding.get("coverage", {}).items()
                if key != "packed_chunk_ids"
            },
        }

    @staticmethod
    def _bounded_finding_summaries(
        findings: list[dict[str, Any]],
        *,
        total_character_budget: int = 36_000,
    ) -> list[dict[str, Any]]:
        """Keep high-fan-in dependency and validation prompts below one model window."""
        if not findings:
            return []
        per_item = max(120, min(1_500, total_character_budget // len(findings)))
        summaries: list[dict[str, Any]] = []
        for finding in findings:
            item = MultiAgentResearchSystem._finding_summary(finding)
            item["summary"] = str(item.get("summary", ""))[:per_item]
            item["facts"] = [str(value)[:240] for value in item.get("facts", [])[:3]]
            item["claims"] = [str(value)[:240] for value in item.get("claims", [])[:3]]
            item["uncertainties"] = [str(value)[:200] for value in item.get("uncertainties", [])[:2]]
            item["contradictions"] = [str(value)[:200] for value in item.get("contradictions", [])[:2]]
            summaries.append(item)
        return summaries

    @staticmethod
    def _bounded_task_summaries(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{
            "task_id": task.get("task_id"),
            "objective": str(task.get("objective", ""))[:180],
            "agent_name": str(task.get("agent_name", ""))[:80],
            "source_policy": task.get("source_policy", "none"),
            "depends_on": task.get("depends_on", [])[:128],
            "phase": task.get("phase", 0),
            "output_mode": task.get("output_mode", "finding"),
            "include_in_final": bool(task.get("include_in_final", False)),
            "sequence": int(task.get("sequence", 0) or 0),
            "target_characters": int(task.get("target_characters", 0) or 0),
            "artifact_label": str(task.get("artifact_label", ""))[:100],
        } for task in tasks]

    @staticmethod
    def _merge_coverage(items: list[dict[str, Any]]) -> dict[str, Any]:
        required_items = [item for item in items if item.get("required")]
        if not required_items:
            return {"required": False, "complete": True, "coverage_percent": 100.0}
        assigned = sum(int(item.get("assigned_chunks", 0) or 0) for item in required_items)
        scanned = sum(int(item.get("scanned_chunks", 0) or 0) for item in required_items)
        return {
            "required": True,
            "complete": all(item.get("complete", False) for item in required_items),
            "coverage_percent": round(scanned / max(1, assigned) * 100, 2),
            "assigned_chunks": assigned,
            "scanned_chunks": scanned,
        }

    @staticmethod
    def _looks_negative(text: str) -> bool:
        compact = re.sub(r"\s+", "", str(text).casefold())
        return any(term in compact for term in (
            "未找到", "不存在", "不包含", "没有发现", "无法提供", "无法确定", "缺少相关",
            "notfound", "doesnotcontain", "cannotprovide", "noevidence",
        ))

    @staticmethod
    def _deduplicate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for finding in findings:
            task_id = str(finding.get("task_id", ""))
            if task_id:
                latest[task_id] = finding
        return list(latest.values())

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5
