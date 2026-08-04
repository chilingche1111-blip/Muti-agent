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
EXHAUSTIVE_SCAN_TERMS = ("全部", "所有", "完整清单", "逐条", "逐项", "员工信息", "员工名单")
SOURCE_POLICIES = {"none", "optional", "required"}
BUILTIN_TOOLS = {"model_reasoning", "source_search", "web_sources"}


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


class DynamicWorkerState(TypedDict):
    question: str
    task: dict[str, Any]


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
        available_tools: set[str] | None = None,
    ) -> None:
        if context_limit < 8_000:
            raise ValueError("模型上下文上限不能低于 8,000 Token")
        if max_workers < 1 or max_workers > 32:
            raise ValueError("专业子 Agent 数量必须在 1至32 之间")
        if default_agents < 1 or default_agents > 32:
            raise ValueError("默认动态 Agent 数量必须在 1至32 之间")
        if shard_byte_limit < 16 * 1024 or shard_byte_limit > 4 * 1024 * 1024:
            raise ValueError("单 Agent 分片阈值必须在 16 KB至4 MB 之间")
        if reduce_fan_in < 2 or reduce_fan_in > 8:
            raise ValueError("Reducer 扇入必须在 2至8 之间")
        if max_replans < 0 or max_replans > 3:
            raise ValueError("确定性外循环重规划次数必须在 0至3 之间")
        self.model = model
        self.index = index
        self.artifacts = artifact_store or ArtifactStore()
        self.context_limit = context_limit
        self.max_workers = max_workers
        self.default_agents = min(default_agents, max_workers)
        self.shard_byte_limit = shard_byte_limit
        self.reduce_fan_in = reduce_fan_in
        self.max_replans = max_replans
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
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("supervisor", self._supervisor)
        graph.add_node("worker", self._dynamic_worker)
        graph.add_node("reducer", self._reduce_tree)
        graph.add_node("validator", self._validator)
        graph.add_node("supervisor_replan", self._supervisor_replan)
        graph.add_node("finalizer", self._finalize)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", self._dispatch_active_tasks, ["worker"])
        graph.add_edge("worker", "reducer")
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
        self.artifacts.clear()
        state = self._graph.invoke(
            {"question": question.strip(), "findings": [], "trace": [], "iteration": 0},
            config={"configurable": {"thread_id": uuid.uuid4().hex}},
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
            context_metrics=self._context_metrics(state.get("tasks", [])),
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
                    "返回JSON：{\"problem_type\":...,\"strategy\":...,\"tasks\":[{"
                    "\"objective\":...,\"query\":...,\"agent_name\":...,"
                    "\"agent_instruction\":...,\"capabilities\":[...],\"tools\":[...],"
                    "\"source_policy\":...,\"priority\":1到100}]}。"
                    f"最多{self.max_workers}项。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"可用工具：{json.dumps(sorted(self.available_tools), ensure_ascii=False)}\n"
                    f"资料元数据：{json.dumps(source_stats, ensure_ascii=False)}\n"
                    f"默认并行下限：{self.default_agents}\n"
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
        tasks = self._normalize_tasks(parsed.get("tasks"), question)
        planning_source = "dynamic_model_supervisor"
        intent = str(parsed.get("problem_type") or "general_problem")[:120]
        strategy = str(parsed.get("strategy") or "动态生成执行 Agent 并按子任务并行求解")[:500]
        tasks = self._allocate_agent_tasks(tasks, question)
        self._planning_source = planning_source
        self._intent = intent
        self._agent_strategy = strategy
        generated_names = list(dict.fromkeys(task["agent_name"] for task in tasks))
        return {
            "tasks": tasks,
            "active_tasks": tasks,
            "iteration": 0,
            "planning_source": planning_source,
            "intent": intent,
            "agent_allocation": dict(self._allocation),
            "trace": [{
                "node": "supervisor",
                "role": "主 Agent",
                "status": "scheduled",
                "detail": (
                    f"问题类型={intent}；策略={strategy}；"
                    f"运行时生成 {len(generated_names)} 种 Agent 规格，"
                    f"分配 {self._allocation.get('allocated_agents', len(tasks))} 个隔离实例"
                ),
                "task_ids": [task["task_id"] for task in tasks],
                "generated_agents": generated_names,
            }],
        }

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
            )]
        return [task.as_dict() for task in sorted(tasks, key=lambda item: -item.priority)]

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
        """Scale worker count from source size and task complexity, then assign bounded shards."""
        chunks = self.index.all_chunks()
        indexed_bytes = self.index.total_indexed_bytes
        compact = re.sub(r"\s+", "", question)
        exhaustive_scan = any(term in compact for term in EXHAUSTIVE_SCAN_TERMS)
        target_shard_bytes = self.shard_byte_limit // 2 if exhaustive_scan else self.shard_byte_limit
        byte_required = max(1, math.ceil(indexed_bytes / target_shard_bytes))
        chunk_required = max(1, math.ceil(len(chunks) / TARGET_CHUNKS_PER_AGENT))
        complexity_markers = sum(
            compact.count(marker)
            for marker in ("同时", "分别", "逐项", "以及", "并且", "、", "；", "比较", "风险")
        )
        complexity_required = min(self.max_workers, 1 + complexity_markers + len(compact) // 500)
        desired = max(
            self.default_agents,
            len(base_tasks),
            byte_required,
            chunk_required,
            complexity_required,
        )
        allocated = min(self.max_workers, desired)

        reasons: list[str] = [f"默认至少 {self.default_agents} 个"]
        if byte_required > self.default_agents:
            reasons.append(
                f"索引文本 {indexed_bytes} Bytes 按 {target_shard_bytes} Bytes 目标分片需要 {byte_required} 个"
            )
        if chunk_required > self.default_agents:
            reasons.append(f"{len(chunks)} 个检索块需要 {chunk_required} 个")
        if complexity_required > self.default_agents:
            reasons.append(f"任务复杂度需要 {complexity_required} 个")
        if desired > self.max_workers:
            reasons.append(f"受最大 Worker 数 {self.max_workers} 限制")

        allocated_tasks: list[dict[str, Any]] = []
        chunk_count = len(chunks)
        for slot in range(allocated):
            template = dict(base_tasks[slot % len(base_tasks)])
            if chunk_count == 0:
                chunk_start = chunk_end = 0
            elif allocated <= chunk_count:
                chunk_start = slot * chunk_count // allocated
                chunk_end = (slot + 1) * chunk_count // allocated
            else:
                chunk_start = slot % chunk_count
                chunk_end = chunk_start + 1
            shard_chunks = chunks[chunk_start:chunk_end]
            combined_query = str(template.get("query", "")).strip()
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
            })
            allocated_tasks.append(template)

        self._allocation = {
            "default_agents": self.default_agents,
            "desired_agents": desired,
            "allocated_agents": allocated,
            "max_agents": self.max_workers,
            "source_indexed_bytes": indexed_bytes,
            "largest_source_indexed_bytes": self.index.largest_source_indexed_bytes,
            "shard_byte_limit": self.shard_byte_limit,
            "target_shard_bytes": target_shard_bytes,
            "source_exceeds_shard_limit": self.index.largest_source_indexed_bytes > self.shard_byte_limit,
            "multi_agent_sharding_active": allocated > 1,
            "exhaustive_scan": exhaustive_scan,
            "allocation_reason": "；".join(reasons),
        }
        return allocated_tasks

    @staticmethod
    def _dispatch_active_tasks(state: GraphState):
        return [
            Send("worker", {"question": state["question"], "task": task})
            for task in state.get("active_tasks", [])
        ]

    def _dynamic_worker(self, state: DynamicWorkerState) -> dict[str, Any]:
        task = state["task"]
        hits, retrieval_strategy = self._retrieve_hits(task)
        with self._record_lock:
            self._retrieval_strategies.add(retrieval_strategy)
        # input_budget is the maximum safe prompt budget, not a target. Leave room for
        # the generated AgentSpec, current question and bounded conversation context.
        evidence_budget = int(task.get("input_budget", 61_000)) - 4_000
        evidence_ids, packed = self._pack_evidence(
            hits,
            budget=max(1_000, evidence_budget),
            task=task,
        )
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
                    "返回JSON：{\"summary\":...,\"claims\":[...],\"evidence_ids\":[...],"
                    "\"confidence\":0到1}。不得编造证据ID。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"task_id：{task['task_id']}\n目标：{task['objective']}\n"
                    f"工作查询：{task['query']}\n"
                    + (
                        f"有界对话记忆（只用于理解当前问题，不可作为事实证据）：\n"
                        f"{self._conversation_context}\n"
                        if self._conversation_context else ""
                    )
                    + f"\n本任务可见的外部资料：\n{packed or '未提供外部资料；请按来源策略使用模型能力完成任务。'}"
                ),
            },
        ]
        parsed, _ = self._call_json(
            "worker",
            messages,
            max_tokens=1_800,
            required_keys=("summary", "evidence_ids"),
        )
        valid_ids = [
            item for item in parsed.get("evidence_ids", [])
            if isinstance(item, str) and item in evidence_ids
        ]
        if not valid_ids:
            valid_ids = evidence_ids[:3]
        claims = [str(item)[:800] for item in parsed.get("claims", []) if str(item).strip()]
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
            "claims": claims[:8],
            "evidence_ids": valid_ids,
            "confidence": self._confidence(parsed.get("confidence")),
            "retrieval_strategy": retrieval_strategy,
            "answer_basis": "external_sources" if valid_ids else "model_reasoning",
            "agent_instance_id": task.get("agent_instance_id"),
            "shard_index": task.get("shard_index"),
            "shard_count": task.get("shard_count"),
            "shard_bytes": task.get("shard_bytes"),
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
                    f"引用 {len(valid_ids)} 个证据对象"
                ),
                "artifact_id": finding["artifact_id"],
            }],
        }

    def _retrieve_hits(
        self,
        task: dict[str, Any],
    ) -> tuple[list[SearchHit], str]:
        """Use bounded external context only when the generated Agent requests that tool."""
        if "source_search" not in task.get("tools", []) or not self.index.chunks:
            return [], "model_reasoning_only"
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
        if any(term in compact for term in EXHAUSTIVE_SCAN_TERMS):
            return [
                SearchHit(chunk=chunk, score=1.0, source="shard_exhaustive_scan")
                for chunk in shard_chunks
            ], "sharded_exhaustive_scan"
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
                return lexical_hits, "sharded_hybrid_exact"
            fallback = [
                SearchHit(chunk=chunk, score=0.01, source="shard_fallback")
                for chunk in shard_chunks[:3]
            ]
            return fallback, "sharded_positional_fallback"

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
        return combined[:10], "sharded_hybrid_plus_positional_coverage"

    def _pack_evidence(
        self,
        hits: list[SearchHit],
        *,
        budget: int,
        task: dict[str, Any],
    ) -> tuple[list[str], str]:
        artifact_ids: list[str] = []
        blocks: list[str] = []
        used = 0
        for hit in hits:
            text = hit.chunk.text.strip()
            cost = estimate_tokens(text) + 80
            if blocks and used + cost > budget:
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
            used += cost
        return artifact_ids, "\n\n".join(blocks)

    def _reduce_tree(self, state: GraphState) -> dict[str, Any]:
        current = [self._finding_summary(item) for item in self._deduplicate_findings(state["findings"])]
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
                        "返回JSON：{\"summary\":...,\"claims\":[...],\"evidence_ids\":[...]}。"
                    )},
                    {"role": "user", "content": json.dumps(group, ensure_ascii=False)},
                ]
                parsed, _ = self._call_json(
                    "reducer",
                    messages,
                    max_tokens=1_800,
                    required_keys=("summary", "evidence_ids"),
                )
                reduced = {
                    "summary": str(parsed.get("summary") or "；".join(item["summary"] for item in group))[:3_000],
                    "claims": [str(item)[:800] for item in parsed.get("claims", [])][:20],
                    "evidence_ids": list(dict.fromkeys(
                        [item for item in parsed.get("evidence_ids", []) if isinstance(item, str)]
                        or [evidence_id for item in group for evidence_id in item["evidence_ids"]]
                    )),
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
        reduced = current[0] if current else {"summary": "没有动态 Agent 结论。", "claims": [], "evidence_ids": []}
        return {"reduced": reduced, "trace": trace}

    def _validator(self, state: GraphState) -> dict[str, Any]:
        tasks = state.get("tasks", [])
        findings = self._deduplicate_findings(state.get("findings", []))
        hard = self._hard_validation(tasks, findings, state.get("reduced", {}))
        messages = [
            {"role": "system", "content": (
                "ROLE:VALIDATOR。你只做语义质量检查，不能改变工作流或放宽程序规则。"
                "没有外部资料的通用推理任务可以不含 evidence_id；只有 source_policy=required 的任务必须有证据。"
                "返回JSON：{\"semantic_pass\":true/false,\"missing_task_ids\":[...],"
                "\"contradictions\":[...],\"notes\":...}。"
            )},
            {"role": "user", "content": json.dumps({
                "tasks": tasks,
                "findings": [self._finding_summary(item) for item in findings],
                "hard_validation": hard,
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
        semantic_pass = bool(parsed.get("semantic_pass", False)) and not semantic_missing and not contradictions
        semantic_failure_codes: list[str] = []
        if not response_valid:
            semantic_failure_codes.append("validator_invalid_json")
        if semantic_missing:
            semantic_failure_codes.append("validator_missing_tasks")
        if contradictions:
            semantic_failure_codes.append("validator_contradictions")
        if response_valid and not semantic_pass and not semantic_missing and not contradictions:
            semantic_failure_codes.append("validator_semantic_rejected")
        retryable = sorted(set(hard["missing_task_ids"] + semantic_missing))
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
            int(call.get("actual_prompt_tokens") or call["estimated_prompt_tokens"])
            + OUTPUT_RESERVE + SAFETY_MARGIN <= self.context_limit
            for call in self._calls
        )
        if not within_budget:
            invalid.append("context_budget_exceeded")
        missing = sorted(set(missing))
        checks = {
            "unique_task_ids": len(task_ids) == len(set(task_ids)) and all(task_ids),
            "all_tasks_have_findings": not missing,
            "artifacts_and_evidence_valid": not invalid,
            "reducer_evidence_closed": "reducer_introduced_unknown_evidence" not in invalid,
            "all_calls_within_context_limit": within_budget,
        }
        return {
            "passed": all(bool(value) for value in checks.values()),
            "checks": checks,
            "missing_task_ids": missing,
            "violations": invalid,
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
        return [
            Send("worker", {"question": state["question"], "task": task})
            for task in active
        ]

    def _finalize(self, state: GraphState) -> dict[str, Any]:
        validation = state.get("validation", {})
        messages = [
            {"role": "system", "content": (
                "ROLE:FINALIZER。根据动态 Agent 的归并结果和 Validator 报告直接回答用户问题。"
                "对于带 evidence_id 的外部事实必须忠于证据；对于 source_policy=none 的任务可以使用"
                "模型通用知识。不要把没有外部来源的通用答案错误描述为资料缺失。"
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
        estimated = estimate_messages_tokens(messages)
        role_budget = min(self.ROLE_BUDGETS[role], self.context_limit - OUTPUT_RESERVE - SAFETY_MARGIN)
        if estimated > role_budget:
            raise ValueError(f"{role} Agent 输入约 {estimated} Token，超过其 {role_budget} Token 独立预算")
        with self._model_lock:
            response = self.model.chat(messages, temperature=0.1, max_tokens=max_tokens)
            usage = getattr(self.model, "last_usage", None)
        actual = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        with self._record_lock:
            self._calls.append({
                "role": role,
                "estimated_prompt_tokens": estimated,
                "actual_prompt_tokens": actual,
                "accounting_source": "api" if actual is not None else "estimate",
                "role_budget": role_budget,
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

    def _context_metrics(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        def measured(call: dict[str, Any]) -> int:
            return int(call.get("actual_prompt_tokens") or call["estimated_prompt_tokens"])

        max_prompt = max((measured(call) for call in self._calls), default=0)
        by_role: dict[str, dict[str, int]] = {}
        for call in self._calls:
            bucket = by_role.setdefault(call["role"], {"calls": 0, "max_prompt_tokens": 0})
            bucket["calls"] += 1
            bucket["max_prompt_tokens"] = max(bucket["max_prompt_tokens"], measured(call))
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
            "all_agent_calls_within_limit": all(
                measured(call) + OUTPUT_RESERVE + SAFETY_MARGIN <= self.context_limit
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
            "available_tools": sorted(self.available_tools),
            "structured_output_retries": self._json_retries,
            "retrieval_strategies": sorted(self._retrieval_strategies),
            "agent_allocation": dict(self._allocation),
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
            "claims": finding.get("claims", [])[:8],
            "evidence_ids": finding.get("evidence_ids", [])[:12],
        }

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
