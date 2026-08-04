from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from long_context_agent.benchmark import DeterministicTestModel, explain_case_failure, run_benchmark
from long_context_agent.document import DocumentIndex
from long_context_agent.orchestrator import MultiAgentResearchSystem
from scripts.generate_test_document import generate


class MultiAgentContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.document_path, cls.cases_path, cls.stats = generate(cls.output_dir)
        cls.index = DocumentIndex()
        cls.index.add_file(cls.document_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_generated_document_exceeds_64k_estimated_tokens(self) -> None:
        self.assertTrue(self.stats["exceeds_64k_estimated_tokens"])

    def test_supervisor_generates_problem_specific_isolated_agents(self) -> None:
        result = MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer(
            "请给出项目代号、验收口令和归档校验值。"
        )
        allocated = result.context_metrics["agent_allocation"]["allocated_agents"]
        self.assertGreaterEqual(allocated, 3)
        self.assertEqual(len(result.tasks), allocated)
        self.assertEqual(len(result.findings), allocated)
        self.assertEqual(sum(item["node"] == "worker" for item in result.trace), allocated)
        self.assertTrue(result.context_metrics["isolated_worker_contexts"])
        self.assertTrue(all(item.get("agent_name") for item in result.tasks))
        self.assertTrue(all(item.get("agent_instruction") for item in result.tasks))
        self.assertTrue(all("agent_type" not in item for item in result.tasks))
        self.assertEqual(result.context_metrics["control_plane"], "deterministic_code")
        self.assertTrue(result.context_metrics["all_agent_calls_within_limit"])
        self.assertTrue(all(call["role_budget"] == 61_000 for call in result.context_metrics["calls"]))
        self.assertIn("reducer", result.context_metrics["by_role"])
        self.assertTrue(result.validation["approved"])
        self.assertEqual(result.validation["decision_source"], "deterministic_policy")
        self.assertTrue(result.validation["hard_checks"]["passed"])

    def test_supervisor_never_receives_full_document(self) -> None:
        result = MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer(
            "项目代号是什么？"
        )
        self.assertEqual(result.context_metrics["planning_source"], "dynamic_model_supervisor")
        self.assertIn("supervisor", result.context_metrics["by_role"])
        self.assertLess(result.context_metrics["max_single_agent_prompt_tokens"], 61_000)
        self.assertGreater(self.stats["estimated_tokens"], 65_536)

    def test_summary_retrieval_covers_document_beginning_middle_and_end(self) -> None:
        system = MultiAgentResearchSystem(DeterministicTestModel(), self.index)
        result = system.answer("请总结全文的主要内容。")
        self.assertEqual(result.context_metrics["intent"], "offline_dynamic_plan")
        self.assertIn(
            "sharded_full_coverage",
            result.context_metrics["retrieval_strategies"],
        )
        self.assertTrue(result.context_metrics["coverage_report"]["complete"])
        self.assertEqual(result.context_metrics["coverage_report"]["coverage_percent"], 100.0)
        coverage = result.context_metrics["coverage_report"]
        self.assertEqual(coverage["covered_chunks"], len(self.index.chunks))
        self.assertEqual(coverage["completed_shards"], coverage["total_shards"])

    def test_invalid_dynamic_worker_json_is_repaired_once(self) -> None:
        class RepairableModel:
            def __init__(self) -> None:
                self.delegate = DeterministicTestModel()
                self.failed = False

            def chat(self, messages, *, temperature=0.1, max_tokens=2_000):
                if "ROLE:DYNAMIC_WORKER" in messages[0]["content"] and not self.failed:
                    self.failed = True
                    return "not-json"
                return self.delegate.chat(messages, temperature=temperature, max_tokens=max_tokens)

        result = MultiAgentResearchSystem(RepairableModel(), self.index).answer("项目代号是什么？")
        self.assertTrue(result.validation["approved"])
        self.assertEqual(result.context_metrics["structured_output_retries"], 1)
        self.assertIn("青峦-7429", result.answer)

    def test_workers_return_artifact_ids_instead_of_raw_source(self) -> None:
        result = MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer(
            "验收口令是什么？"
        )
        self.assertTrue(result.findings[0]["artifact_id"].startswith("finding_"))
        self.assertTrue(all(item.startswith("evidence_") for item in result.findings[0]["evidence_ids"]))
        self.assertTrue(result.citations)

    def test_small_task_still_uses_default_agent_floor(self) -> None:
        index = DocumentIndex()
        index.add_text("small.txt", "项目代号是青峦-7429。")
        result = MultiAgentResearchSystem(
            DeterministicTestModel(),
            index,
            default_agents=3,
            max_workers=8,
        ).answer("项目代号是什么？")
        allocation = result.context_metrics["agent_allocation"]
        self.assertEqual(allocation["default_agents"], 3)
        self.assertEqual(allocation["allocated_agents"], 3)
        self.assertEqual(len(result.tasks), 3)
        self.assertTrue(all(task["input_budget"] == 61_000 for task in result.tasks))
        self.assertTrue(result.validation["approved"])

    def test_large_source_scales_agents_and_exposes_byte_metadata(self) -> None:
        index = DocumentIndex()
        text = "项目代号是青峦-7429。\n" + "员工信息记录与组织字段。" * 20_000
        index.add_text("employees.txt", text, source_bytes=len(text.encode("utf-8")))
        result = MultiAgentResearchSystem(
            DeterministicTestModel(),
            index,
            default_agents=3,
            max_workers=8,
        ).answer("项目代号是什么？")
        allocation = result.context_metrics["agent_allocation"]
        self.assertTrue(allocation["source_exceeds_shard_limit"])
        self.assertGreater(allocation["allocated_agents"], 3)
        self.assertLessEqual(allocation["allocated_agents"], 8)
        self.assertTrue(result.context_metrics["all_agent_calls_within_limit"])
        self.assertTrue(result.citations)
        self.assertTrue(all(int(item["document_bytes"]) > 0 for item in result.citations))
        self.assertTrue(all(int(item["chunk_bytes"]) > 0 for item in result.citations))
        self.assertTrue(any(int(item["shard_count"]) > 1 for item in result.citations))

    def test_exhaustive_employee_scan_survives_sharded_reduction(self) -> None:
        class EmployeeModel:
            last_usage = None

            def chat(self, messages, *, temperature=0.1, max_tokens=2_000):
                del temperature, max_tokens
                system = messages[0]["content"]
                prompt = messages[-1]["content"]
                employee_ids = list(dict.fromkeys(re.findall(r"EMP-\d{4}", prompt)))
                evidence_ids = list(dict.fromkeys(re.findall(r"evidence_[a-f0-9]+", prompt)))
                if "ROLE:SUPERVISOR" in system:
                    return json.dumps({
                        "problem_type": "employee_inventory",
                        "strategy": "按资料规模动态扩展员工清单 Agent",
                        "tasks": [{
                            "objective": "完整列出所有员工编号",
                            "query": "所有员工信息 完整清单",
                            "agent_name": "员工清单覆盖 Agent",
                            "agent_instruction": "扫描分配到的输入范围并保留全部员工编号。",
                            "capabilities": ["完整扫描", "清单抽取"],
                            "tools": ["model_reasoning", "source_search"],
                            "source_policy": "required",
                            "priority": 90,
                        }],
                    }, ensure_ascii=False)
                if "ROLE:DYNAMIC_WORKER" in system:
                    return json.dumps({
                        "summary": "；".join(employee_ids),
                        "claims": employee_ids,
                        "evidence_ids": evidence_ids,
                        "confidence": 1.0,
                    }, ensure_ascii=False)
                if "ROLE:REDUCER" in system:
                    return json.dumps({
                        "summary": "；".join(employee_ids),
                        "claims": employee_ids,
                        "evidence_ids": evidence_ids,
                    }, ensure_ascii=False)
                if "ROLE:VALIDATOR" in system:
                    payload = json.loads(prompt)
                    passed = bool(payload.get("hard_validation", {}).get("passed"))
                    return json.dumps({
                        "semantic_pass": passed,
                        "missing_task_ids": [],
                        "contradictions": [],
                        "notes": "员工分片结果完整进入验证阶段",
                    }, ensure_ascii=False)
                if "ROLE:FINALIZER" in system:
                    return "；".join(employee_ids)
                raise AssertionError("unexpected role")

        records = [
            f"## 员工 {index:04d}\n\n员工编号 EMP-{index:04d}，在职状态正常。\n" + "岗位资料。" * 260
            for index in range(1, 61)
        ]
        text = "\n\n".join(records)
        index = DocumentIndex()
        index.add_text("employees.md", text, source_bytes=len(text.encode("utf-8")))
        result = MultiAgentResearchSystem(
            EmployeeModel(),
            index,
            default_agents=3,
            max_workers=16,
            reduce_fan_in=4,
        ).answer("所有员工信息有哪些？请形成完整清单。")
        allocation = result.context_metrics["agent_allocation"]
        self.assertTrue(allocation["exhaustive_scan"])
        self.assertGreater(allocation["allocated_agents"], 3)
        self.assertIn("sharded_full_coverage", result.context_metrics["retrieval_strategies"])
        self.assertIn("EMP-0001", result.answer)
        self.assertIn("EMP-0060", result.answer)
        self.assertTrue(result.validation["approved"])

    def test_main_agent_generates_problem_specific_agent_profile(self) -> None:
        result = MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer(
            "请识别资料中的风险和矛盾。"
        )
        self.assertEqual(result.tasks[0]["agent_name"], "矛盾核查 Agent")
        self.assertIn("自检", result.tasks[0]["capabilities"])
        self.assertTrue(any(item["node"] == "worker" for item in result.trace))

    def test_dynamic_agents_solve_general_problem_without_retrieval(self) -> None:
        class GeneralProblemModel:
            last_usage = None

            def chat(self, messages, *, temperature=0.1, max_tokens=2_000):
                del temperature, max_tokens
                system = messages[0]["content"]
                prompt = messages[-1]["content"]
                if "ROLE:SUPERVISOR" in system:
                    tasks = [
                        {
                            "objective": objective,
                            "query": objective,
                            "agent_name": name,
                            "agent_instruction": instruction,
                            "capabilities": capabilities,
                            "tools": ["model_reasoning"],
                            "source_policy": "none",
                            "priority": priority,
                        }
                        for name, objective, instruction, capabilities, priority in (
                            ("架构构思 Agent", "提出可维护的插件系统架构", "从模块边界和扩展点构造方案。", ["architecture_design"], 90),
                            ("失效模式 Agent", "识别插件系统的主要失败模式", "从隔离、兼容性和恢复角度审查。", ["failure_analysis"], 85),
                            ("落地计划 Agent", "形成分阶段实施计划", "把方案转换为可执行步骤。", ["implementation_planning"], 80),
                        )
                    ]
                    return json.dumps({"problem_type": "software_design", "strategy": "并行设计、审查和落地", "tasks": tasks}, ensure_ascii=False)
                if "ROLE:DYNAMIC_WORKER" in system:
                    name = re.search(r"Agent 名称：([^。]+)", system).group(1)
                    return json.dumps({"summary": f"{name}已完成独立分析", "claims": [f"{name}结论"], "evidence_ids": [], "confidence": 0.9}, ensure_ascii=False)
                if "ROLE:REDUCER" in system:
                    return json.dumps({"summary": "形成插件架构、失效防护和实施路线", "claims": ["分层插件接口", "故障隔离", "分阶段交付"], "evidence_ids": []}, ensure_ascii=False)
                if "ROLE:VALIDATOR" in system:
                    hard_passed = json.loads(prompt)["hard_validation"]["passed"]
                    return json.dumps({"semantic_pass": hard_passed, "missing_task_ids": [], "contradictions": [], "notes": "通用任务覆盖完整"}, ensure_ascii=False)
                if "ROLE:FINALIZER" in system:
                    return "建议采用分层插件接口、故障隔离和分阶段交付。"
                raise AssertionError("unexpected role")

        result = MultiAgentResearchSystem(
            GeneralProblemModel(),
            DocumentIndex(),
            default_agents=3,
        ).answer("设计一个可扩展、可维护的插件系统，并给出风险和落地计划。")
        self.assertTrue(result.validation["approved"])
        self.assertEqual(len(result.tasks), 3)
        self.assertTrue(all(task["source_policy"] == "none" for task in result.tasks))
        self.assertTrue(all(task["tools"] == ["model_reasoning"] for task in result.tasks))
        self.assertFalse(result.citations)
        self.assertIn("分层插件接口", result.answer)
        self.assertEqual(result.context_metrics["retrieval_strategies"], ["model_reasoning_only"])

    def test_loaded_document_does_not_scale_source_free_agents(self) -> None:
        class SourceFreeModel(DeterministicTestModel):
            def chat(self, messages, *, temperature=0.1, max_tokens=2_000):
                if "ROLE:SUPERVISOR" in messages[0]["content"]:
                    return json.dumps({
                        "problem_type": "creative_writing",
                        "strategy": "多视角写作",
                        "tasks": [{
                            "objective": "撰写产品介绍",
                            "query": "撰写产品介绍",
                            "agent_name": "产品文案 Agent",
                            "agent_instruction": "使用模型原生能力完成文案。",
                            "capabilities": ["writing"],
                            "tools": ["model_reasoning"],
                            "source_policy": "none",
                            "priority": 80,
                            "depends_on": [],
                        }],
                    }, ensure_ascii=False)
                return super().chat(messages, temperature=temperature, max_tokens=max_tokens)

        result = MultiAgentResearchSystem(
            SourceFreeModel(), self.index, default_agents=3, max_workers=16,
        ).answer("写一段产品介绍。")
        allocation = result.context_metrics["agent_allocation"]
        self.assertFalse(allocation["source_scaling_active"])
        self.assertEqual(allocation["allocated_agents"], 3)
        self.assertFalse(result.context_metrics["coverage_report"]["required"])

    def test_dependency_graph_runs_ready_tasks_in_waves(self) -> None:
        class DependencyModel:
            last_usage = None

            def chat(self, messages, *, temperature=0.1, max_tokens=2_000):
                system = messages[0]["content"]
                prompt = messages[-1]["content"]
                if "ROLE:SUPERVISOR" in system:
                    return json.dumps({
                        "problem_type": "novel",
                        "strategy": "先大纲、后章节、再一致性检查",
                        "tasks": [
                            {"objective": "生成总纲", "query": "总纲", "agent_name": "总纲 Agent", "agent_instruction": "生成总纲", "capabilities": ["outline"], "tools": ["model_reasoning"], "source_policy": "none", "priority": 90, "depends_on": []},
                            {"objective": "撰写第一章", "query": "第一章", "agent_name": "章节 Agent", "agent_instruction": "依据总纲写第一章", "capabilities": ["writing"], "tools": ["model_reasoning"], "source_policy": "none", "priority": 80, "depends_on": ["task_01"]},
                            {"objective": "一致性检查", "query": "一致性", "agent_name": "一致性 Agent", "agent_instruction": "检查总纲和章节", "capabilities": ["consistency"], "tools": ["model_reasoning"], "source_policy": "none", "priority": 70, "depends_on": ["task_02"]},
                        ],
                    }, ensure_ascii=False)
                if "ROLE:DYNAMIC_WORKER" in system:
                    dependency_seen = "依赖任务的结构化结果" in prompt
                    return json.dumps({"summary": "完成", "facts": ["依赖已读取" if dependency_seen else "根任务"], "claims": ["完成"], "uncertainties": [], "contradictions": [], "evidence_ids": [], "confidence": 1.0}, ensure_ascii=False)
                if "ROLE:REDUCER" in system:
                    return json.dumps({"summary": "小说任务完成", "claims": [], "evidence_ids": []}, ensure_ascii=False)
                if "ROLE:VALIDATOR" in system:
                    return json.dumps({"semantic_pass": True, "missing_task_ids": [], "contradictions": [], "notes": "依赖链完整"}, ensure_ascii=False)
                if "ROLE:FINALIZER" in system:
                    return "小说任务完成"
                raise AssertionError("unexpected role")

        result = MultiAgentResearchSystem(
            DependencyModel(), DocumentIndex(), default_agents=1, max_workers=8,
        ).answer("写一章小说并检查一致性")
        scheduler_steps = [item for item in result.trace if item["node"] == "dependency_scheduler"]
        self.assertGreaterEqual(len(scheduler_steps), 3)
        self.assertEqual([task["phase"] for task in result.tasks], [0, 1, 2])
        self.assertTrue(result.validation["approved"])

    def test_complete_deliverable_can_exceed_64k_without_being_summarized(self) -> None:
        class FullArtifactModel:
            last_usage = None

            def __init__(self) -> None:
                self.finalizer_called = False

            def chat(self, messages, *, temperature=0.1, max_tokens=2_000):
                del temperature, max_tokens
                system = messages[0]["content"]
                prompt = messages[-1]["content"]
                if "ROLE:SUPERVISOR" in system:
                    return json.dumps({
                        "problem_type": "complete_report",
                        "strategy": "先规划，再分段生成完整产物",
                        "tasks": [{
                            "objective": "规划报告结构",
                            "query": "生成三部分报告结构",
                            "agent_name": "当前报告结构 Agent",
                            "agent_instruction": "形成三部分结构和衔接约束",
                            "capabilities": ["structure"],
                            "tools": ["model_reasoning"],
                            "source_policy": "none",
                            "priority": 100,
                            "depends_on": [],
                        }],
                    }, ensure_ascii=False)
                if "本次响应会被程序作为最终产物原文保存" in system:
                    label = re.search(r"当前部分：([^\n]+)", prompt).group(1)
                    index = int(re.search(r"第(\d+)部分", label).group(1))
                    return f"## {label}\n\nPART-{index:03d}|" + "正文" * 650
                if "ROLE:DYNAMIC_WORKER" in system:
                    return json.dumps({
                        "summary": "结构或审校任务已形成有效结果",
                        "facts": [], "claims": [], "uncertainties": [],
                        "contradictions": [], "evidence_ids": [], "confidence": 1.0,
                    }, ensure_ascii=False)
                if "ROLE:REDUCER" in system:
                    return json.dumps({
                        "summary": "只归并调度摘要，不接触完整正文",
                        "claims": [], "evidence_ids": [],
                    }, ensure_ascii=False)
                if "ROLE:VALIDATOR" in system:
                    hard_passed = json.loads(prompt)["hard_validation"]["passed"]
                    return json.dumps({
                        "semantic_pass": hard_passed,
                        "missing_task_ids": [], "contradictions": [],
                        "notes": "完整产物满足交付契约",
                    }, ensure_ascii=False)
                if "ROLE:FINALIZER" in system:
                    self.finalizer_called = True
                    return "错误：正文被概括"
                raise AssertionError("unexpected role")

        model = FullArtifactModel()
        result = MultiAgentResearchSystem(
            model, DocumentIndex(), default_agents=3, max_workers=110,
            target_artifact_characters=1_200,
        ).answer("请生成一份120000字、100部分的完整报告，不要概括，只输出全文。")
        self.assertTrue(result.validation["approved"])
        self.assertFalse(model.finalizer_called)
        self.assertGreaterEqual(len(result.answer), 120_000)
        self.assertIn("PART-001", result.answer)
        self.assertIn("PART-050", result.answer)
        self.assertIn("PART-100", result.answer)
        self.assertNotIn("错误：正文被概括", result.answer)
        report = result.context_metrics["deliverable_report"]
        self.assertTrue(report["complete"])
        self.assertEqual(report["completed_parts"], 100)
        self.assertGreater(report["actual_characters"], 64_000)
        self.assertTrue(report["assembled_outside_model_window"])
        self.assertFalse(report["summary_may_replace_deliverable"])
        self.assertEqual(result.stop_reason, "validated_full_artifact")

    def test_delivery_mode_follows_question_intent_and_explicit_user_preference(self) -> None:
        system = MultiAgentResearchSystem(
            DeterministicTestModel(), DocumentIndex(), default_agents=3, max_workers=16,
        )

        summary = system._infer_delivery_contract("请总结这部小说的人物关系。", "")
        summary = system._resolve_delivery_contract(summary, {
            "response_mode": "full_artifact", "artifact_type": "小说",
        })
        self.assertEqual(summary["response_mode"], "synthesis")
        self.assertEqual(summary["decision_source"], "explicit_user_summary")

        full = system._infer_delivery_contract("请写出完整小说全文，不要概括。", "")
        full = system._resolve_delivery_contract(full, {"response_mode": "synthesis"})
        self.assertEqual(full["response_mode"], "full_artifact")
        self.assertEqual(full["decision_source"], "explicit_user_full")

        automatic = system._infer_delivery_contract("为管理层准备季度复盘材料。", "")
        automatic = system._resolve_delivery_contract(automatic, {
            "response_mode": "full_artifact",
            "artifact_type": "季度复盘报告",
            "target_characters": 6000,
            "target_parts": 4,
        })
        self.assertEqual(automatic["response_mode"], "full_artifact")
        self.assertEqual(automatic["decision_source"], "dynamic_supervisor")
        self.assertEqual(automatic["target_parts"], 4)

        analysis = system._infer_delivery_contract("分析这部小说的叙事结构。", "")
        analysis = system._resolve_delivery_contract(analysis, {"response_mode": "synthesis"})
        self.assertEqual(analysis["response_mode"], "synthesis")

    def test_negative_claim_requires_complete_coverage(self) -> None:
        system = MultiAgentResearchSystem(DeterministicTestModel(), self.index)
        task = {
            "task_id": "task_01", "agent_name": "核查 Agent", "agent_instruction": "完整核查",
            "source_policy": "required", "coverage_required": True,
            "chunk_start": 0, "chunk_end": len(self.index.chunks), "shard_chunks": len(self.index.chunks),
        }
        finding = {
            "task_id": "task_01", "agent_name": "核查 Agent", "summary": "未找到目标值",
            "facts": [], "claims": [], "evidence_ids": [], "negative_claim": True,
            "coverage": {"required": True, "complete": False, "scanned_chunks": 3, "assigned_chunks": len(self.index.chunks)},
        }
        hard = system._hard_validation([task], [finding], {"evidence_ids": []})
        self.assertFalse(hard["passed"])
        self.assertFalse(hard["checks"]["source_coverage_complete"])
        self.assertFalse(hard["checks"]["negative_claims_supported"])
        self.assertIn("unsupported_negative_claim", hard["violations"])

    def test_context_accounting_exposes_reserve_and_total(self) -> None:
        result = MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer("项目代号是什么？")
        accounting = result.context_metrics["token_accounting"]
        self.assertEqual(accounting["output_reserve_tokens"], 2_000)
        self.assertEqual(accounting["safety_margin_tokens"], 1_000)
        self.assertTrue(all("accounted_total_tokens" in call for call in result.context_metrics["calls"]))
        self.assertLessEqual(result.context_metrics["max_accounted_total_tokens"], 64_000)

    def test_validator_hard_checks_cannot_be_bypassed_by_model(self) -> None:
        system = MultiAgentResearchSystem(DeterministicTestModel(), self.index)
        hard = system._hard_validation(
            [{
                "task_id": "task_01",
                "agent_name": "临时核查 Agent",
                "agent_instruction": "核查当前目标",
                "source_policy": "required",
            }], [], {}
        )
        self.assertFalse(hard["passed"])
        self.assertEqual(hard["missing_task_ids"], ["task_01"])

    def test_full_window_budget_rejects_oversized_supervisor_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "supervisor Agent 输入"):
            MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer("很长的问题" * 20_000)

    def test_full_offline_benchmark_passes(self) -> None:
        report = run_benchmark(self.document_path, self.cases_path, mode="offline")
        self.assertTrue(report["passed"])
        self.assertTrue(report["context_proof"]["divide_and_conquer_verified"])
        self.assertFalse(report["context_proof"]["supervisor_received_raw_document"])
        self.assertTrue(report["context_proof"]["isolated_worker_contexts"])
        self.assertEqual(report["context_proof"]["passed_cases"], 4)
        self.assertEqual(len(report["assertions"]), 6)
        self.assertTrue(all(item["passed"] for item in report["assertions"]))
        self.assertEqual(report["configuration"]["context_limit_tokens"], 64_000)
        self.assertGreaterEqual(report["duration_ms"], 0)

    def test_failure_reason_contract_explains_validator_rejection(self) -> None:
        reasons = explain_case_failure({
            "citations": [{"artifact_id": "evidence_1"}],
            "validation": {
                "approved": False,
                "hard_checks": {
                    "checks": {
                        "unique_task_ids": True,
                        "all_tasks_have_findings": True,
                        "artifacts_and_evidence_valid": True,
                        "reducer_evidence_closed": True,
                        "all_calls_within_context_limit": True,
                    },
                    "missing_task_ids": [],
                    "violations": [],
                },
                "semantic_checks": {
                    "passed": False,
                    "response_valid": False,
                    "failure_codes": ["validator_invalid_json"],
                    "missing_task_ids": [],
                    "contradictions": [],
                    "notes": "Validator 未返回可解析的 JSON。",
                },
            },
        }, ["银杏-5813"])
        self.assertEqual(reasons[0]["code"], "expected_terms_missing")
        self.assertEqual(reasons[1]["code"], "validator_invalid_json")
        self.assertIn("未按要求返回有效 JSON", reasons[1]["message"])
        self.assertTrue(reasons[1]["retryable"])

    def test_all_expected_facts_survive_tree_reduction(self) -> None:
        suite = json.loads(self.cases_path.read_text(encoding="utf-8"))
        case = next(item for item in suite["cases"] if item["id"] == "multi_hop_all_positions")
        result = MultiAgentResearchSystem(DeterministicTestModel(), self.index, reduce_fan_in=2).answer(case["question"])
        for expected in case["expected_terms"]:
            self.assertIn(expected, result.answer)
        reducer_steps = [item for item in result.trace if item["node"] == "reducer"]
        self.assertGreaterEqual(len(reducer_steps), 2)


if __name__ == "__main__":
    unittest.main()
