from __future__ import annotations

import json
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

    def test_supervisor_splits_compound_goal_into_isolated_specialists(self) -> None:
        result = MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer(
            "请给出项目代号、验收口令和归档校验值。"
        )
        self.assertEqual(len(result.tasks), 3)
        self.assertEqual(len(result.findings), 3)
        self.assertEqual(sum(item["node"] == "fact_agent" for item in result.trace), 3)
        self.assertTrue(result.context_metrics["isolated_specialist_contexts"])
        self.assertEqual(result.context_metrics["control_plane"], "deterministic_code")
        self.assertTrue(result.context_metrics["all_agent_calls_within_limit"])
        self.assertIn("reducer", result.context_metrics["by_role"])
        self.assertTrue(result.validation["approved"])
        self.assertEqual(result.validation["decision_source"], "deterministic_policy")
        self.assertTrue(result.validation["hard_checks"]["passed"])

    def test_supervisor_never_receives_full_document(self) -> None:
        result = MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer(
            "项目代号是什么？"
        )
        self.assertEqual(result.context_metrics["planning_source"], "deterministic_router")
        self.assertNotIn("supervisor", result.context_metrics["by_role"])
        self.assertLess(result.context_metrics["max_single_agent_prompt_tokens"], 16_000)
        self.assertGreater(self.stats["estimated_tokens"], 65_536)

    def test_summary_retrieval_covers_document_beginning_middle_and_end(self) -> None:
        system = MultiAgentResearchSystem(DeterministicTestModel(), self.index)
        result = system.answer("请总结全文的主要内容。")
        self.assertEqual(result.context_metrics["intent"], "document_summary")
        self.assertIn(
            "hybrid_plus_positional_coverage",
            result.context_metrics["retrieval_strategies"],
        )
        cited_indices = sorted(
            int(str(item["chunk_id"]).split("_")[-1]) for item in result.citations
        )
        self.assertLessEqual(cited_indices[0], 1)
        self.assertGreaterEqual(cited_indices[-1], len(self.index.chunks) - 2)

    def test_invalid_specialist_json_is_repaired_once(self) -> None:
        class RepairableModel:
            def __init__(self) -> None:
                self.delegate = DeterministicTestModel()
                self.failed = False

            def chat(self, messages, *, temperature=0.1, max_tokens=2_000):
                if "ROLE:SPECIALIST:" in messages[0]["content"] and not self.failed:
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

    def test_main_agent_routes_to_allowed_professional_agent(self) -> None:
        result = MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer(
            "请识别资料中的风险和矛盾。"
        )
        self.assertEqual(result.tasks[0]["agent_type"], "risk_reviewer")
        self.assertTrue(any(item["node"] == "risk_agent" for item in result.trace))

    def test_validator_hard_checks_cannot_be_bypassed_by_model(self) -> None:
        system = MultiAgentResearchSystem(DeterministicTestModel(), self.index)
        hard = system._hard_validation(
            [{"task_id": "task_01", "agent_type": "fact_extractor"}], [], {}
        )
        self.assertFalse(hard["passed"])
        self.assertEqual(hard["missing_task_ids"], ["task_01"])

    def test_role_budget_rejects_oversized_supervisor_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "supervisor Agent 输入"):
            MultiAgentResearchSystem(DeterministicTestModel(), self.index).answer("很长的问题" * 10_000)

    def test_full_offline_benchmark_passes(self) -> None:
        report = run_benchmark(self.document_path, self.cases_path, mode="offline")
        self.assertTrue(report["passed"])
        self.assertTrue(report["context_proof"]["divide_and_conquer_verified"])
        self.assertFalse(report["context_proof"]["supervisor_received_raw_document"])
        self.assertTrue(report["context_proof"]["isolated_specialist_contexts"])
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
        self.assertEqual(len(reducer_steps), 2)


if __name__ == "__main__":
    unittest.main()
