from __future__ import annotations

import unittest
from unittest.mock import patch

from long_context_agent.web_server import AuthManager, ResearchApplication, VerifiedAPIProfiles, WEB_ROOT, settings_from_payload


class WebApplicationTests(unittest.TestCase):
    def test_static_ui_assets_exist(self) -> None:
        for name in ("index.html", "styles.css", "app.js", "test.html", "test.js"):
            self.assertTrue((WEB_ROOT / name).is_file())
        test_html = (WEB_ROOT / "test.html").read_text(encoding="utf-8")
        index_html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        test_js = (WEB_ROOT / "test.js").read_text(encoding="utf-8")
        self.assertNotIn('id="test-api-key"', test_html)
        self.assertIn("api_profile_token", test_js)
        self.assertNotIn('data-mode="offline"', index_html)
        self.assertIn("流程演示与验收", index_html)

    def test_tester_login_uses_role_session_and_logout(self) -> None:
        auth = AuthManager("tester", "123456", session_ttl=60)
        self.assertIsNone(auth.login("tester", "wrong"))
        token, session = auth.login("tester", "123456") or ("", {})
        self.assertEqual(session["role"], "tester")
        self.assertEqual(auth.resolve(token)["username"], "tester")
        auth.logout(token)
        self.assertIsNone(auth.resolve(token))

    def test_api_profile_validation(self) -> None:
        settings = settings_from_payload({"base_url": "https://example.internal/v1/", "api_key": "unit-test-key", "model": "qwen", "timeout_seconds": 30})
        self.assertEqual(settings.base_url, "https://example.internal/v1")
        with self.assertRaises(ValueError):
            settings_from_payload({"base_url": "invalid", "api_key": "x", "model": "qwen"})

    def test_verified_api_profile_uses_random_expiring_handle(self) -> None:
        profiles = VerifiedAPIProfiles(ttl_seconds=60)
        settings = settings_from_payload({"base_url": "https://example.internal/v1", "api_key": "secret", "model": "qwen"})
        public = profiles.register(settings)
        self.assertNotIn("api_key", public)
        self.assertEqual(public["model"], "qwen")
        self.assertIs(profiles.resolve(public["profile_token"]), settings)
        self.assertIsNone(profiles.resolve("unknown"))

    def test_offline_workbench_runs_multi_agent_graph(self) -> None:
        application = ResearchApplication()
        document = application.load_bundled_test()
        self.assertTrue(document["exceeds_64k_tokens"])
        result = application.answer({"mode": "offline", "question": "请给出项目代号、验收口令和归档校验值。", "max_workers": 8, "reduce_fan_in": 2})
        self.assertEqual(len(result["tasks"]), 3)
        self.assertTrue(result["validation"]["approved"])
        self.assertEqual(result["context_metrics"]["control_plane"], "deterministic_code")
        self.assertTrue(result["capacity_report"]["divide_and_conquer_verified"])
        self.assertFalse(result["capacity_report"]["supervisor_received_raw_document"])
        self.assertTrue(result["context_metrics"]["all_agent_calls_within_limit"])

    def test_general_chat_works_without_a_document(self) -> None:
        class FakeClient:
            def __init__(self, settings) -> None:
                self.settings = settings
                self.last_usage = {"prompt_tokens": 42}

            def chat(self, messages, **kwargs) -> str:
                self.asserted_messages = messages
                return "这是底层 LLM 的直接回答。"

        application = ResearchApplication()
        with patch("long_context_agent.web_server.OpenAICompatibleClient", FakeClient):
            result = application.answer({
                "mode": "live",
                "answer_scope": "general",
                "question": "你好，请介绍你的能力。",
                "history": [{"role": "user", "content": "上一轮问题"}],
                "api": {"base_url": "https://example.internal/v1", "api_key": "secret", "model": "qwen"},
            })
        self.assertEqual(result["execution_mode"], "general_chat")
        self.assertEqual(result["answer"], "这是底层 LLM 的直接回答。")
        self.assertEqual(result["context_metrics"]["model_calls"], 1)
        self.assertFalse(result["citations"])

    def test_general_chat_can_include_web_sources(self) -> None:
        from long_context_agent.web_search import WebSearchResult

        class FakeClient:
            def __init__(self, settings) -> None:
                self.last_usage = {"prompt_tokens": 80}

            def chat(self, messages, **kwargs) -> str:
                self.messages = messages
                return "根据联网结果回答。[网页1]"

        sources = [WebSearchResult("官方来源", "https://example.com", "最新资料摘要")]
        application = ResearchApplication()
        with (
            patch("long_context_agent.web_server.OpenAICompatibleClient", FakeClient),
            patch("long_context_agent.web_server.search_web", return_value=sources),
        ):
            result = application.answer({
                "mode": "live",
                "answer_scope": "general",
                "web_search": True,
                "question": "请查询最新资料",
                "api": {"base_url": "https://example.internal/v1", "api_key": "secret", "model": "qwen"},
            })
        self.assertEqual(result["citations"][0]["url"], "https://example.com")
        self.assertTrue(result["context_metrics"]["web_search_enabled"])
        self.assertEqual(result["web_sources"][0]["title"], "官方来源")

    def test_explicit_page_range_bypasses_semantic_retrieval(self) -> None:
        application = ResearchApplication()
        application.load_text(
            "twenty-pages.pdf",
            "\n\n".join(f"## 第 {page} 页\n\n这是第 {page} 页的唯一内容。" for page in range(1, 21)),
        )
        result = application.answer({
            "mode": "live",
            "answer_scope": "document",
            "question": "请输出第16-20页的内容。",
        })
        self.assertEqual(result["stop_reason"], "direct_page_read")
        self.assertIn("这是第 16 页的唯一内容", result["answer"])
        self.assertIn("这是第 20 页的唯一内容", result["answer"])
        self.assertNotIn("这是第 15 页的唯一内容", result["answer"])
        self.assertEqual(result["capacity_report"]["model_calls"], 0)

    def test_benchmark_accepts_controlled_graph_configuration(self) -> None:
        application = ResearchApplication()
        report = application.benchmark({
            "mode": "offline",
            "max_workers": 6,
            "reduce_fan_in": 2,
            "max_replans": 0,
        })
        self.assertTrue(report["passed"])
        self.assertEqual(report["configuration"]["max_workers"], 6)
        self.assertEqual(report["configuration"]["reduce_fan_in"], 2)
        self.assertEqual(report["configuration"]["max_replans"], 0)
        self.assertEqual(len(report["assertions"]), 6)

        with self.assertRaises(ValueError):
            application.benchmark({"mode": "unsupported"})
        with self.assertRaisesRegex(ValueError, "已验证 API 配置不存在或已过期"):
            application.benchmark({"mode": "live", "api": {"api_key": "must-not-be-used"}})


if __name__ == "__main__":
    unittest.main()
