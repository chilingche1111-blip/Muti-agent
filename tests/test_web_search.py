from __future__ import annotations

import unittest
from unittest.mock import patch

from long_context_agent.web_search import WebSearchError, search_web


SEARCH_HTML = """
<html><body>
<ol>
  <li class="b_algo"><h2><a href="https://example.com/one"><strong>第一条</strong>结果</a></h2><div><p>第一条搜索摘要。</p></div></li>
  <li class="b_algo"><h2><a href="https://example.com/two">第二条结果</a></h2><div><p>第二条搜索摘要。</p></div></li>
</ol>
</body></html>
"""


class WebSearchTests(unittest.TestCase):
    def test_bing_html_results_are_normalized(self) -> None:
        with patch("long_context_agent.web_search._curl", return_value=SEARCH_HTML):
            results = search_web("测试搜索", limit=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "第一条 结果")
        self.assertEqual(results[0].url, "https://example.com/one")
        self.assertEqual(results[1].snippet, "第二条搜索摘要。")

    def test_empty_or_unparseable_search_fails_explicitly(self) -> None:
        with patch("long_context_agent.web_search._curl", return_value="<html>verification</html>"):
            with self.assertRaisesRegex(WebSearchError, "没有解析到结果"):
                search_web("测试搜索")


if __name__ == "__main__":
    unittest.main()
