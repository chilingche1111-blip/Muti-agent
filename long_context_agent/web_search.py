from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from urllib.parse import quote_plus, urljoin, urlparse


class WebSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _plain_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _curl(url: str, *, timeout: int = 20) -> str:
    executable = shutil.which("curl")
    if executable is None:
        raise WebSearchError("系统未安装 curl，无法执行联网检索。")
    try:
        completed = subprocess.run(
            [
                executable,
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                str(timeout),
                "--user-agent",
                "Mozilla/5.0 ContextAtlas/0.9",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WebSearchError("联网检索请求超时或无法启动。") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "未知网络错误").strip()[:240]
        raise WebSearchError(f"联网检索失败：{detail}")
    return completed.stdout


def _bing_search(query: str, limit: int) -> list[WebSearchResult]:
    body = _curl(
        "https://www.bing.com/search?"
        f"q={quote_plus(query)}&count={max(5, min(10, limit))}&setlang=zh-Hans&adlt=strict"
    )
    results: list[WebSearchResult] = []
    blocks = re.findall(r'<li\s+class="b_algo"[^>]*>(.*?)</li>', body, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        heading = re.search(
            r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>\s*</h2>',
            block,
            re.DOTALL | re.IGNORECASE,
        )
        if not heading:
            continue
        url = html.unescape(heading.group(1)).strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        paragraph = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL | re.IGNORECASE)
        title = _plain_text(heading.group(2))[:300]
        snippet = _plain_text(paragraph.group(1) if paragraph else "")[:1_000]
        if title and all(item.url != url for item in results):
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= limit:
            break
    if not results:
        raise WebSearchError("搜索服务可访问，但没有解析到结果；可能触发了搜索站点验证。")
    return results


def _searxng_search(query: str, limit: int, base_url: str) -> list[WebSearchResult]:
    parsed_base = urlparse(base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
        raise WebSearchError("CONTEXT_ATLAS_SEARXNG_URL 配置无效。")
    endpoint = urljoin(base_url.rstrip("/") + "/", "search")
    body = _curl(f"{endpoint}?q={quote_plus(query)}&format=json&language=zh-CN")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise WebSearchError("企业 SearXNG 未返回 JSON。") from exc
    results: list[WebSearchResult] = []
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if urlparse(url).scheme not in {"http", "https"}:
            continue
        results.append(WebSearchResult(
            title=str(item.get("title") or url)[:300],
            url=url,
            snippet=str(item.get("content") or "")[:1_000],
        ))
        if len(results) >= limit:
            break
    if not results:
        raise WebSearchError("企业 SearXNG 没有返回搜索结果。")
    return results


def search_web(query: str, *, limit: int = 5) -> list[WebSearchResult]:
    cleaned = re.sub(r"\s+", " ", query).strip()[:500]
    if not cleaned:
        raise WebSearchError("联网检索词不能为空。")
    searxng_url = os.getenv("CONTEXT_ATLAS_SEARXNG_URL", "").strip()
    return _searxng_search(cleaned, limit, searxng_url) if searxng_url else _bing_search(cleaned, limit)
