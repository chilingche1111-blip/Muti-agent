from __future__ import annotations

import math
import re


_CJK = re.compile(r"[\u3400-\u9fff]")


def estimate_tokens(text: str) -> int:
    """Conservative fallback when the company's exact tokenizer is unavailable.

    Chinese characters are counted one-for-one. Other characters are estimated
    at 3.5 characters per token. Production should replace this with the exact
    tokenizer exposed by the deployed model when available.
    """

    cjk_count = len(_CJK.findall(text))
    non_cjk_count = max(0, len(text) - cjk_count)
    return max(1, cjk_count + math.ceil(non_cjk_count / 3.5))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate a complete Chat Completions input, including message framing.

    Provider tokenizers account for role/message separators in addition to text.
    The exact overhead varies by model, so this intentionally leaves a small,
    conservative allowance per message until the gateway returns real ``usage``.
    """

    return 3 + sum(
        5 + estimate_tokens(str(message.get("role", ""))) + estimate_tokens(str(message.get("content", "")))
        for message in messages
    )
