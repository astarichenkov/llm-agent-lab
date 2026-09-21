"""Day 8 — token accounting primitives.

Two clearly separated layers:

* **Exact values** come from the provider ``usage`` block
  (``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``,
  ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``). These are the
  source of truth for billing and are never invented locally.
* **Local estimates** for individual prompt parts (system prompt, history,
  the new user message). DeepSeek does not ship a public, officially
  supported tokenizer for the current models in this project, so these are
  clearly labelled estimates.

The estimator is dependency-free, deterministic and calibrated against the
real DeepSeek API. Rather than trusting DeepSeek's documentation's very rough
"1 English char ~= 0.3 token" figure, the coefficients below were measured
empirically against ``deepseek-v4-flash`` by sending controlled inputs and
reading the returned ``prompt_tokens``:

===========================  ==============  ================
character class              chars sent      tokens/char
===========================  ==============  ================
ASCII filler (``a``)               1000         ~0.129
English words (mixed)               880         ~0.210
Cyrillic (``привет``)              1800         ~0.336
CJK ideographs (``漢``)             200         ~1.020
===========================  ==============  ================

A small per-message overhead (role/separator markers) is added on top. The
result is still an ESTIMATE: the UI always labels it "≈" and shows the exact
API number next to it.
"""
from __future__ import annotations

import math

# Measured coefficients (tokens per character) for deepseek-v4-flash.
_ASCII_TOKENS_PER_CHAR = 0.13
_ENGLISH_TOKENS_PER_CHAR = 0.21
_CYRILLIC_TOKENS_PER_CHAR = 0.336
_CJK_TOKENS_PER_CHAR = 1.0
# Per-message structural overhead (role + separators). Measured ~4 tokens for
# a message pair; 2 per single message is a good conservative fit.
_PER_MESSAGE_OVERHEAD = 4


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF      # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF   # CJK Extension A
        or 0x3040 <= code <= 0x30FF   # Hiragana / Katakana
        or 0xAC00 <= code <= 0xD7AF   # Hangul syllables
        or 0xF900 <= code <= 0xFAFF   # CJK Compatibility Ideographs
    )


def _is_cyrillic(char: str) -> bool:
    code = ord(char)
    return 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F


def _is_ascii_letter(char: str) -> bool:
    return ("a" <= char <= "z") or ("A" <= char <= "Z")


def estimate_text_tokens(text: str) -> int:
    """Estimate the tokens of a raw text fragment.

    Deterministic, offline and content-driven. Characters are bucketed:
    CJK (measured ~1.0/char), Cyrillic (~0.336/char), ASCII letters
    (~0.21/char), and everything else (digits, punctuation, spaces) at the
    lower ASCII-ish rate (~0.13/char). Rounded up so non-empty text never
    estimates to zero.
    """
    if not text:
        return 0
    cjk = cyrillic = ascii_letters = other = 0
    for char in text:
        if _is_cjk(char):
            cjk += 1
        elif _is_cyrillic(char):
            cyrillic += 1
        elif _is_ascii_letter(char):
            ascii_letters += 1
        else:
            other += 1
    estimate = (
        cjk * _CJK_TOKENS_PER_CHAR
        + cyrillic * _CYRILLIC_TOKENS_PER_CHAR
        + ascii_letters * _ENGLISH_TOKENS_PER_CHAR
        + other * _ASCII_TOKENS_PER_CHAR
    )
    return int(math.ceil(estimate))


def estimate_message_tokens(content: str) -> int:
    """Estimate one chat message including its small structural overhead."""
    tokens = estimate_text_tokens(content)
    if tokens == 0:
        return 0
    return tokens + _PER_MESSAGE_OVERHEAD


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate the tokens of a full ``messages`` payload."""
    return sum(
        estimate_message_tokens(message.get("content", "")) for message in messages
    )


def estimate_context_breakdown(
    *,
    system_prompt: str | None,
    history: list[dict[str, str]],
    current_user_message: str,
) -> dict[str, int]:
    """Break an outgoing prompt into its estimable parts.

    ``history`` is the list of ALREADY STORED turns that will be sent before
    the new user message. Returns:

    * ``system_prompt_tokens``  — local estimate;
    * ``history_tokens``        — local estimate of prior turns;
    * ``current_user_tokens``   — local estimate of the new message;
    * ``estimated_input_tokens``— sum of the three (the whole input estimate).
    """
    system_tokens = (
        estimate_message_tokens(system_prompt) if system_prompt else 0
    )
    history_tokens = estimate_messages_tokens(history)
    current_tokens = estimate_message_tokens(current_user_message)
    return {
        "system_prompt_tokens": system_tokens,
        "history_tokens": history_tokens,
        "current_user_tokens": current_tokens,
        "estimated_input_tokens": system_tokens + history_tokens + current_tokens,
    }
