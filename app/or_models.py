"""Curated OpenRouter model catalog for Day 5.

NOTE: OpenRouter's live catalog could not be fetched from this environment
(HTTP 403 blocked; no OPENROUTER_API_KEY), so these defaults are UNVERIFIED
placeholders chosen from widely-used OpenRouter model ids. Confirm each id
against https://openrouter.ai/models before relying on availability, and
edit the allowlist below. Pricing/context are intentionally ``None`` here
rather than invented — the live metadata endpoint would supply real values.
"""
from __future__ import annotations


def _entry(model_id: str, name: str, category: str) -> dict:
    return {
        "id": model_id,
        "name": name,
        "category": category,
        "url": f"https://openrouter.ai/{model_id}",
        # input/output price in USD per 1M tokens and context length are
        # None (unverified) — the live OpenRouter /models endpoint provides
        # authoritative values when it is reachable.
        "input_per_million": None,
        "output_per_million": None,
        "context_length": None,
    }


# UNVERIFIED curated allowlist (small/cheap/well-known OpenRouter models).
MODEL_ALLOWLIST = [
    _entry("qwen/qwen-2.5-7b-instruct", "Qwen 2.5 7B Instruct", "weak"),
    _entry("openai/gpt-4o-mini", "GPT-4o mini", "medium"),
    _entry("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet", "strong"),
]

DEFAULT_MODELS = {e["category"]: e["id"] for e in MODEL_ALLOWLIST}

_ALLOWED_IDS = {e["id"] for e in MODEL_ALLOWLIST}


def is_allowed(model_id: str) -> bool:
    return model_id in _ALLOWED_IDS


def get_model(model_id: str) -> dict | None:
    for e in MODEL_ALLOWLIST:
        if e["id"] == model_id:
            return e
    return None


def models_for_ui() -> list[dict]:
    return list(MODEL_ALLOWLIST)
