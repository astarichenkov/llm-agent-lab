"""Curated OpenRouter model catalog for Day 5.

Values below were taken from the LIVE OpenRouter catalog
(GET https://openrouter.ai/api/v1/models, 2026) for three currently available
text chat models. Prices are USD per 1M tokens (converted from OpenRouter's
per-token rates). Model page URL follows OpenRouter's <provider>/<model> slug.
"""
from __future__ import annotations


def _entry(model_id: str, name: str, category: str, context, inp, outp) -> dict:
    return {
        "id": model_id,
        "name": name,
        "category": category,
        "url": f"https://openrouter.ai/{model_id}",
        "input_per_million": inp,   # USD per 1M input tokens
        "output_per_million": outp,  # USD per 1M output tokens
        "context_length": context,
    }


MODEL_ALLOWLIST = [
    _entry("qwen/qwen-2.5-7b-instruct", "Qwen2.5 7B Instruct", "weak", 32768, 0.10, 0.20),
    _entry("openai/gpt-4o-mini", "GPT-4o mini", "medium", 128000, 0.15, 0.60),
    _entry("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6", "strong", 1_000_000, 3.0, 15.0),
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
