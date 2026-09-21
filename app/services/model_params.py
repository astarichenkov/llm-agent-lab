"""Day 8 — model parameters and pricing.

Every value here is taken from the official DeepSeek documentation and is
kept in ONE backend module so the frontend never hardcodes a price or a limit.

Sources (verified against https://api-docs.deepseek.com):

* Models & Pricing — model ids, context length, max output, token prices;
* Chat Completions API — ``usage`` field names (incl. cache hit/miss).

Notes
-----
* ``deepseek-v4-flash`` is a legacy alias: DeepSeek still accepts it, serves
  it with the DeepSeek-V4.1-Flash model, and bills it at the Flash price.
  The current canonical id is ``deepseek-flash``. Both are mapped here.
* Prices are quoted per 1M tokens and differ between peak and off-peak hours.
  Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday. Because the
  exact billing instant is not something the API reports per request, cost is
  labelled an **estimate**: we use the off-peak rate as the baseline and the
  UI states the assumption explicitly (never presenting it as exact).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

# Prices are per 1,000,000 tokens, in USD (USD -> {price_key: rate}).
_PER_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class ModelPricing:
    """USD price per 1M tokens for one model."""

    input_cache_hit: Decimal
    input_cache_miss: Decimal
    output: Decimal


@dataclass(frozen=True)
class ModelLimits:
    """Context window and maximum output tokens for one model."""

    context_window: int
    max_output_tokens: int


# --- deepseek-flash / deepseek-v4-flash (DeepSeek-V4.1-Flash) --------------
# Context length 1M, maximum output 384K (official Models & Pricing table).
# The docs say "1M"; the provider's own error for an oversized request states
# the exact usable window: 1048576 tokens (1 Mi = 2^20). We use the exact
# value the API enforces. The completion budget counts toward this window too,
# which is why the overflow experiment adds both together.
_EXACT_1M_CONTEXT = 1_048_576
_FLASH_LIMITS = ModelLimits(
    context_window=_EXACT_1M_CONTEXT, max_output_tokens=384_000
)
# Off-peak rates (peak = 2x). Cache hit $0.003 / miss $0.15 / output $0.6.
_FLASH_PRICING = ModelPricing(
    input_cache_hit=Decimal("0.003"),
    input_cache_miss=Decimal("0.15"),
    output=Decimal("0.6"),
)

# --- deepseek-v4-pro (DeepSeek-V4-Pro-0813) --------------------------------
_PRO_LIMITS = ModelLimits(
    context_window=_EXACT_1M_CONTEXT, max_output_tokens=384_000
)
_PRO_PRICING = ModelPricing(
    input_cache_hit=Decimal("0.022"),
    input_cache_miss=Decimal("0.66"),
    output=Decimal("1.98"),
)

# Canonical ids plus the legacy aliases DeepSeek still accepts.
_MODELS: dict[str, tuple[ModelLimits, ModelPricing]] = {
    "deepseek-flash": (_FLASH_LIMITS, _FLASH_PRICING),
    "deepseek-v4-flash": (_FLASH_LIMITS, _FLASH_PRICING),
    "deepseek-v4-flash-vision-exp": (_FLASH_LIMITS, _FLASH_PRICING),
    "deepseek-v4-pro": (_PRO_LIMITS, _PRO_PRICING),
}

# Fallback used when a model is unknown (e.g. an OpenRouter slug). We do not
# invent a price for unknown models: the limits fall back to the flash values
# only for *display*, and cost is reported as not available.
_UNKNOWN_LIMITS = _FLASH_LIMITS

# Human-facing names of the price keys used by the API.
COST_CURRENCY = "USD"


def get_limits(model: str | None) -> ModelLimits:
    """Return the known limits for ``model`` (falls back to flash limits)."""
    if model and model in _MODELS:
        return _MODELS[model][0]
    return _UNKNOWN_LIMITS


def get_pricing(model: str | None) -> ModelPricing | None:
    """Return pricing for ``model``, or ``None`` when it is not a known model."""
    if model and model in _MODELS:
        return _MODELS[model][1]
    return None


def is_known_model(model: str | None) -> bool:
    """True when the model has documented limits and pricing."""
    return bool(model) and model in _MODELS


def estimate_cost(
    model: str | None,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cache_hit_tokens: int | None = None,
    cache_miss_tokens: int | None = None,
) -> dict:
    """Estimate the USD cost of one exchange.

    Uses the API-provided token counts. When the provider reports the cache
    split (``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``) the
    accurate rates are applied; otherwise the whole prompt is billed at the
    cache-miss rate (the conservative, more expensive assumption).

    Returns ``{"input", "output", "total", "currency", "estimated"}``. Costs
    are ``Decimal`` quantized to 8 decimals to avoid float drift. For unknown
    models the cost is ``None`` and ``estimated`` is ``False``.
    """
    pricing = get_pricing(model)
    if pricing is None:
        return {
            "input": None,
            "output": None,
            "total": None,
            "currency": COST_CURRENCY,
            "estimated": False,
            "reason": "unknown model pricing",
        }

    prompt = prompt_tokens or 0
    completion = completion_tokens or 0

    if cache_hit_tokens is not None or cache_miss_tokens is not None:
        hit = cache_hit_tokens or 0
        miss = cache_miss_tokens if cache_miss_tokens is not None else max(prompt - hit, 0)
    else:
        hit = 0
        miss = prompt

    input_cost = (
        Decimal(hit) * pricing.input_cache_hit
        + Decimal(miss) * pricing.input_cache_miss
    ) / _PER_MILLION
    output_cost = Decimal(completion) * pricing.output / _PER_MILLION
    total = input_cost + output_cost

    quantum = Decimal("0.00000001")  # 8 decimal places
    return {
        "input": input_cost.quantize(quantum),
        "output": output_cost.quantize(quantum),
        "total": total.quantize(quantum),
        "currency": COST_CURRENCY,
        "estimated": True,
    }
