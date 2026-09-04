"""Schemas for Day 5 — OpenRouter model comparison.

One and the same prompt is sent to three different models via OpenRouter's
unified Chat Completions API. No DeepSeek config is used here.
"""
from pydantic import BaseModel, Field, field_validator

from app.schemas.chat import MAX_MESSAGE_LENGTH, ensure_not_blank
from app.schemas.compare import (
    MAX_MAX_TOKENS,
    MAX_STOP_SEQUENCE_LENGTH,
    MIN_MAX_TOKENS,
    MIN_STOP_SEQUENCE_LENGTH,
)

DEFAULT_MAX_TOKENS_DAY5 = 500
DEFAULT_TEMPERATURE_DAY5 = 0.3


class OpenRouterRunRequest(BaseModel):
    """Payload for ``POST /api/openrouter/run``."""

    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    model: str = Field(..., min_length=1, max_length=200)
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS_DAY5, ge=MIN_MAX_TOKENS, le=MAX_MAX_TOKENS
    )
    temperature: float = Field(DEFAULT_TEMPERATURE_DAY5, ge=0.0, le=2.0)
    stop_sequence: str | None = None

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        return ensure_not_blank(v)

    @field_validator("stop_sequence")
    @classmethod
    def _clean_stop(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if len(v) > MAX_STOP_SEQUENCE_LENGTH:
            raise ValueError(f"stop too long (max {MAX_STOP_SEQUENCE_LENGTH})")
        if len(v) < MIN_STOP_SEQUENCE_LENGTH or not any(c.isalnum() for c in v):
            raise ValueError("stop must be a distinctive marker (>=4 chars)")
        return v


class OpenRouterRunResponse(BaseModel):
    answer: str
    requested_model: str
    actual_model: str
    finish_reason: str | None = None
    elapsed_ms: float | None = None
    usage: dict | None = None
    cost: float | None = None  # provider-reported if present; else None
    cost_estimated: bool = False
    pricing: dict | None = None  # from catalog (may be empty/None if unverified)
    model_url: str | None = None
