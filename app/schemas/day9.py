"""Pydantic schemas for Day 9 — context compression (history summarization)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.chat import ChatRequest


class Day9Usage(BaseModel):
    """Token accounting for ONE exchange, split by compression mode.

    Exact values come from the provider ``usage`` block; local estimates are
    labelled ``*_estimated`` and never presented as API usage.
    """

    # --- what a NON-compressed request would send ---
    full_context_tokens_estimated: int
    # --- what the compressed request actually sends ---
    summary_tokens_estimated: int
    recent_tokens_estimated: int
    compressed_context_tokens_estimated: int
    current_user_tokens_estimated: int
    system_prompt_tokens_estimated: int

    # --- exact, from the DeepSeek API ---
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # --- savings ---
    tokens_saved: int
    tokens_saved_percent: float


class Day9Cost(BaseModel):
    """Estimated USD cost of one exchange (never presented as exact)."""

    input: float | None = None
    output: float | None = None
    total: float | None = None
    currency: str = "USD"
    estimated: bool = True
    reason: str | None = None


class Day9Diagnostics(BaseModel):
    """The compression state, exposed for the diagnostic panel / video."""

    full_history_messages: int
    summarized_messages: int
    recent_messages_sent: int
    full_history_tokens_estimated: int
    summary_tokens_estimated: int
    recent_tokens_estimated: int
    compressed_context_tokens_estimated: int
    tokens_saved: int
    tokens_saved_percent: float
    compression_cycles: int
    summary: str | None = None
    recent_messages_limit: int
    compression_batch_size: int
    pending_messages: int


class Day9ChatRequest(ChatRequest):
    """One Day 9 turn.

    ``mode`` selects the experiment arm. ``recent_messages_limit`` and
    ``compression_batch_size`` override the Agent defaults for this dialog.
    """

    mode: str = Field(default="compressed", pattern="^(compressed|full)$")
    recent_messages_limit: int | None = Field(default=None, ge=1, le=50)
    compression_batch_size: int | None = Field(default=None, ge=2, le=50)
    model: str | None = None
    system_prompt: str | None = None


class Day9ChatResponse(BaseModel):
    """Answer + token accounting + compression diagnostics for one turn."""

    answer: str
    mode: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: Day9Usage
    cost: Day9Cost
    diagnostics: Day9Diagnostics
    # True when this turn triggered a summary rebuild before answering.
    compression_performed: bool = False
    # Populated when a compression attempt failed (history is left intact).
    compression_error: str | None = None


class Day9SeedRequest(BaseModel):
    """Seed a deterministic demo dialog (facts stated at the beginning)."""

    recent_messages_limit: int | None = Field(default=None, ge=1, le=50)
    compression_batch_size: int | None = Field(default=None, ge=2, le=50)
    model: str | None = None


class Day9SeedResponse(BaseModel):
    """The seeded dialog plus its starting diagnostics."""

    messages: list[dict[str, str]]
    count: int
    diagnostics: Day9Diagnostics


class Day9DiagnosticsResponse(BaseModel):
    """Current compression state (no provider call)."""

    model: str
    diagnostics: Day9Diagnostics
    last_answer: str | None = None
