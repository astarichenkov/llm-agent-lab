"""Pydantic schemas for Day 8 — token accounting and context-limit experiments."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatRequest


class Day8Usage(BaseModel):
    """Token accounting for ONE exchange.

    ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` are EXACT
    values from the provider ``usage`` block. The ``*_estimated`` fields are
    LOCAL estimates of individual prompt parts and are labelled as such.
    """

    # exact, from the provider
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None
    # local estimates
    current_user_tokens_estimated: int
    history_tokens_estimated: int
    system_prompt_tokens_estimated: int
    estimated_input_tokens: int


class Day8Cost(BaseModel):
    """Estimated USD cost of one exchange (never presented as exact)."""

    input: float | None = None
    output: float | None = None
    total: float | None = None
    currency: str = "USD"
    estimated: bool = True
    reason: str | None = None


class Day8Limits(BaseModel):
    """Model limits used by the overflow experiment."""

    context_window: int
    max_output_tokens: int
    model_known: bool = True


class Day8ChatRequest(ChatRequest):
    """Day 8 chat request.

    ``history`` may optionally carry synthetic prior turns for the long /
    overflow experiments. When omitted, the in-memory Day 8 dialog is used.
    The synthetic history is NOT persisted to the Day 7 database.
    """

    model: str | None = None
    history: list[dict[str, str]] | None = None
    system_prompt: str | None = None


class Day8ChatResponse(BaseModel):
    """Day 8 response: answer + exact usage + estimates + cost + limits."""

    answer: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: Day8Usage
    cost: Day8Cost
    limits: Day8Limits
    message_count: int


class Day8EstimateRequest(BaseModel):
    """Local-only estimation of a synthetic prompt (no provider call)."""

    message: str = Field(default="", max_length=200_000)
    history: list[dict[str, str]] = Field(default_factory=list)
    model: str | None = None
    system_prompt: str | None = None


class Day8EstimateResponse(BaseModel):
    """Result of the local estimation endpoint."""

    model: str
    usage: Day8Usage
    limits: Day8Limits
    # Estimated messages input + the completion budget, i.e. what the
    # provider actually counts against the context window.
    effective_input_tokens: int
    exceeds_context: bool
    overflow_tokens: int


class Day8OverflowRequest(BaseModel):
    """Ask the backend to build a deterministic, oversized synthetic context.

    The history is generated SERVER-SIDE on purpose: a multi-megabyte
    synthetic dialog must not be uploaded from the browser (nginx limits the
    request body, and uploading it would be wasteful). ``overshoot_factor``
    scales how far past the context window we aim (default 1.15 = +15%).
    """

    model: str | None = None
    system_prompt: str | None = None
    overshoot_factor: float = Field(default=1.15, ge=1.0, le=2.0)


class Day8OverflowResponse(BaseModel):
    """A generated oversized context plus its local estimate.

    ``request`` is the exact chat payload to send to ``/api/day8/chat`` when
    the learner clicks the real-request button. It is returned to the client
    so the SAME deterministic context is used for estimation and execution.
    """

    model: str
    usage: Day8Usage
    limits: Day8Limits
    effective_input_tokens: int
    exceeds_context: bool
    overflow_tokens: int
    turns: int
    message_tokens: int
    request: Day8ChatRequest
