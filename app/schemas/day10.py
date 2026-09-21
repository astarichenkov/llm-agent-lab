"""Pydantic schemas for Day 10 — context-management strategies.

Day 10 exposes THREE switchable strategies that all shape the payload sent to
the model differently:

* ``sliding_window`` — only the last ``window_size`` history messages survive;
* ``sticky_facts``   — a structured key/value memory block plus a recent window;
* ``branching``      — independent branches grown from a shared checkpoint.

Unlike Day 9 this module intentionally contains NO summary/history-compression
schema: Day 10 never summarises the conversation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatRequest

# The three context strategies selectable from the UI and the API.
ContextStrategyName = Literal["sliding_window", "sticky_facts", "branching"]


class Facts(BaseModel):
    """Structured, durable memory of the conversation.

    This is deliberately NOT a summary: it stores stable, meaningful values
    (goal, constraints, decisions...), never a retelling of the dialog.
    """

    goal: str | None = None
    constraints: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            (self.goal or "").strip()
            or self.constraints
            or self.preferences
            or self.decisions
            or self.agreements
        )

    def count(self) -> int:
        """Number of stored facts (the goal counts as one)."""
        return (
            (1 if (self.goal or "").strip() else 0)
            + len(self.constraints)
            + len(self.preferences)
            + len(self.decisions)
            + len(self.agreements)
        )

    def to_block(self) -> str:
        """Human/LLM-readable FACTS block injected before the recent window."""
        if self.is_empty():
            return "Known facts about the current conversation:\n(no facts yet)"
        parts = ["Known facts about the current conversation:"]
        if (self.goal or "").strip():
            parts.append(f"Goal:\n{self.goal.strip()}")
        for title, values in (
            ("Constraints", self.constraints),
            ("Preferences", self.preferences),
            ("Decisions", self.decisions),
            ("Agreements", self.agreements),
        ):
            if values:
                parts.append(title + ":\n" + "\n".join(f"- {v}" for v in values))
        return "\n\n".join(parts)


class Day10Usage(BaseModel):
    """Token accounting for one Day 10 turn.

    Exact numbers come from the provider ``usage`` block; local estimates are
    explicitly suffixed ``_estimated``. The facts-extraction call (when it
    happens) is accounted separately from the answer-generation call.
    """

    # answer generation — exact
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # facts extraction — exact (None when no extraction happened)
    facts_extraction_prompt_tokens: int | None = None
    facts_extraction_completion_tokens: int | None = None
    facts_extraction_total_tokens: int | None = None
    # local estimates of the actual payload
    system_prompt_tokens_estimated: int = 0
    facts_tokens_estimated: int = 0
    recent_tokens_estimated: int = 0
    current_user_tokens_estimated: int = 0
    estimated_input_tokens: int = 0
    # context composition
    total_history_messages: int = 0
    sent_history_messages: int = 0
    dropped_messages: int = 0
    window_size: int | None = None


class Day10Cost(BaseModel):
    """Estimated USD cost of the answer call (never exact)."""

    input: float | None = None
    output: float | None = None
    total: float | None = None
    currency: str = "USD"
    estimated: bool = True
    reason: str | None = None


class BranchInfo(BaseModel):
    """Public description of one conversation branch."""

    id: str
    name: str
    parent_branch_id: str | None = None
    checkpoint_message_id: str | None = None
    message_count: int = 0
    is_active: bool = False


class Day10ContextInfo(BaseModel):
    """Strategy-specific state returned with every turn (drives the UI)."""

    strategy: ContextStrategyName
    total_history_messages: int = 0
    sent_history_messages: int = 0
    dropped_messages: int = 0
    dropped_preview: list[dict[str, str]] = Field(default_factory=list)
    sent_preview: list[dict[str, str]] = Field(default_factory=list)
    window_size: int | None = None
    facts: Facts | None = None
    facts_block: str | None = None
    branches: list[BranchInfo] = Field(default_factory=list)
    active_branch_id: str | None = None
    active_branch_name: str | None = None
    checkpoint_message_id: str | None = None


class Day10ChatRequest(ChatRequest):
    """One Day 10 turn under a chosen context strategy."""

    strategy: ContextStrategyName = "sliding_window"
    # sliding window
    window_size: int | None = Field(default=None, ge=1, le=100)
    # sticky facts (recent window)
    recent_messages_limit: int | None = Field(default=None, ge=1, le=100)
    # sticky facts: run the LLM extraction before answering
    update_facts: bool = True
    # branching: target branch (defaults to the active one)
    branch_id: str | None = None
    model: str | None = None
    system_prompt: str | None = None


class Day10ChatResponse(BaseModel):
    """Answer + context diagnostics + token accounting for one turn."""

    strategy: ContextStrategyName
    answer: str
    model: str
    provider: str = "deepseek"
    finish_reason: str | None = None
    usage: Day10Usage
    cost: Day10Cost
    context: Day10ContextInfo
    facts_extraction_performed: bool = False
    facts_error: str | None = None
    active_branch_id: str | None = None


class Day10StateResponse(BaseModel):
    """Current strategy state (no provider call)."""

    strategy: ContextStrategyName
    model: str
    context: Day10ContextInfo


class Day10BranchCreateRequest(BaseModel):
    """Create a new branch from a checkpoint of an existing branch."""

    strategy: ContextStrategyName = "branching"
    name: str | None = Field(default=None, max_length=80)
    parent_branch_id: str | None = None
    checkpoint_message_id: str | None = None


class Day10BranchCreateResponse(BaseModel):
    branch: BranchInfo
    context: Day10ContextInfo


class Day10BranchActivateRequest(BaseModel):
    """Switch the active branch."""

    strategy: ContextStrategyName = "branching"
    branch_id: str


class Day10BranchActivateResponse(BaseModel):
    context: Day10ContextInfo


class Day10ScenarioResponse(BaseModel):
    """The shared, deterministic demo scenario definition."""

    messages: list[str]
    expected: list[str]
    branching: dict


class Day10EvaluateRequest(BaseModel):
    """Score an answer against the expected demo facts."""

    answer: str
    expected: list[str] | None = None


class Day10EvaluateResponse(BaseModel):
    """Demo-only evaluation metric (NOT an objective AI benchmark)."""

    matched: list[str]
    missing: list[str]
    score: int
    total: int
    metric: str = "demo_evaluation_metric"


class Day10DemoStep(BaseModel):
    index: int
    message: str
    answer: str
    usage: Day10Usage
    context: Day10ContextInfo
    facts_error: str | None = None


class Day10DemoRunResponse(BaseModel):
    """Result of running the whole scenario server-side (tests / batch)."""

    strategy: ContextStrategyName
    steps: list[Day10DemoStep]
    final_answer: str
    usage: Day10Usage
    evaluation: Day10EvaluateResponse
    context: Day10ContextInfo
