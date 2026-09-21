"""Pydantic schemas for Day 11 — the agent MEMORY MODEL.

Day 11 adds THREE explicitly separated memory layers:

* **short-term**  — the messages of the current dialog/session;
* **working**     — structured state of the CURRENT task (goal, constraints...);
* **long-term**   — durable facts/preferences that survive a new session.

This is a DIFFERENT axis from Day 10 context-management strategies:

* memory layers answer "what to store, where and for how long?";
* a context strategy answers "which part of the available context do we send
  to the model in THIS request?".

So the Day 11 flow is:

    user message -> MemoryClassifier -> short-term / working / long-term
                 -> Context Builder -> existing context strategy -> DeepSeek

The module deliberately contains NO embeddings / vector-store concepts.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatRequest

# "full" sends the whole short-term history; "sliding_window" reuses the
# existing Day 10 strategy. This is the ONLY context strategy Day 11 needs:
# the point of Day 11 is the memory layers, not new context strategies.
MemoryContextStrategy = Literal["sliding_window", "full"]


# ----------------------------------------------------------------------
# Memory layers
# ----------------------------------------------------------------------
class ShortTermMessage(BaseModel):
    """One message of the CURRENT session (short-term memory)."""

    role: Literal["user", "assistant"]
    content: str


class WorkingMemory(BaseModel):
    """Structured state of the CURRENT task.

    It is NOT a copy of the conversation: it stores the distilled state the
    task needs (goal, constraints, requirements, decisions, data). Cleared
    independently from long-term memory.
    """

    goal: str | None = None
    constraints: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    data: dict[str, str] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (
            (self.goal or "").strip()
            or self.constraints
            or self.requirements
            or self.decisions
            or self.data
        )

    def count(self) -> int:
        return (
            (1 if (self.goal or "").strip() else 0)
            + len(self.constraints)
            + len(self.requirements)
            + len(self.decisions)
            + len(self.data)
        )

    def to_block(self) -> str:
        """Human/LLM-readable WORKING MEMORY block."""
        if self.is_empty():
            return "WORKING MEMORY (current task):\n(empty)"
        parts = ["WORKING MEMORY (current task):"]
        if (self.goal or "").strip():
            parts.append(f"Goal: {self.goal.strip()}")
        for title, values in (
            ("Constraints", self.constraints),
            ("Requirements", self.requirements),
            ("Decisions", self.decisions),
        ):
            if values:
                parts.append(title + ":\n" + "\n".join(f"- {v}" for v in values))
        if self.data:
            parts.append(
                "Data:\n"
                + "\n".join(f"- {key}: {value}" for key, value in self.data.items())
            )
        return "\n\n".join(parts)

    def as_lines(self) -> list[str]:
        """Flat, UI-friendly representation of the stored state."""
        lines: list[str] = []
        if (self.goal or "").strip():
            lines.append(f"goal = {self.goal.strip()}")
        for label, values in (
            ("constraint", self.constraints),
            ("requirement", self.requirements),
            ("decision", self.decisions),
        ):
            for value in values:
                lines.append(f"{label} = {value}")
        for key, value in self.data.items():
            lines.append(f"{key} = {value}")
        return lines


class LongTermMemory(BaseModel):
    """Durable key/value facts that survive a new session.

    Examples: ``preferred_language = Python``, ``answer_style = concise``.
    Stored physically separately from short-term/working memory.
    """

    entries: dict[str, str] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.entries

    def count(self) -> int:
        return len(self.entries)

    def to_block(self) -> str:
        """Human/LLM-readable LONG-TERM MEMORY block."""
        if not self.entries:
            return "LONG-TERM MEMORY (durable preferences):\n(empty)"
        lines = "\n".join(f"- {key} = {value}" for key, value in self.entries.items())
        return "LONG-TERM MEMORY (durable preferences):\n" + lines


class ClassifierDecision(BaseModel):
    """Observable result of ONE MemoryClassifier run.

    ``working_memory`` / ``long_term_memory`` contain ONLY the delta the
    classifier decided to write. ``nothing_to_save`` is the explicit
    "do not pollute memory" outcome.
    """

    performed: bool = False
    nothing_to_save: bool = True
    error: str | None = None
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)
    long_term_memory: dict[str, str] = Field(default_factory=dict)

    def has_updates(self) -> bool:
        return not self.working_memory.is_empty() or bool(self.long_term_memory)

    def summary_lines(self) -> list[str]:
        """Short human-readable description of what was decided."""
        if self.error:
            return [f"Ошибка классификации: {self.error}"]
        if not self.performed:
            return ["Классификация не выполнялась."]
        if self.nothing_to_save or not self.has_updates():
            return ["nothing_to_save — память не изменялась."]
        lines: list[str] = []
        for line in self.working_memory.as_lines():
            lines.append("Working: " + line)
        for key, value in self.long_term_memory.items():
            lines.append(f"Long-term: {key} = {value}")
        return lines


# ----------------------------------------------------------------------
# Diagnostics / token accounting
# ----------------------------------------------------------------------
class Day11Usage(BaseModel):
    """Token accounting for one Day 11 turn."""

    # answer generation — exact
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # classifier call — exact (None when no call happened)
    classifier_prompt_tokens: int | None = None
    classifier_completion_tokens: int | None = None
    classifier_total_tokens: int | None = None
    # local estimates of the actual answer payload
    system_prompt_tokens_estimated: int = 0
    # Day 12 — optional personalization block injected by the profile layer
    personalization_tokens_estimated: int = 0
    long_term_tokens_estimated: int = 0
    working_tokens_estimated: int = 0
    short_term_tokens_estimated: int = 0
    current_user_tokens_estimated: int = 0
    estimated_input_tokens: int = 0


class Day11Cost(BaseModel):
    """Estimated USD cost of the answer call (never exact)."""

    input: float | None = None
    output: float | None = None
    total: float | None = None
    currency: str = "USD"
    estimated: bool = True
    reason: str | None = None


class Day11ContextInfo(BaseModel):
    """What the Context Builder actually assembled for this turn."""

    strategy: MemoryContextStrategy
    total_short_term_messages: int = 0
    sent_short_term_messages: int = 0
    dropped_messages: int = 0
    dropped_preview: list[ShortTermMessage] = Field(default_factory=list)
    sent_preview: list[ShortTermMessage] = Field(default_factory=list)
    window_size: int | None = None
    long_term_included: bool = False
    working_included: bool = False
    # Day 12 — user profile instructions (presentation only)
    personalization_included: bool = False
    included_layers: list[str] = Field(default_factory=list)
    personalization_block: str | None = None
    long_term_block: str | None = None
    working_block: str | None = None


class Day11MemoryState(BaseModel):
    """Full, observable snapshot of all three memory layers."""

    session_id: str
    short_term: list[ShortTermMessage] = Field(default_factory=list)
    short_term_count: int = 0
    working: WorkingMemory = Field(default_factory=WorkingMemory)
    long_term: LongTermMemory = Field(default_factory=LongTermMemory)
    last_decision: ClassifierDecision | None = None


# ----------------------------------------------------------------------
# Requests / responses
# ----------------------------------------------------------------------
class Day11ChatRequest(ChatRequest):
    """One Day 11 turn with explicit memory + context controls."""

    session_id: str | None = None
    strategy: MemoryContextStrategy = "sliding_window"
    window_size: int | None = Field(default=None, ge=1, le=100)
    # master switch: disables classifier AND both structured memory blocks
    use_memory: bool = True
    # run the MemoryClassifier before answering
    classify: bool = True
    include_long_term: bool = True
    include_working: bool = True
    model: str | None = None
    system_prompt: str | None = None


class Day11ChatResponse(BaseModel):
    """Answer + memory snapshot + context diagnostics for one turn."""

    session_id: str
    answer: str
    model: str
    provider: str = "deepseek"
    finish_reason: str | None = None
    usage: Day11Usage
    cost: Day11Cost
    context: Day11ContextInfo
    memory: Day11MemoryState
    last_decision: ClassifierDecision | None = None
    classifier_error: str | None = None


class Day11StateResponse(BaseModel):
    """Current memory state (no provider call)."""

    model: str
    memory: Day11MemoryState


class Day11EvaluateRequest(BaseModel):
    answer: str
    expected: list[str] | None = None


class Day11EvaluateResponse(BaseModel):
    matched: list[str]
    missing: list[str]
    score: int
    total: int
    metric: str = "demo_evaluation_metric"


class Day11ScenarioResponse(BaseModel):
    """The deterministic demo scenario definition (no provider call)."""

    messages: list[str]
    expected: list[str]
