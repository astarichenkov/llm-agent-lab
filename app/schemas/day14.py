"""Pydantic schemas for Day 14 — invariants and state constraints.

Day 14 adds a layer of MANDATORY constraints that the assistant is not
allowed to violate. Invariants are stored SEPARATELY from the dialog:

* :class:`Invariant`         — one mandatory constraint (category + rule);
* :class:`Day14Message`      — one ordinary chat message;
* :class:`InvariantConflict` — a user request that violates an invariant.

The three concerns stay logically separated, exactly like the assignment
asks::

    Task State   (Day 13, unchanged)
    Invariants   (Day 14, this module)
    Conversation (Day 14 messages)

Invariants are ordinary structured data (no DSL, no policy engine): a small
list of ``{id, category, rule}`` records. Conflict detection is deterministic
and lives in ``app.services.day14.invariants``; code decides whether a request
is allowed, so the behaviour never depends on the free-form wording of an LLM.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatRequest

# The four invariant families used by the assignment.
InvariantCategory = Literal[
    "architecture",
    "technical_decision",
    "stack",
    "business",
]

# Human-readable labels for the UI ("Technical decision", "Business rule", ...).
CATEGORY_LABELS: dict[str, str] = {
    "architecture": "Architecture",
    "technical_decision": "Technical decision",
    "stack": "Stack",
    "business": "Business rule",
}


class Invariant(BaseModel):
    """One mandatory constraint the assistant must respect.

    Deliberately minimal: an id, a category and the rule itself. Invariants
    are NOT messages and are never appended to the conversation history.
    """

    id: str
    category: InvariantCategory
    rule: str

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category)


class Day14Message(BaseModel):
    """One ordinary conversation message (never an invariant)."""

    role: Literal["user", "assistant"]
    content: str


class InvariantConflict(BaseModel):
    """A request fragment that violates a specific invariant."""

    invariant_id: str
    category: str
    category_label: str
    rule: str
    # The exact trigger fragments found in the user request (for the UI/audit).
    matched: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------
class Day14ChatRequest(ChatRequest):
    """One user request to validate against the active invariants."""

    model: str | None = None


# ----------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------
class Day14StateResponse(BaseModel):
    """Active invariants + the (separate) conversation history."""

    model: str
    invariants: list[Invariant] = Field(default_factory=list)
    messages: list[Day14Message] = Field(default_factory=list)
    count: int = 0


class Day14ChatResponse(BaseModel):
    """The result of validating and answering one request.

    * ``allowed``   — True when no invariant conflict was found;
    * ``conflicts`` — every violated invariant (empty when allowed);
    * ``answer``    — the answer (or the explicit refusal explanation);
    * ``context_block`` — the exact ACTIVE INVARIANTS block that was placed
      into the model request (proof that invariants reach the model).
    """

    answer: str
    model: str
    provider: str = "deepseek"
    allowed: bool = True
    conflicts: list[InvariantConflict] = Field(default_factory=list)
    invariants: list[Invariant] = Field(default_factory=list)
    context_block: str = ""
    finish_reason: str | None = None
    usage: dict | None = None
