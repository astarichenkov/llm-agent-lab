"""Pydantic schemas for Day 13 — Task State Machine.

Day 13 introduces a first-class, STRUCTURED task state that lives separately
from the conversation history:

* **stage**          — the phase of the task (planning / execution / validation / done);
* **current_step**   — which step of the plan is being executed right now;
* **expected_action** — explicit description of what happens next.

The state also keeps a small amount of task bookkeeping (goal, plan,
completed steps, pause flag). It is deliberately compact: Day 13 is ONLY about
the task state machine (no profiles, invariants, long-term memory, RAG, ...).

The stage is NEVER set freely by the LLM: allowed transitions are enforced in
``app.services.day13.task_state`` and the service only applies a transition
after a code-side decision.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatRequest

# The four stages of the task state machine.
TaskStage = Literal["planning", "execution", "validation", "done"]


class TaskState(BaseModel):
    """Structured state of the CURRENT task.

    Answers the three questions the assignment requires:

    * Where are we?            -> ``stage``
    * What are we doing?       -> ``current_step`` / ``completed_steps``
    * What is expected next?   -> ``expected_action``
    """

    task_id: str
    goal: str = ""
    stage: TaskStage = "planning"
    # 1-based index into ``plan``; during planning it is 1 as well.
    current_step: int = 1
    expected_action: str = ""
    plan: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    # ``paused`` is NOT a separate FSM stage: the stage is preserved so the
    # task can be resumed exactly where it stopped.
    paused: bool = False
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_finished(self) -> bool:
        return self.stage == "done"

    @property
    def total_steps(self) -> int:
        return len(self.plan)


# ----------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------
class Day13TaskCreateRequest(BaseModel):
    """Start a NEW task from a natural-language goal."""

    goal: str = Field(..., min_length=1, max_length=2000)


class Day13ChatRequest(ChatRequest):
    """One turn of work on the CURRENT task.

    The message usually does not repeat the task: the backend already knows the
    goal, the stage, the current step and the expected action, and injects the
    formalised Task State into the model request.
    """

    model: str | None = None


class Day13TransitionRequest(BaseModel):
    """Request an explicit FSM transition (validated by code)."""

    to_stage: TaskStage


# ----------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------
class Day13StateResponse(BaseModel):
    """Current Task State plus the transitions the FSM allows right now."""

    model: str
    has_task: bool = False
    state: TaskState | None = None
    allowed_transitions: list[str] = Field(default_factory=list)
    stages: list[str] = Field(
        default_factory=lambda: ["planning", "execution", "validation", "done"]
    )


class Day13ChatResponse(BaseModel):
    """Answer for one turn + the resulting Task State."""

    answer: str
    model: str
    provider: str = "deepseek"
    finish_reason: str | None = None
    state: TaskState
    allowed_transitions: list[str] = Field(default_factory=list)
    # The exact TASK STATE block that was placed into the model request.
    context_block: str = ""
    # Human-readable description of a code-applied transition, if any.
    stage_event: str | None = None
    usage: dict | None = None
