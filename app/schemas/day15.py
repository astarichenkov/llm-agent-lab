"""Pydantic schemas for Day 15 — controlled state transitions.

Day 15 turns the Day 13 ``stage`` field into a real, **controlled lifecycle**:

* the task has a fixed set of states;
* transitions are explicitly allowed in code (``ALLOWED_TRANSITIONS``);
* additional **guards** (``plan_approved``, ``validation_passed``) must hold
  before critical transitions;
* every attempt (allowed AND blocked) is written to a **transition history**.

The model is deliberately close to Day 13's ``TaskState`` so it is an obvious
evolution, but it lives in its OWN module/service. Day 13 stays untouched and
backward compatible, while Day 15 adds the missing ``plan_approval`` state and
the guard layer the assignment asks for.

Fields
------
* ``state``             — current lifecycle state;
* ``previous_state``    — the ACTIVE state a pause was taken from;
* ``plan_approved``     — guard for ``plan_approval -> execution``;
* ``validation_passed`` — guard for ``validation -> done`` (``None`` = not run);
* ``transitions``       — full, ordered history including blocked attempts.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatRequest

# The five states of the controlled lifecycle.
TaskStateName = Literal[
    "planning",
    "plan_approval",
    "execution",
    "validation",
    "done",
]

TransitionStatus = Literal["allowed", "blocked"]

# Semantic intents the LLM is allowed to recognise. The LLM NEVER returns a
# target state — the application maps an intent to an action and the state
# machine decides the transition. Any other value is downgraded to
# ``normal_message`` by the parser.
IntentName = Literal[
    "continue_planning",
    "approve_plan",
    "request_plan_changes",
    "revise_plan",
    "start_execution",
    "request_validation",
    "validation_passed",
    "validation_failed",
    "complete_task",
    "pause",
    "resume",
    "normal_message",
]


class TransitionRecord(BaseModel):
    """One transition attempt — successful or blocked.

    Blocked attempts are kept on purpose: they are the proof that the state
    machine rejected an illegal move. ``state`` is NOT changed by a blocked
    attempt, only the history grows.
    """

    seq: int
    timestamp: str
    from_state: str
    to_state: str
    status: TransitionStatus
    reason: str = ""
    # "transition" | "prepare_plan" | "approve_plan" | "validation" |
    # "pause" | "resume" | "chat"
    action: str = "transition"
    # Where the attempt came from: a recognised natural-language intent, a UI
    # button or a direct API call. ``user_message`` keeps the raw phrase so the
    # log proves WHY the application tried the transition.
    intent: str = ""
    trigger: str = "api"  # "user_message" | "ui" | "api" | "system"
    user_message: str = ""


class TransitionOption(BaseModel):
    """A candidate transition shown in the UI, with its lock status.

    ``allowed=False`` means the transition is part of the lifecycle but a guard
    is not satisfied yet (e.g. ``execution`` from PLAN_APPROVAL before approval).
    """

    state: str
    allowed: bool
    locked_reason: str = ""
    requires: str = ""


class TransitionDecision(BaseModel):
    """Structured result of a transition request (the central guard output)."""

    allowed: bool
    from_state: str
    to_state: str
    reason: str = ""
    # Which rule produced the decision: "allowed_transitions", "guard",
    # "paused" or "not_allowed".
    rule: str = ""


class IntentResult(BaseModel):
    """The LLM's structured interpretation of ONE user message.

    It carries a semantic ``intent`` and a ``confidence`` — never a target
    state. The application converts the intent into an action; the state
    machine then decides whether that action is legal.

    ``error`` is EMPTY when the model was actually understood. It is set when
    the recognition itself failed (provider error, empty output, invalid JSON,
    unknown intent). This distinction matters: a recognition FAILURE is not
    the same as the user saying ``normal_message``, and the assistant must be
    able to report it instead of silently pretending nothing happened.
    """

    intent: IntentName = "normal_message"
    confidence: float = 0.0
    raw: str = ""
    error: str = ""

    @property
    def is_confident(self) -> bool:
        return (
            not self.error
            and self.intent != "normal_message"
            and self.confidence >= INTENT_MIN_CONFIDENCE
        )


class ChatMessage(BaseModel):
    """One entry of the HUMAN-readable execution journal of a task.

    This is deliberately separate from ``transitions`` (the technical
    transition log): the chat is what a human reads, the transition log is
    what the state machine records. Assistant WORK always produces one of
    these, so a state change never happens unseen.

    ``kind`` lets the UI tell the conversation apart from the debug/diagnostic
    entries without hiding either one:

    * ``message``   — a normal user/assistant conversational turn;
    * ``plan``      — the generated plan;
    * ``execution`` — the result of the current plan step;
    * ``validation``— the validation verdict;
    * ``transition``— an ALLOWED lifecycle transition;
    * ``blocked``   — a BLOCKED transition (state kept unchanged);
    * ``info``      — a short status note;
    * ``debug``     — the structured intent/decision trace for the demo UI.
    """

    seq: int = 0
    timestamp: str = ""
    role: Literal["user", "assistant", "system"] = "assistant"
    content: str = ""
    kind: str = "message"
    # Debug/diagnostic metadata attached to the turn (never authoritative).
    intent: str = ""
    confidence: float = 0.0
    from_state: str = ""
    to_state: str = ""
    status: str = ""  # "" | "allowed" | "blocked" | "no-op"
    error: str = ""


# Below this confidence an intent is treated as ``normal_message`` so an
# ambiguous message never mutates the lifecycle.
INTENT_MIN_CONFIDENCE = 0.5


class Day15TaskState(BaseModel):
    """Structured state of the CURRENT task with full lifecycle bookkeeping."""

    task_id: str
    goal: str = ""
    state: TaskStateName = "planning"
    # The state a pause was taken from. ``resume`` restores exactly this state,
    # so pause can never be used to jump to a different state.
    previous_state: TaskStateName | None = None
    current_step: int = 1
    expected_action: str = ""
    plan: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    plan_approved: bool = False
    # None = validation has not run yet; True/False = explicit result.
    validation_passed: bool | None = None
    validation_summary: str = ""
    # The concrete deliverable produced in the EXECUTION stage. It is the
    # input of the validation work, so the agent can actually check something.
    execution_result: str = ""
    paused: bool = False
    transitions: list[TransitionRecord] = Field(default_factory=list)
    # Human-readable execution journal (chat), separate from ``transitions``.
    chat_messages: list[ChatMessage] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_finished(self) -> bool:
        return self.state == "done"

    @property
    def total_steps(self) -> int:
        return len(self.plan)


# ----------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------
class Day15TaskCreateRequest(BaseModel):
    """Start a NEW task in the ``planning`` state."""

    goal: str = Field(..., min_length=1, max_length=2000)


class Day15PlanRequest(BaseModel):
    """Optionally override the goal when preparing a plan (usually empty)."""

    message: str = Field(default="", max_length=4000)


class Day15TransitionRequest(BaseModel):
    """Request an explicit lifecycle transition (validated by code)."""

    to_state: TaskStateName


class Day15ValidationRequest(BaseModel):
    """Record the outcome of a validation run (the guard's source of truth)."""

    passed: bool
    summary: str = Field(default="", max_length=2000)


class Day15ChatRequest(ChatRequest):
    """One assistant turn.

    Normal flow: the backend asks the LLM to recognise the user's semantic
    intent and then runs the resulting action through the state machine.

    ``proposed_transition`` is a low-level debug escape hatch (it still goes
    through the machine). ``forced_intent`` skips the LLM and is used by the
    debug UI / manual testing to demonstrate deterministic behaviour.
    """

    model: str | None = None
    proposed_transition: TaskStateName | None = None
    forced_intent: IntentName | None = None


# ----------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------
class Day15StateResponse(BaseModel):
    """Current task state + what the lifecycle allows right now."""

    model: str
    has_task: bool = False
    state: Day15TaskState | None = None
    states: list[str] = Field(
        default_factory=lambda: [
            "planning",
            "plan_approval",
            "execution",
            "validation",
            "done",
        ]
    )
    allowed_transitions: list[str] = Field(default_factory=list)
    transition_options: list[TransitionOption] = Field(default_factory=list)
    transition_history: list[TransitionRecord] = Field(default_factory=list)
    plan_approved: bool = False
    validation_passed: bool | None = None
    paused: bool = False


class Day15TransitionResponse(BaseModel):
    """Result of an explicit transition / control action.

    ``allowed`` is the structured verdict of the state machine. A blocked
    action returns this same shape (HTTP 200) so the UI can explain WHY,
    while the persisted state stays exactly where it was.
    """

    allowed: bool
    reason: str = ""
    action: str = "transition"
    state: Day15TaskState
    allowed_transitions: list[str] = Field(default_factory=list)
    transition_options: list[TransitionOption] = Field(default_factory=list)
    transition_history: list[TransitionRecord] = Field(default_factory=list)


class Day15ChatResponse(BaseModel):
    """Assistant answer + the state machine verdict for a proposed action.

    For the natural-language path it also carries the recognised intent and
    the raw user message so the UI can show the full reasoning chain:
    ``message -> intent -> action -> ALLOWED/BLOCKED -> state``.
    """

    answer: str
    model: str
    provider: str = "deepseek"
    state: Day15TaskState
    decision: TransitionDecision | None = None
    detected_intent: str = "normal_message"
    intent_confidence: float = 0.0
    # Non-empty when intent recognition itself FAILED (provider error, bad
    # JSON, unknown intent). Surfaced to the UI so the failure is never
    # silently shown as a confident ``normal_message``.
    intent_error: str = ""
    # Chat messages produced by THIS turn (user + every assistant work item +
    # the debug trace), so the UI can render the human conversation and the
    # state-machine trace as two visually separate streams.
    messages: list[ChatMessage] = Field(default_factory=list)
    user_message: str = ""
    trigger: str = "user_message"
    allowed_transitions: list[str] = Field(default_factory=list)
    transition_options: list[TransitionOption] = Field(default_factory=list)
    transition_history: list[TransitionRecord] = Field(default_factory=list)
