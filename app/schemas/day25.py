"""Pydantic schemas for Week 5 / Day 25 — Mini Chat with RAG + Task Memory.

Day 25 adds *stateful* conversation on top of the Day 24 grounded pipeline.
It deliberately keeps three different kinds of context apart:

* **Conversation history** — the raw ``user`` / ``assistant`` messages that
  were actually exchanged (persisted per session);
* **Task state** — long-lived, structured memory about the current task
  (goal, vehicle, known facts, constraints, checks, results, hypotheses,
  open questions). It is NEVER embedded into the Day 21 vector index;
* **RAG context** — chunks retrieved freshly for every new meaningful user
  turn through the Day 23 improved retrieval + Day 24 grounding gate.

The models here are the single contract shared by the repository, the chat
service, the HTTP endpoints, the evaluation runner and the UI.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.day24 import GroundedRAGResult

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLES = (ROLE_USER, ROLE_ASSISTANT)

# Assistant message statuses (also stored in ``chat_messages.status``).
MSG_STATUS_OK = "ok"
MSG_STATUS_ANSWERED = "answered"
MSG_STATUS_INSUFFICIENT = "insufficient_context"
MSG_STATUS_GROUNDING_FAILED = "grounding_failed"
MSG_STATUS_ACKNOWLEDGED = "acknowledged"
MSG_STATUS_ERROR = "error"

# Turn classification (lightweight, derived from the task-state operations).
TURN_QUESTION = "question"
TURN_FACT_UPDATE = "fact_update"
TURN_CORRECTION = "correction"
TURN_GOAL_CHANGE = "goal_change"

VEHICLE_DEFAULT_MODEL = "Mitsubishi Xpander"
MAX_MESSAGE_LENGTH = 4000
MAX_TITLE_LENGTH = 80


# ----------------------------------------------------------------------
# Task state
# ----------------------------------------------------------------------
class VehicleState(BaseModel):
    """What is known about the vehicle. Empty fields mean "not yet known"."""

    model: str = VEHICLE_DEFAULT_MODEL
    year: int | None = None
    engine: str | None = None
    transmission: str | None = None
    mileage_km: int | None = None


class FactItem(BaseModel):
    """One active fact plus an optional stable ``key`` for corrections.

    ``key`` (e.g. ``"mileage"``, ``"vibration_condition"``) lets the updater
    SUPERSEDE an earlier value instead of accumulating two contradicting
    facts. ``origin`` is always ``user`` for facts stated by the user.
    """

    text: str
    key: str | None = None
    origin: str = "user"


class ValueItem(BaseModel):
    text: str
    origin: str = "user"


class TaskState(BaseModel):
    """Persistent structured task memory for ONE chat session."""

    goal: str | None = None
    # Short, retrieval-friendly description of what the conversation is
    # currently about (e.g. "защита картера и КПП Sheriff 5307/5308"). It is
    # what lets a short follow-up ("что делать?") retrieve the right topic.
    active_topic: str | None = None
    vehicle: VehicleState = Field(default_factory=VehicleState)
    known_facts: list[FactItem] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    checks_performed: list[ValueItem] = Field(default_factory=list)
    results: list[ValueItem] = Field(default_factory=list)
    hypotheses: list[ValueItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class StateOperation(BaseModel):
    """One deterministic update operation produced by the task-state extractor.

    The operation vocabulary maps 1:1 to the capabilities required by the
    Day 25 spec: ADD fact, UPDATE fact, SUPERSEDE fact, ADD constraint,
    ADD term, ADD performed check, ADD result, ADD hypothesis, UPDATE goal,
    ADD/remove open question.
    """

    op: Literal[
        "set_goal",
        "update_goal",
        "set_active_topic",
        "clear_active_topic",
        "add_fact",
        "update_fact",
        "supersede_fact",
        "remove_fact",
        "add_constraint",
        "add_term",
        "add_check",
        "add_result",
        "add_hypothesis",
        "update_vehicle",
        "add_open_question",
        "remove_open_question",
    ]
    text: str = ""
    key: str | None = None
    field: str | None = None
    value: str | int | float | None = None
    origin: str = "user"


class TaskUpdateInfo(BaseModel):
    """Diagnostics about one task-state update (shown in technical details)."""

    applied_operations: int = 0
    operations: list[StateOperation] = Field(default_factory=list)
    turn_type: str = TURN_QUESTION
    error: str | None = None


# ----------------------------------------------------------------------
# Chat session / messages / evidence
# ----------------------------------------------------------------------
class ChatSession(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageEvidence(BaseModel):
    """Persisted Day 24 evidence for one assistant answer.

    Historical answers show exactly the sources that were used when they were
    generated — reopening an old chat NEVER runs retrieval again.
    """

    id: int | None = None
    message_id: str
    chunk_id: str
    source_type: str = ""
    source: str = ""
    section: str = ""
    page: int | None = None
    page_from: int | None = None
    page_to: int | None = None
    message_ids: list[int] = Field(default_factory=list)
    date_from: str = ""
    date_to: str = ""
    chat_name: str = ""
    quote: str = ""
    quote_valid: bool = False
    chunk_text: str = ""
    ordinal: int = 0


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    status: str = MSG_STATUS_OK
    created_at: str
    trace: dict[str, Any] | None = None
    evidence: list[MessageEvidence] = Field(default_factory=list)


# ----------------------------------------------------------------------
# API requests / responses
# ----------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_TITLE_LENGTH)


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("content")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("content must not be blank")
        return value


class ChatTrace(BaseModel):
    """Debug/demo trace of the LAST turn (technical details block)."""

    turn_type: str = TURN_QUESTION
    current_message: str = ""
    contextual_query: str = ""
    rewritten_query: str = ""
    retrieval_query: str = ""
    active_topic: str = ""
    previous_topic_hint: str = ""
    followup: bool = False
    topic_switch: bool = False
    retrieved_count: int = 0
    accepted_count: int = 0
    used_count: int = 0
    grounding: str = ""
    grounding_status: str = ""
    task_state_error: str | None = None
    task_operations: list[StateOperation] = Field(default_factory=list)
    pipeline: list[str] = Field(default_factory=list)


class SendMessageResponse(BaseModel):
    session: ChatSession
    user_message: ChatMessage
    assistant_message: ChatMessage
    state: TaskState
    grounded: GroundedRAGResult | None = None


class SessionDetailResponse(BaseModel):
    session: ChatSession
    messages: list[ChatMessage] = Field(default_factory=list)
    state: TaskState
    task_state_updated_at: str = ""


class SessionListResponse(BaseModel):
    sessions: list[ChatSession] = Field(default_factory=list)


class TaskStateResponse(BaseModel):
    session_id: str
    state: TaskState
    updated_at: str = ""


class Day25StatusResponse(BaseModel):
    generation_provider: str
    generation_model: str
    embedding_model: str
    index_path: str
    index_exists: bool
    chunks: int = 0
    recent_messages: int = 0
    chat_db_path: str = ""
    demo_opening_question: str = ""
    upgrade_note: str = ""


# ----------------------------------------------------------------------
# Scenario evaluation
# ----------------------------------------------------------------------
class ScenarioTurn(BaseModel):
    user: str
    kind: str = TURN_QUESTION
    expect_rag: bool = True
    expected_state_contains: list[str] = Field(default_factory=list)
    # Keywords that MUST appear in the contextual retrieval query for a
    # deliberately incomplete follow-up ("А что проверить первым?").
    expect_contextual_terms: list[str] = Field(default_factory=list)
    # A correction turn: ``expect_active`` maps a state path to the value that
    # must be active at the END of the scenario (the old value must be gone).
    correction: bool = False
    expect_active: dict[str, str] = Field(default_factory=dict)
    # Earlier expected facts that this turn supersedes (must be gone at the end).
    supersedes: list[str] = Field(default_factory=list)
    expect_refusal: bool = False


class Scenario(BaseModel):
    id: str
    title: str
    initial_goal: str = ""
    description: str = ""
    turns: list[ScenarioTurn] = Field(default_factory=list)


class ScenarioMetrics(BaseModel):
    turns_total: int = 0
    turns_completed: int = 0
    goal_retained: bool = False
    facts_expected: int = 0
    facts_retained: int = 0
    memory_retention: str = "0/0"
    corrections_expected: int = 0
    corrections_applied: int = 0
    contextual_followups: int = 0
    contextual_followups_resolved: int = 0
    grounded_answers: int = 0
    answers_with_sources: int = 0
    quotes_valid: int = 0
    insufficient_context_turns: int = 0
    correct_refusals: int = 0


class ScenarioTurnResult(BaseModel):
    index: int
    user: str
    kind: str
    assistant_status: str = ""
    contextual_query: str = ""
    current_message: str = ""
    rewritten_query: str = ""
    sources: int = 0
    quotes_valid: bool = False
    state_contains: list[str] = Field(default_factory=list)


class ScenarioRunResult(BaseModel):
    scenario: Scenario
    session_id: str
    metrics: ScenarioMetrics
    turns: list[ScenarioTurnResult] = Field(default_factory=list)
    final_state: TaskState = Field(default_factory=TaskState)
    error: str | None = None
