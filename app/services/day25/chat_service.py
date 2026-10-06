"""Day 25 chat orchestration: sessions + task memory + grounded RAG.

This is the application service behind the Web Chat. It owns the full
per-message pipeline and NEVER weakens Day 24: every technical answer still
goes through the Day 23 improved retrieval, the grounding gate and the
citation validator.

Flow for one user message::

    User Message
       ↓
    Load Chat Session / Task State
       ↓
    Save User Message
       ↓
    Task State Updater        (safe fallback to the previous state)
       ↓
    Contextual Query Builder  (current message + task state)
       ↓
    Day 23 Improved Retrieval → Day 24 Grounding Gate
       ├── insufficient → grounded refusal
       └── sufficient   → grounded LLM → citation validator
       ↓
    Save Assistant Message + Evidence + Task State
       ↓
    Return to Web UI
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.schemas.day24 import STATUS_ANSWERED, STATUS_INSUFFICIENT_CONTEXT
from app.schemas.day25 import (
    MSG_STATUS_ACKNOWLEDGED,
    MSG_STATUS_ANSWERED,
    MSG_STATUS_GROUNDING_FAILED,
    MSG_STATUS_INSUFFICIENT,
    MSG_STATUS_OK,
    ROLE_ASSISTANT,
    ROLE_USER,
    TURN_FACT_UPDATE,
    TURN_GOAL_CHANGE,
    ChatMessage,
    ChatSession,
    ChatTrace,
    CreateSessionRequest,
    MessageEvidence,
    SessionDetailResponse,
    SendMessageResponse,
    TaskState,
)
from app.services.day25.contextual_query import (
    build_contextual_query,
    derive_active_topic,
)
from app.services.day25.repository import ChatRepository
from app.services.day25.turn_analysis import (
    evidence_topic_hint,
    is_context_dependent_followup,
    is_explicit_topic_switch,
)
from app.services.day25.task_state import (
    default_task_state,
    state_to_prompt_text,
)
from app.services.day25.updater import (
    LLMTaskStateExtractor,
    RuleBasedTaskStateExtractor,
    TaskStateUpdater,
)
from app.services.rag.answer_service import RagRetrievalError
from app.services.rag.generation import GenerationError
from app.services.rag.grounding.service import GroundedRagService

logger = logging.getLogger("app.services.day25.chat_service")

DEFAULT_RECENT_MESSAGES = 6
_TITLE_FALLBACK = "Новый чат"


class ChatServiceError(Exception):
    """Chat error with a browser-safe message + HTTP status code."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SessionNotFoundError(ChatServiceError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Чат-сессия '{session_id}' не найдена.", status_code=404)


def _truncate_title(text: str) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return _TITLE_FALLBACK
    if len(cleaned) <= 60:
        return cleaned
    return cleaned[:57].rstrip() + "…"


def _citation_to_evidence(citation, ordinal: int) -> MessageEvidence:
    return MessageEvidence(
        message_id="",
        chunk_id=citation.chunk_id,
        source_type=citation.source_type or "",
        source=citation.source or "",
        section=citation.section or "",
        page=citation.page,
        page_from=citation.page_from,
        page_to=citation.page_to,
        message_ids=list(citation.message_ids or []),
        date_from=citation.date_from or "",
        date_to=citation.date_to or "",
        chat_name=citation.chat_name or "",
        quote=citation.quote or "",
        quote_valid=bool(citation.quote_valid),
        chunk_text=citation.chunk_text or "",
        ordinal=ordinal,
    )


class Day25ChatService:
    """Persistent, stateful RAG chat with task memory."""

    def __init__(
        self,
        settings: Settings,
        *,
        repository: ChatRepository | None = None,
        grounded_service: GroundedRagService | None = None,
        updater: TaskStateUpdater | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository or ChatRepository(settings.day25_chat_db_path)
        self.grounded = grounded_service or GroundedRagService(settings)
        if updater is not None:
            self.updater = updater
        else:
            self.updater = TaskStateUpdater(self._build_extractor())

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    @property
    def recent_messages(self) -> int:
        return int(
            getattr(self.settings, "day25_recent_messages", DEFAULT_RECENT_MESSAGES)
        )

    @property
    def chat_similarity_threshold(self) -> float:
        """Optional chat-only threshold (defaults to the Day 24 threshold)."""
        value = getattr(self.settings, "rag_chat_similarity_threshold", None)
        if value is None:
            return self.grounded.answer_service.default_similarity_threshold
        return float(value)

    def _build_extractor(self):
        """Prefer the LLM extractor; fall back to deterministic rules.

        The generation provider is built lazily (never at construction), so
        this method only assembles the extractor object. If the provider is
        unavailable the updater catches the error and keeps the previous state.
        """
        try:
            generation = self.grounded.answer_service.generation
        except Exception:  # noqa: BLE001 - rules-only mode is still valid
            return RuleBasedTaskStateExtractor()
        if (getattr(self.settings, "day25_state_extractor", "llm") or "llm").lower() == "rules":
            return RuleBasedTaskStateExtractor()
        return LLMTaskStateExtractor(generation)

    def status(self) -> dict:
        base = self.grounded.answer_service.status()
        return {
            "generation_provider": base.get("generation_provider", ""),
            "generation_model": base.get("generation_model", ""),
            "embedding_model": base.get("embedding_model", ""),
            "index_path": base.get("index_path", ""),
            "index_exists": bool(base.get("index_exists", False)),
            "chunks": int(base.get("chunks", 0) or 0),
            "recent_messages": self.recent_messages,
            "chat_db_path": str(self.repository.db_path),
            "demo_opening_question": getattr(
                self.settings, "day25_demo_opening_question", ""
            ),
            "upgrade_note": (
                "Task state хранится отдельно от knowledge base и не "
                "индексируется через embeddings."
            ),
        }

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------
    def create_session(self, request: CreateSessionRequest | None = None) -> ChatSession:
        title = request.title if request else None
        session = self.repository.create_session(title)
        self.repository.save_state(session.id, default_task_state())
        return session

    def list_sessions(self) -> list[ChatSession]:
        # Evaluation runs live in the same DB but must not pollute the user's
        # chat list; they are prefixed with ``[eval] ``.
        return [
            session
            for session in self.repository.list_sessions()
            if not session.title.startswith("[eval]")
        ]

    def get_session(self, session_id: str) -> ChatSession:
        session = self.repository.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def delete_session(self, session_id: str) -> None:
        if not self.repository.delete_session(session_id):
            raise SessionNotFoundError(session_id)

    def get_state(self, session_id: str) -> TaskState:
        self.get_session(session_id)
        return self.repository.get_state(session_id)

    def get_detail(self, session_id: str) -> SessionDetailResponse:
        session = self.get_session(session_id)
        messages = self.repository.list_messages(session_id)
        evidence = self.repository.list_evidence_for_session(session_id)
        for message in messages:
            message.evidence = evidence.get(message.id, [])
        return SessionDetailResponse(
            session=session,
            messages=messages,
            state=self.repository.get_state(session_id),
            task_state_updated_at=self.repository.get_state_updated_at(session_id),
        )

    # ------------------------------------------------------------------
    # message pipeline
    # ------------------------------------------------------------------
    async def send_message(self, session_id: str, content: str) -> SendMessageResponse:
        session = self.get_session(session_id)
        history = self.repository.list_messages(session_id)
        previous_state = self.repository.get_state(session_id)

        user_message = self.repository.add_message(
            session_id, ROLE_USER, content, status=MSG_STATUS_OK
        )
        if session.title in (_TITLE_FALLBACK, "", None) and not any(
            message.role == ROLE_USER for message in history
        ):
            session = self.repository.set_title(session_id, _truncate_title(content)) or session

        evidence_by_message = self.repository.list_evidence_for_session(session_id)
        previous_evidence = self._previous_evidence(evidence_by_message, history)
        followup = is_context_dependent_followup(content)
        topic_switch = is_explicit_topic_switch(content)

        state, update_info = await self._update_state(
            previous_state, content, turn_type_hint=None
        )

        # The history participates BEFORE retrieval, not only in the final
        # prompt. This is what makes a short follow-up self-contained.
        recent_window = history[-self.recent_messages :]
        contextual_query = build_contextual_query(
            content,
            state,
            recent_messages=recent_window,
            previous_evidence=previous_evidence,
        )
        previous_topic_hint = evidence_topic_hint(previous_evidence)
        recent = [
            {"role": message.role, "content": message.content}
            for message in recent_window
        ]

        is_question = update_info.turn_type != TURN_FACT_UPDATE or "?" in content
        grounded = None
        if is_question:
            try:
                grounded = await self.grounded.ask(
                    content,
                    retrieval_query=contextual_query,
                    task_state_text=state_to_prompt_text(state),
                    recent_messages=recent,
                    similarity_threshold=self.chat_similarity_threshold,
                    expand_context=True,
                    resolve_quote_chunk=True,
                    allow_partial_citations=True,
                )
            except (GenerationError, RagRetrievalError) as exc:
                # The user message is already stored; the session stays valid.
                # No fake assistant answer is written.
                logger.warning("Day 25 generation/retrieval failed: %s", exc)
                raise ChatServiceError(exc.message, status_code=exc.status_code) from exc
            assistant_content, assistant_status = self._assistant_reply(grounded)
            citations = list(grounded.citations or [])
        else:
            assistant_content, assistant_status = self._acknowledge(content, state, update_info)
            citations = []

        # Update the short active topic for the NEXT turn. Fact updates never
        # replace it; a follow-up keeps it; a switch starts a new one.
        if is_question or topic_switch:
            state.active_topic = derive_active_topic(
                content,
                previous_state.active_topic,
                evidence=(grounded.citations if grounded else None),
                followup=followup,
                topic_switch=topic_switch,
            )
        else:
            state.active_topic = previous_state.active_topic

        trace = self._build_trace(
            content=content,
            contextual_query=contextual_query,
            update_info=update_info,
            grounded=grounded,
            is_question=is_question,
            state=state,
            previous_topic_hint=previous_topic_hint,
            followup=followup,
            topic_switch=topic_switch,
        )
        self._log_turn(
            session_id=session_id,
            content=content,
            previous_goal=previous_state.goal,
            current_goal=state.goal,
            update_info=update_info,
            trace=trace,
            grounded=grounded,
        )
        assistant_message = self.repository.add_message(
            session_id,
            ROLE_ASSISTANT,
            assistant_content,
            status=assistant_status,
            trace=trace.model_dump(),
        )
        evidence = [
            _citation_to_evidence(citation, ordinal=index)
            for index, citation in enumerate(citations)
        ]
        for item in evidence:
            item.message_id = assistant_message.id
        self.repository.add_evidence(session_id, assistant_message.id, evidence)
        assistant_message.evidence = evidence

        self.repository.save_state(session_id, state)
        refreshed = self.repository.get_session(session_id) or session
        return SendMessageResponse(
            session=refreshed,
            user_message=user_message,
            assistant_message=assistant_message,
            state=state,
            grounded=grounded,
        )

    @staticmethod
    def _previous_evidence(evidence_by_message, history) -> list:
        """Evidence of the most recent answered turn (topic continuity only)."""
        for message in reversed(history):
            entries = evidence_by_message.get(message.id) or []
            if entries:
                return list(entries)
        return []

    async def _update_state(
        self, previous: TaskState, content: str, *, turn_type_hint: str | None
    ) -> tuple[TaskState, object]:
        state, info = await self.updater.update(previous, content)
        if turn_type_hint:
            info.turn_type = turn_type_hint
        topic_switch = is_explicit_topic_switch(content)
        if topic_switch:
            # A genuinely new task: archive the old task-specific memory.
            if previous.goal:
                state = self._reset_for_goal_change(state)
            info.turn_type = TURN_GOAL_CHANGE
        elif info.turn_type == TURN_GOAL_CHANGE:
            # An update_goal produced by the extractor is a refinement of the
            # SAME task, not a new one. Keep all memory.
            info.turn_type = TURN_QUESTION
        if not topic_switch and info.turn_type != TURN_GOAL_CHANGE:
            # The extractor must never wipe the active topic mid-task; only a
            # real switch or the post-generation derivation may change it.
            state.active_topic = previous.active_topic
        return state, info

    @staticmethod
    def _reset_for_goal_change(state: TaskState) -> TaskState:
        """Archive task-specific memory while keeping vehicle-level facts.

        Vehicle fields (model/year/engine/transmission/mileage) describe the
        CAR and survive a topic change. Everything that belongs to the OLD
        goal is dropped so the two tasks are never mixed silently. The user
        can always open the original messages — task state is derived memory,
        not the source of truth.
        """
        return state.model_copy(
            update={
                "active_topic": None,
                "known_facts": [
                    fact
                    for fact in state.known_facts
                    if fact.key and fact.key.startswith("vehicle_")
                ],
                "constraints": list(state.constraints),
                "terms": list(state.terms),
                "checks_performed": [],
                "results": [],
                "hypotheses": [],
                "open_questions": [],
            }
        )

    @staticmethod
    def _assistant_reply(grounded) -> tuple[str, str]:
        if grounded.status == STATUS_ANSWERED:
            return (grounded.answer or "", MSG_STATUS_ANSWERED)
        if grounded.status == STATUS_INSUFFICIENT_CONTEXT:
            return (
                grounded.message or "Недостаточно контекста для ответа.",
                MSG_STATUS_INSUFFICIENT,
            )
        return (
            grounded.message or "Ответ не подтверждён источниками.",
            MSG_STATUS_GROUNDING_FAILED,
        )

    @staticmethod
    def _acknowledge(content: str, state: TaskState, update_info) -> tuple[str, str]:
        """Reply to an informational (non-question) turn WITHOUT the LLM.

        A short fact update ("Пробег 82 тысячи.") primarily updates task
        memory; forcing a cited technical answer for it would be noise.
        """
        prefix = "Принял"
        if update_info.turn_type == TURN_FACT_UPDATE:
            prefix = "Принял, записал"
        parts = [f"{prefix}: {content.strip()}"]
        open_questions = state.open_questions
        if open_questions:
            parts.append("Открытые вопросы: " + "; ".join(open_questions[:3]) + ".")
        parts.append("Можете задать следующий вопрос.")
        return " ".join(parts), MSG_STATUS_ACKNOWLEDGED

    @staticmethod
    def _build_trace(
        *,
        content: str,
        contextual_query: str,
        update_info,
        grounded,
        is_question: bool,
        state: TaskState,
        previous_topic_hint: str = "",
        followup: bool = False,
        topic_switch: bool = False,
    ) -> ChatTrace:
        trace = ChatTrace(
            turn_type=update_info.turn_type,
            current_message=content,
            contextual_query=contextual_query,
            task_state_error=update_info.error,
            task_operations=list(update_info.operations),
            active_topic=state.active_topic or "",
            previous_topic_hint=previous_topic_hint,
            followup=followup,
            topic_switch=topic_switch,
        )
        if grounded is None:
            trace.pipeline = [
                "User Message",
                "Task State Updater",
                "Informational turn: answer not generated",
            ]
            return trace
        retrieval = grounded.retrieval
        trace.rewritten_query = retrieval.rewritten_query or ""
        trace.retrieval_query = (
            retrieval.rewrite.original_query if retrieval.rewrite else contextual_query
        )
        trace.retrieved_count = retrieval.retrieved_count
        trace.accepted_count = retrieval.accepted_count
        trace.used_count = retrieval.context_count
        trace.grounding = retrieval.gate_reason
        trace.grounding_status = grounded.status
        trace.pipeline = list(grounded.pipeline or [])
        return trace

    @staticmethod
    def _log_turn(
        *,
        session_id: str,
        content: str,
        previous_goal: str | None,
        current_goal: str | None,
        update_info,
        trace: ChatTrace,
        grounded,
    ) -> None:
        """Non-sensitive turn diagnostics for /xpander debugging.

        Only ids, statuses, counts, scores and source file names are logged;
        raw Telegram content/authors are never written.
        """
        retrieval = grounded.retrieval if grounded is not None else None
        top_scores = []
        if retrieval is not None:
            top_scores = [
                round(float(c.similarity), 4)
                for c in retrieval.retrieved_candidates[:5]
            ]
        sources = sorted({c.source for c in (grounded.sources if grounded else [])})
        logger.info(
            "day25 turn session=%s followup=%s switch=%s turn_type=%s "
            "prev_goal=%r new_goal=%r active_topic=%r "
            "current=%r contextual=%r rewritten=%r top=%s accepted=%s "
            "used=%s gate=%s llm_called=%s status=%s sources=%s",
            session_id,
            trace.followup,
            trace.topic_switch,
            update_info.turn_type,
            previous_goal,
            current_goal,
            trace.active_topic,
            content[:160],
            trace.contextual_query[:400],
            trace.rewritten_query[:200],
            top_scores,
            retrieval.accepted_count if retrieval else 0,
            retrieval.context_count if retrieval else 0,
            retrieval.gate_reason if retrieval else "",
            grounded.llm_called if grounded else False,
            grounded.status if grounded else "",
            sources,
        )
