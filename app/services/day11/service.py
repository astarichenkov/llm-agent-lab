"""Day 11 — Memory Layers service (orchestration).

Flow of ONE turn (the core teaching pipeline):

    user message
      -> short-term memory records it DETERMINISTICALLY (no LLM)
      -> MemoryClassifier decides what (if anything) goes to
         working memory / long-term memory
      -> Context Builder assembles:
             system + long-term block + working block
             + short-term window (existing Day 10 context strategy)
             + current user message
      -> existing DeepSeekService generates the answer
      -> assistant message is appended to short-term memory

The service reuses:
* the existing ``DeepSeekService`` (no second client);
* the existing ``SlidingWindowStrategy`` from Day 10 as the context strategy;
* the Day 8 token estimator and model cost helper.

It does NOT touch the Day 7 SQLite agent history.
"""
from __future__ import annotations

import logging
from uuid import uuid4

from app.config import Settings
from app.schemas.day11 import (
    ClassifierDecision,
    Day11ChatResponse,
    Day11ContextInfo,
    Day11Cost,
    Day11EvaluateResponse,
    Day11MemoryState,
    Day11ScenarioResponse,
    Day11Usage,
    LongTermMemory,
    ShortTermMessage,
    WorkingMemory,
)
from app.services import model_params
from app.services.day10.strategies import SlidingWindowStrategy
from app.services.day11.classifier import MemoryClassifier
from app.services.day11.scenario import (
    EXPECTED_FACTS_LABELS,
    SCENARIO_MESSAGES,
    evaluate_answer,
)
from app.services.day11.store import (
    LongTermMemoryStore,
    MemorySession,
    apply_long_term_update,
    apply_working_update,
)
from app.services.deepseek import DeepSeekService
from app.services.token_estimate import (
    estimate_message_tokens,
    estimate_messages_tokens,
)

logger = logging.getLogger("app.services.day11")

_PREVIEW_LIMIT = 16


def _new_session_id() -> str:
    return f"sess-{uuid4().hex[:8]}"


class Day11MemoryService:
    """Owns the three memory layers plus the context builder."""

    def __init__(
        self,
        settings: Settings,
        deepseek: DeepSeekService | None = None,
        classifier: MemoryClassifier | None = None,
        long_term_store: LongTermMemoryStore | None = None,
    ) -> None:
        self._settings = settings
        self._deepseek = deepseek if deepseek is not None else DeepSeekService(settings)
        self._classifier = classifier if classifier is not None else MemoryClassifier()
        self._long_term_store = long_term_store or LongTermMemoryStore(
            settings.day11_long_term_path
        )
        # Long-term memory is loaded once and is SHARED by every session.
        self._long_term = self._long_term_store.load()
        self._sessions: dict[str, MemorySession] = {}
        self._active_session_id = ""
        # Reuse the Day 10 context strategy as a pure window selector.
        self._window = SlidingWindowStrategy(window_size=settings.day11_window_size)
        self._last_decision: ClassifierDecision | None = None
        self.new_session()

    # ------------------------------------------------------------------
    # sessions / state
    # ------------------------------------------------------------------
    def new_session(self) -> Day11MemoryState:
        """Start a fresh session: short-term and working are empty.

        Long-term memory is NOT touched — it is shared across sessions.
        """
        session = MemorySession(id=_new_session_id())
        self._sessions[session.id] = session
        self._active_session_id = session.id
        self._last_decision = None
        logger.info("day11 new session id=%s", session.id)
        return self.state()

    def _ensure_session(self, session_id: str | None) -> MemorySession:
        if session_id and session_id in self._sessions:
            self._active_session_id = session_id
        elif session_id and session_id not in self._sessions:
            session = MemorySession(id=session_id)
            self._sessions[session_id] = session
            self._active_session_id = session_id
        elif self._active_session_id not in self._sessions:
            self.new_session()
        return self._sessions[self._active_session_id]

    def _active(self) -> MemorySession:
        return self._sessions[self._active_session_id]

    def activate_session(self, session_id: str) -> Day11MemoryState:
        """Make ``session_id`` the active session (creating it if needed).

        Used by the Day 12 profile service to run an isolated comparison in
        temporary sessions and then restore the user's previous session.
        """
        self._ensure_session(session_id)
        return self.state()

    @property
    def model(self) -> str:
        return self._settings.deepseek_model

    def state(self) -> Day11MemoryState:
        session = self._active()
        return Day11MemoryState(
            session_id=session.id,
            short_term=[ShortTermMessage(**m) for m in session.short_term],
            short_term_count=len(session.short_term),
            working=session.working,
            long_term=self._long_term,
            last_decision=self._last_decision,
        )

    # ------------------------------------------------------------------
    # explicit, independent clears
    # ------------------------------------------------------------------
    def reset_short_term(self) -> Day11MemoryState:
        """Clear ONLY short-term memory (working + long-term untouched)."""
        self._active().short_term = []
        logger.info("day11 short-term cleared session=%s", self._active().id)
        return self.state()

    def reset_working(self) -> Day11MemoryState:
        """Clear ONLY working memory (short-term + long-term untouched)."""
        self._active().working = WorkingMemory()
        logger.info("day11 working memory cleared session=%s", self._active().id)
        return self.state()

    def reset_long_term(self) -> Day11MemoryState:
        """Clear ONLY long-term memory (short-term + working untouched)."""
        self._long_term = LongTermMemory()
        self._long_term_store.clear()
        logger.info("day11 long-term memory cleared")
        return self.state()

    # ------------------------------------------------------------------
    # deterministic seeding (tests / demo shortcut, no provider call)
    # ------------------------------------------------------------------
    def seed_short_term(self, messages: list[dict[str, str]]) -> Day11MemoryState:
        session = self._active()
        session.short_term = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        return self.state()

    def set_working(self, working: WorkingMemory) -> Day11MemoryState:
        self._active().working = working
        return self.state()

    def set_long_term(self, long_term: LongTermMemory) -> Day11MemoryState:
        self._long_term = long_term
        self._long_term_store.save(self._long_term)
        return self.state()

    def seed_demo(self) -> Day11MemoryState:
        """Install the deterministic demo memory WITHOUT any provider call.

        Opening facts are placed in short-term, working and long-term memory
        so the UI can immediately show the final answer still respecting them
        even after the opening message leaves the sliding window.
        """
        self.new_session()
        session = self._active()
        history: list[dict[str, str]] = []
        for index, message in enumerate(SCENARIO_MESSAGES[:-1]):
            history.append({"role": "user", "content": message})
            if index == 0:
                history.append(
                    {"role": "assistant", "content": "Понял: Антон, Python, кратко, сервис бронирования, без PostgreSQL."}
                )
            elif not message.startswith(("Спасибо", "Хорошо", "Продолжай", "Давай")):
                history.append({"role": "assistant", "content": "Принято."})
        session.short_term = history
        session.working = WorkingMemory(
            goal="Спроектировать сервис бронирования",
            constraints=["Не использовать PostgreSQL"],
        )
        self._long_term = apply_long_term_update(
            self._long_term,
            {"preferred_language": "Python", "answer_style": "concise"},
        )
        self._long_term_store.save(self._long_term)
        logger.info("day11 demo memory seeded session=%s", session.id)
        return self.state()

    # ------------------------------------------------------------------
    # context builder
    # ------------------------------------------------------------------
    def _build_context(
        self,
        *,
        session: MemorySession,
        current_user_message: str,
        system: str | None,
        strategy: str,
        window_size: int | None,
        include_long_term: bool,
        include_working: bool,
        personalization_block: str | None = None,
    ) -> tuple[list[dict[str, str]], Day11ContextInfo, list[dict[str, str]]]:
        # Short-term memory already contains the current user message (added
        # before classification). Exclude it so it is added exactly once.
        history = [dict(m) for m in session.short_term[:-1]]

        if strategy == "full":
            sent = history
            dropped: list[dict[str, str]] = []
            effective_window: int | None = None
        else:
            effective_window = window_size or self._settings.day11_window_size
            self._window.window_size = effective_window
            self._window.full_history = list(history)
            # Execute the EXISTING Day 10 context strategy to select the window.
            selector = self._window.build_context(
                system_prompt=None, user_message=""
            )
            sent = [dict(m) for m in selector.history_messages]
            dropped = [dict(m) for m in selector.dropped_messages]

        long_block = (
            self._long_term.to_block()
            if include_long_term and not self._long_term.is_empty()
            else None
        )
        working_block = (
            session.working.to_block()
            if include_working and not session.working.is_empty()
            else None
        )

        messages: list[dict[str, str]] = []
        included_layers: list[str] = []
        if system:
            messages.append({"role": "system", "content": system})
            included_layers.append("system")
        # Day 12 — personalization comes FIRST after the application system
        # prompt and BEFORE any memory/context, so system rules keep priority.
        if personalization_block:
            messages.append({"role": "system", "content": personalization_block})
            included_layers.append("personalization")
        if long_block:
            messages.append({"role": "system", "content": long_block})
            included_layers.append("long_term")
        if working_block:
            messages.append({"role": "system", "content": working_block})
            included_layers.append("working")
        messages.extend(sent)
        included_layers.append("short_term")
        messages.append({"role": "user", "content": current_user_message})
        included_layers.append("current_user")

        info = Day11ContextInfo(
            strategy=strategy,  # type: ignore[arg-type]
            total_short_term_messages=len(session.short_term),
            sent_short_term_messages=len(sent),
            dropped_messages=len(dropped),
            dropped_preview=[ShortTermMessage(**m) for m in dropped[-_PREVIEW_LIMIT:]],
            sent_preview=[ShortTermMessage(**m) for m in sent[-_PREVIEW_LIMIT:]],
            window_size=effective_window,
            long_term_included=long_block is not None,
            working_included=working_block is not None,
            personalization_included=bool(personalization_block),
            included_layers=included_layers,
            personalization_block=personalization_block,
            long_term_block=long_block,
            working_block=working_block,
        )
        return messages, info, sent

    # ------------------------------------------------------------------
    # usage / cost
    # ------------------------------------------------------------------
    def _build_usage(
        self,
        info: Day11ContextInfo,
        *,
        system: str | None,
        current_user_message: str,
        sent_history: list[dict[str, str]],
        provider_usage: dict | None,
        classifier_usage: dict | None,
        personalization_block: str | None = None,
    ) -> Day11Usage:
        system_tokens = estimate_message_tokens(system) if system else 0
        personalization_tokens = (
            estimate_message_tokens(personalization_block)
            if personalization_block
            else 0
        )
        long_tokens = (
            estimate_message_tokens(info.long_term_block)
            if info.long_term_block
            else 0
        )
        working_tokens = (
            estimate_message_tokens(info.working_block)
            if info.working_block
            else 0
        )
        short_tokens = estimate_messages_tokens(sent_history)
        current_tokens = (
            estimate_message_tokens(current_user_message)
            if current_user_message
            else 0
        )
        provider_usage = provider_usage or {}
        classifier_usage = classifier_usage or {}
        return Day11Usage(
            prompt_tokens=provider_usage.get("prompt_tokens"),
            completion_tokens=provider_usage.get("completion_tokens"),
            total_tokens=provider_usage.get("total_tokens"),
            classifier_prompt_tokens=classifier_usage.get("prompt_tokens"),
            classifier_completion_tokens=classifier_usage.get("completion_tokens"),
            classifier_total_tokens=classifier_usage.get("total_tokens"),
            system_prompt_tokens_estimated=system_tokens,
            personalization_tokens_estimated=personalization_tokens,
            long_term_tokens_estimated=long_tokens,
            working_tokens_estimated=working_tokens,
            short_term_tokens_estimated=short_tokens,
            current_user_tokens_estimated=current_tokens,
            estimated_input_tokens=(
                system_tokens
                + personalization_tokens
                + long_tokens
                + working_tokens
                + short_tokens
                + current_tokens
            ),
        )

    def _cost(self, model: str, usage: Day11Usage) -> Day11Cost:
        raw = model_params.estimate_cost(
            model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return Day11Cost(
            input=float(raw["input"]) if raw["input"] is not None else None,
            output=float(raw["output"]) if raw["output"] is not None else None,
            total=float(raw["total"]) if raw["total"] is not None else None,
            currency=raw["currency"],
            estimated=raw["estimated"],
            reason=raw.get("reason"),
        )

    # ------------------------------------------------------------------
    # main operation
    # ------------------------------------------------------------------
    async def chat(
        self,
        *,
        message: str,
        session_id: str | None = None,
        strategy: str = "sliding_window",
        window_size: int | None = None,
        use_memory: bool = True,
        classify: bool = True,
        include_long_term: bool = True,
        include_working: bool = True,
        model: str | None = None,
        system_prompt: str | None = None,
        personalization_block: str | None = None,
    ) -> Day11ChatResponse:
        session = self._ensure_session(session_id)
        resolved_model = model or self._settings.deepseek_model
        system = (
            system_prompt if system_prompt is not None else self._settings.system_prompt
        )

        # 1) short-term memory is deterministic and ALWAYS records the turn.
        session.short_term.append({"role": "user", "content": message})

        # 2) MemoryClassifier decides what goes to working / long-term memory.
        classifier_usage: dict = {}
        if use_memory and classify:
            logger.info(
                "day11 classifier input session=%s message_chars=%s "
                "working_entries=%s long_term_entries=%s",
                session.id,
                len(message),
                session.working.count(),
                self._long_term.count(),
            )
            decision, classifier_usage = await self._classifier.classify(
                user_message=message,
                working=session.working,
                long_term=self._long_term,
                provider=self._deepseek,
                model=resolved_model,
                max_tokens=self._settings.day11_classifier_max_tokens,
            )
            logger.info(
                "day11 classifier decision performed=%s nothing_to_save=%s "
                "working_update=%s long_term_update=%s error=%s",
                decision.performed,
                decision.nothing_to_save,
                decision.working_memory.model_dump(),
                decision.long_term_memory,
                decision.error,
            )
        else:
            decision = ClassifierDecision(
                performed=False,
                nothing_to_save=True,
                error=None,
            )
        self._last_decision = decision

        if decision.performed and not decision.nothing_to_save:
            session.working = apply_working_update(
                session.working, decision.working_memory
            )
            if decision.long_term_memory:
                self._long_term = apply_long_term_update(
                    self._long_term, decision.long_term_memory
                )
                self._long_term_store.save(self._long_term)

        # 3) build the context explicitly from the memory layers.
        eff_long_term = use_memory and include_long_term
        eff_working = use_memory and include_working
        messages, info, sent_history = self._build_context(
            session=session,
            current_user_message=message,
            system=system,
            strategy=strategy,
            window_size=window_size,
            include_long_term=eff_long_term,
            include_working=eff_working,
            personalization_block=personalization_block,
        )

        # 4) main answer call (existing client).
        limits = model_params.get_limits(resolved_model)
        content, finish_reason, provider_usage = await self._deepseek.generate(
            messages,
            model=resolved_model,
            max_tokens=limits.max_output_tokens,
            thinking=False,
        )

        # 5) commit the assistant turn to short-term memory.
        session.short_term.append({"role": "assistant", "content": content})

        # Rebuild the context info so counts/preview include the assistant turn.
        info.total_short_term_messages = len(session.short_term)

        usage = self._build_usage(
            info,
            system=system,
            current_user_message=message,
            sent_history=sent_history,
            provider_usage=provider_usage,
            classifier_usage=classifier_usage,
            personalization_block=personalization_block,
        )

        logger.info(
            "day11 turn completed session=%s strategy=%s model=%s "
            "short_term=%s sent=%s dropped=%s layers=%s classifier_performed=%s "
            "nothing_to_save=%s working_written=%s long_term_written=%s "
            "finish_reason=%s prompt_tokens=%s completion_tokens=%s",
            session.id,
            strategy,
            resolved_model,
            len(session.short_term),
            info.sent_short_term_messages,
            info.dropped_messages,
            ",".join(info.included_layers),
            decision.performed,
            decision.nothing_to_save,
            decision.working_memory.count(),
            len(decision.long_term_memory),
            finish_reason,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        return Day11ChatResponse(
            session_id=session.id,
            answer=content,
            model=resolved_model,
            finish_reason=finish_reason,
            usage=usage,
            cost=self._cost(resolved_model, usage),
            context=info,
            memory=self.state(),
            last_decision=decision,
            classifier_error=decision.error,
        )

    # ------------------------------------------------------------------
    # scenario helpers
    # ------------------------------------------------------------------
    def scenario(self) -> Day11ScenarioResponse:
        return Day11ScenarioResponse(
            messages=list(SCENARIO_MESSAGES),
            expected=list(EXPECTED_FACTS_LABELS),
        )

    def evaluate(
        self, answer: str, expected: list[str] | None = None
    ) -> Day11EvaluateResponse:
        return Day11EvaluateResponse(**evaluate_answer(answer, expected))
