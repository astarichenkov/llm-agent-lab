"""Day 15 — controlled state transitions service.

This service is the thin orchestration layer around the PURE
:class:`~app.services.day15.lifecycle.LifecycleMachine`:

    LLM proposes content / an action
            │
            ▼
    LifecycleMachine.evaluate(state, target)   <- deterministic guards
            │
       allowed? ── yes ──► LifecycleMachine.apply + persist + history(allowed)
            │
            └──────── no ──► state UNCHANGED + history(blocked, reason)

Every path that could change the lifecycle state goes through
:meth:`Day15LifecycleService.transition` / :meth:`approve_plan` /
:meth:`set_validation` / :meth:`pause` / :meth:`resume`, and every one of them
records the attempt in ``state.transitions``. A blocked attempt is written to
the history but never mutates ``state``, ``current_step`` or ``expected_action``.

The assistant (``chat``) can only PROPOSE a transition; the same machine
decides it. This is the assignment's central idea: the LLM is not the source
of truth about the task state.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.config import Settings
from app.schemas.day15 import (
    ChatMessage,
    Day15ChatResponse,
    Day15StateResponse,
    Day15TaskState,
    Day15TransitionResponse,
    IntentResult,
    TaskStateName,
    TransitionDecision,
    TransitionRecord,
)
from app.services.day15.agent import (
    build_answer_request,
    build_execution_request,
    build_validation_request,
    parse_verdict,
)
from app.services.day15.intent import (
    VALID_INTENTS,
    recognize_intent,
)
from app.services.day15.lifecycle import (
    STATES,
    LifecycleMachine,
    expected_action_for,
)
from app.services.day15.store import LifecycleStore
from app.services.deepseek import DeepSeekService
from app.services.day13.service import build_planning_request, parse_plan

logger = logging.getLogger("app.services.day15")

# Deterministic fallback plan so the demo works even without a provider.
DEFAULT_PLAN = [
    "Analyse the task and requirements",
    "Implement the main part",
    "Review and refine the result",
]


class Day15Error(Exception):
    """Raised for invalid operations on the current task (no task, etc.)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_task_id() -> str:
    return f"task-{uuid4().hex[:8]}"


# Human-readable Russian labels for the assistant's natural-language reply.
_INTENT_LABEL_RU: dict[str, str] = {
    "continue_planning": "подготовка плана",
    "approve_plan": "утверждение плана",
    "request_plan_changes": "доработка плана",
    "revise_plan": "возврат к планированию (пересмотр плана)",
    "start_execution": "запуск выполнения",
    "request_validation": "запуск проверки",
    "validation_passed": "успешная проверка",
    "validation_failed": "провал проверки",
    "complete_task": "завершение задачи",
    "pause": "пауза",
    "resume": "продолжение",
    "normal_message": "обычное сообщение",
}


def build_assistant_response(
    state: Day15TaskState,
    intent: str,
    decision: TransitionDecision | None,
) -> str:
    """Build a natural-language reply that is DERIVED from the machine verdict.

    The wording can never contradict the decision: allowed/blocked is decided
    first by the state machine, and only then turned into text.
    """
    label = _INTENT_LABEL_RU.get(intent, intent)

    if decision is None:
        hint = "Напишите, что делать дальше"
        if state.paused:
            hint = "Напишите «продолжай», чтобы снять паузу"
        elif state.state == "planning":
            hint = "Напишите «составь план», чтобы перейти к утверждению"
        elif state.state == "plan_approval":
            hint = "Напишите «утверждаю» или «нужно поправить план»"
        elif state.state == "execution":
            hint = "Напишите «проверка» или «остановись»"
        elif state.state == "validation":
            hint = "Напишите «проверка прошла» или «нужно исправить»"
        return (
            f"Понял. Текущее состояние: {state.state.upper()} — "
            f"{state.expected_action}. {hint}."
        )

    if decision.allowed:
        if intent == "revise_plan":
            return (
                "Возвращаю задачу на этап планирования. "
                "Предыдущее утверждение плана сброшено. "
                f"Давайте скорректируем план. Ожидаемое действие: {state.expected_action}."
            )
        if intent == "pause":
            return (
                f"Ставлю задачу на паузу (этап {state.previous_state or state.state}). "
                f"Прогресс, шаг и контекст сохранены — напишите «продолжай», чтобы вернуться."
            )
        if intent == "resume":
            return (
                f"Продолжаю с этапа {state.state.upper()}. "
                f"Ожидаемое действие: {state.expected_action}."
            )
        return (
            f"Готово ({label}): переход {decision.from_state} → {decision.to_state} "
            f"выполнен. Теперь состояние: {state.state.upper()}; "
            f"ожидаемое действие: {state.expected_action}."
        )

    # Blocked: the intent was understood, but the lifecycle does not allow it.
    return (
        f"Я понял ваше намерение ({label}), но выполнить его сейчас нельзя. "
        f"Причина: {decision.reason} "
        f"Состояние остаётся {state.state.upper()}; ожидаемое действие: "
        f"{state.expected_action}."
    )


def _format_history(state: Day15TaskState) -> str:
    if not state.transitions:
        return "(no transitions)"
    lines = []
    for record in state.transitions[-8:]:
        marker = "ALLOWED" if record.status == "allowed" else "BLOCKED"
        suffix = f"  ({record.reason})" if record.reason else ""
        lines.append(
            f"{record.from_state} -> {record.to_state}  {marker}{suffix}"
        )
    return "\n".join(lines)


class Day15LifecycleService:
    """Owns the current lifecycle state and enforces the controlled transitions."""

    def __init__(
        self,
        settings: Settings,
        deepseek: DeepSeekService | None = None,
        store: LifecycleStore | None = None,
        machine: LifecycleMachine | None = None,
    ) -> None:
        self._settings = settings
        self._deepseek = deepseek if deepseek is not None else DeepSeekService(settings)
        self._store = store or LifecycleStore(settings.day15_task_path)
        self._machine = machine or LifecycleMachine()
        self._state: Day15TaskState | None = self._store.load()
        logger.info(
            "day15 lifecycle service initialised path=%s has_task=%s",
            self._store.path,
            self._state is not None,
        )

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------
    @property
    def model(self) -> str:
        return self._settings.deepseek_model

    @property
    def machine(self) -> LifecycleMachine:
        return self._machine

    def state(self) -> Day15TaskState | None:
        return self._state

    def history(self) -> list[TransitionRecord]:
        return list(self._state.transitions) if self._state else []

    def _require(self) -> Day15TaskState:
        if self._state is None:
            raise Day15Error(
                "Task has not been created yet. Create a task first.", status_code=404
            )
        return self._state

    def _save(self, state: Day15TaskState) -> Day15TaskState:
        state.updated_at = _now()
        self._state = state
        self._store.save(state)
        return state

    def _record(
        self,
        state: Day15TaskState,
        *,
        from_state: str,
        to_state: str,
        status: str,
        reason: str,
        action: str,
        intent: str = "",
        trigger: str = "api",
        user_message: str = "",
    ) -> TransitionRecord:
        record = TransitionRecord(
            seq=len(state.transitions) + 1,
            timestamp=_now(),
            from_state=from_state,
            to_state=to_state,
            status="allowed" if status == "allowed" else "blocked",
            reason=reason,
            action=action,
            intent=intent,
            trigger=trigger,
            user_message=user_message,
        )
        state.transitions.append(record)
        return record

    # ------------------------------------------------------------------
    # chat journal (the HUMAN part of the demo, separate from transitions)
    # ------------------------------------------------------------------
    def _emit(
        self,
        *,
        role: str,
        content: str,
        kind: str = "message",
        intent: str = "",
        confidence: float = 0.0,
        from_state: str = "",
        to_state: str = "",
        status: str = "",
        error: str = "",
    ) -> ChatMessage:
        """Append ONE entry to the task's chat journal and persist it.

        Every piece of agent work goes through here, which is what guarantees
        the assignment's rule: a state change never happens with nothing shown
        in the chat.
        """
        state = self._require()
        message = ChatMessage(
            seq=len(state.chat_messages) + 1,
            timestamp=_now(),
            role=role,  # type: ignore[arg-type]
            content=content,
            kind=kind,
            intent=intent,
            confidence=confidence,
            from_state=from_state,
            to_state=to_state,
            status=status,
            error=error,
        )
        state.chat_messages.append(message)
        # Bound the journal so it cannot grow without limit.
        if len(state.chat_messages) > 80:
            del state.chat_messages[:-80]
        self._save(state)
        return message

    def _emit_debug(
        self,
        *,
        intent: str,
        confidence: float,
        decision: TransitionDecision | None,
        error: str = "",
    ) -> None:
        """Append the structured intent/transition trace for the demo UI.

        This entry is a ``system``/``debug`` message so it is kept in the same
        journal (and survives reloads) while the UI can render it visually
        APART from the human conversation.
        """
        lines = ["DEBUG", f"intent: {intent}", f"confidence: {confidence:.2f}"]
        if error:
            lines.append(f"recognition: FAILED — {error}")
        if decision is not None:
            lines.append(
                f"transition: {decision.from_state.upper()} → "
                f"{decision.to_state.upper()}"
            )
            lines.append(
                "status: " + ("ALLOWED" if decision.allowed else "BLOCKED")
            )
            if decision.reason:
                lines.append(f"reason: {decision.reason}")
        else:
            lines.append("transition: NO TRANSITION")
            lines.append("status: NO-OP")
        self._emit(
            role="system",
            content="\n".join(lines),
            kind="debug",
            intent=intent,
            confidence=confidence,
            from_state=decision.from_state if decision else "",
            to_state=decision.to_state if decision else "",
            status=(
                "allowed"
                if decision and decision.allowed
                else ("blocked" if decision else "no-op")
            ),
            error=error,
        )

    def _decision_from_transition(
        self, response: Day15TransitionResponse
    ) -> TransitionDecision:
        """Rebuild the machine's decision from the recorded transition."""
        record = response.transition_history[-1]
        return TransitionDecision(
            allowed=record.status == "allowed",
            from_state=record.from_state,
            to_state=record.to_state,
            reason=record.reason,
            rule=record.action,
        )

    def _last_assistant_text(self, since: int) -> str:
        """Join the assistant content emitted after index ``since``."""
        state = self._require()
        parts = [
            message.content
            for message in state.chat_messages[since:]
            if message.role == "assistant" and message.content
        ]
        return "\n\n".join(parts)

    def _turn_slice(self, since: int) -> list[ChatMessage]:
        return list(self._require().chat_messages[since:])

    # ------------------------------------------------------------------
    # decision helpers (used by the natural-language path)
    # ------------------------------------------------------------------
    def _last_decision(self) -> TransitionDecision | None:
        """Build a decision from the most recent history record."""
        state = self._state
        if state is None or not state.transitions:
            return None
        record = state.transitions[-1]
        return TransitionDecision(
            allowed=record.status == "allowed",
            from_state=record.from_state,
            to_state=record.to_state,
            reason=record.reason,
            rule=record.action,
        )

    def _last_allowed(self) -> bool:
        state = self._state
        return bool(state and state.transitions and state.transitions[-1].status == "allowed")

    # ------------------------------------------------------------------
    # responses
    # ------------------------------------------------------------------
    def state_response(self) -> Day15StateResponse:
        state = self._state
        if state is None:
            return Day15StateResponse(model=self.model, has_task=False)
        return Day15StateResponse(
            model=self.model,
            has_task=True,
            state=state,
            allowed_transitions=self._machine.allowed_transitions(state.state),
            transition_options=self._machine.options(state),
            transition_history=list(state.transitions),
            plan_approved=state.plan_approved,
            validation_passed=state.validation_passed,
            paused=state.paused,
        )

    def _transition_response(
        self,
        state: Day15TaskState,
        decision: TransitionDecision,
        action: str,
    ) -> Day15TransitionResponse:
        return Day15TransitionResponse(
            allowed=decision.allowed,
            reason=decision.reason,
            action=action,
            state=state,
            allowed_transitions=self._machine.allowed_transitions(state.state),
            transition_options=self._machine.options(state),
            transition_history=list(state.transitions),
        )

    # ------------------------------------------------------------------
    # lifecycle creation
    # ------------------------------------------------------------------
    def create_task(self, goal: str) -> Day15TaskState:
        """Start a NEW task in ``planning`` (history reset)."""
        cleaned = (goal or "").strip()
        if not cleaned:
            raise Day15Error("Task goal must not be empty.")
        now = _now()
        state = Day15TaskState(
            task_id=_new_task_id(),
            goal=cleaned,
            state="planning",
            current_step=1,
            plan=[],
            completed_steps=[],
            plan_approved=False,
            validation_passed=None,
            paused=False,
            transitions=[],
            created_at=now,
            updated_at=now,
        )
        state.expected_action = expected_action_for(state)
        logger.info("day15 task created id=%s", state.task_id)
        return self._save(state)

    def reset(self) -> None:
        """Forget the current task (demo/tests convenience)."""
        self._state = None
        self._store.clear()

    # ------------------------------------------------------------------
    # the CENTRAL transition mechanism
    # ------------------------------------------------------------------
    def transition(
        self,
        target: TaskStateName,
        *,
        intent: str = "",
        trigger: str = "api",
        user_message: str = "",
    ) -> Day15TransitionResponse:
        """Request an explicit lifecycle transition.

        The decision is made by the pure machine. A blocked request is recorded
        in the history but the state is NOT changed.
        """
        state = self._require()
        decision = self._machine.evaluate(state, target)
        self._record(
            state,
            from_state=state.state,
            to_state=target,
            status="allowed" if decision.allowed else "blocked",
            reason=decision.reason,
            action="transition",
            intent=intent,
            trigger=trigger,
            user_message=user_message,
        )
        if decision.allowed:
            updated = self._machine.apply(state, target)
            self._save(updated)
            logger.info("day15 transition %s -> %s allowed", decision.from_state, target)
            return self._transition_response(updated, decision, "transition")
        # Blocked: persist ONLY the history change; state is untouched.
        self._save(state)
        logger.info(
            "day15 transition %s -> %s BLOCKED (%s)",
            decision.from_state,
            target,
            decision.reason,
        )
        return self._transition_response(state, decision, "transition")

    async def prepare_plan(
        self,
        message: str = "",
        model: str | None = None,
        *,
        intent: str = "",
        trigger: str = "api",
        user_message: str = "",
    ) -> Day15TransitionResponse:
        """Generate a plan and move ``planning -> plan_approval`` (guarded)."""
        state = self._require()
        if state.state != "planning":
            decision = TransitionDecision(
                allowed=False,
                from_state=state.state,
                to_state="plan_approval",
                reason="A plan can only be prepared in PLANNING state.",
                rule="not_allowed",
            )
            self._record(
                state,
                from_state=state.state,
                to_state="plan_approval",
                status="blocked",
                reason=decision.reason,
                action="prepare_plan",
                intent=intent,
                trigger=trigger,
                user_message=user_message,
            )
            self._save(state)
            return self._transition_response(state, decision, "prepare_plan")

        resolved_model = model or self.model
        note = ""
        try:
            messages, _ = build_planning_request(state.goal, message or "Prepare a plan")
            content, _, _ = await self._deepseek.generate(
                messages, model=resolved_model, max_tokens=900, thinking=False
            )
            plan = parse_plan(content)
            if not plan:
                raise ValueError("empty plan")
        except Exception as exc:  # noqa: BLE001 - demo must keep working offline
            logger.warning(
                "day15 plan generation fell back to deterministic plan (%s): %s",
                type(exc).__name__,
                exc,
            )
            plan = list(DEFAULT_PLAN)
            note = "Provider unavailable; used a deterministic fallback plan."

        state.plan = plan
        state.current_step = 1
        # The machine decides: planning -> plan_approval is guarded by a
        # non-empty plan, which now holds.
        decision = self._machine.evaluate(state, "plan_approval")
        self._record(
            state,
            from_state=state.state,
            to_state="plan_approval",
            status="allowed" if decision.allowed else "blocked",
            reason=note or decision.reason,
            action="prepare_plan",
            intent=intent,
            trigger=trigger,
            user_message=user_message,
        )
        if decision.allowed:
            state = self._machine.apply(state, "plan_approval")
        self._save(state)
        if decision.allowed:
            # The plan is REAL work, so it MUST appear in the chat before the
            # next state is entered. The transition already went through the
            # machine above; this only records the human-readable result.
            numbered = "\n".join(
                f"{index}. {step}" for index, step in enumerate(plan, 1)
            )
            self._emit(
                role="assistant",
                content=(
                    f"План:\n\n{numbered}\n\n"
                    "План готов. Если он подходит — напишите «утверждаю».\n"
                    "Если нужны изменения — напишите, что именно изменить."
                ),
                kind="plan",
            )
        response = self._transition_response(state, decision, "prepare_plan")
        if note:
            response.reason = note
        return response

    def approve_plan(
        self,
        *,
        intent: str = "",
        trigger: str = "api",
        user_message: str = "",
    ) -> Day15TransitionResponse:
        """Set ``plan_approved = True`` — the guard for plan_approval -> execution."""
        state = self._require()
        if state.state != "plan_approval":
            decision = TransitionDecision(
                allowed=False,
                from_state=state.state,
                to_state="plan_approval",
                reason="A plan can only be approved in PLAN_APPROVAL state.",
                rule="not_allowed",
            )
            self._record(
                state,
                from_state=state.state,
                to_state="plan_approval",
                status="blocked",
                reason=decision.reason,
                action="approve_plan",
                intent=intent,
                trigger=trigger,
                user_message=user_message,
            )
            self._save(state)
            return self._transition_response(state, decision, "approve_plan")

        state.plan_approved = True
        decision = TransitionDecision(
            allowed=True,
            from_state=state.state,
            to_state=state.state,
            reason="Plan approved. EXECUTION is now unlocked.",
            rule="guard",
        )
        self._record(
            state,
            from_state=state.state,
            to_state=state.state,
            status="allowed",
            reason="plan approved",
            action="approve_plan",
            intent=intent,
            trigger=trigger,
            user_message=user_message,
        )
        self._save(state)
        return self._transition_response(state, decision, "approve_plan")

    def set_validation(
        self,
        passed: bool,
        summary: str = "",
        *,
        intent: str = "",
        trigger: str = "api",
        user_message: str = "",
    ) -> Day15TransitionResponse:
        """Record the deterministic outcome of a validation run."""
        state = self._require()
        if state.state != "validation":
            decision = TransitionDecision(
                allowed=False,
                from_state=state.state,
                to_state="validation",
                reason="A validation result can only be recorded in VALIDATION state.",
                rule="not_allowed",
            )
            self._record(
                state,
                from_state=state.state,
                to_state="validation",
                status="blocked",
                reason=decision.reason,
                action="validation",
                intent=intent,
                trigger=trigger,
                user_message=user_message,
            )
            self._save(state)
            return self._transition_response(state, decision, "validation")

        state.validation_passed = passed
        state.validation_summary = summary
        reason = (
            "Validation passed. DONE is now unlocked."
            if passed
            else "Validation failed. DONE stays locked."
        )
        decision = TransitionDecision(
            allowed=True,
            from_state=state.state,
            to_state=state.state,
            reason=reason,
            rule="guard",
        )
        self._record(
            state,
            from_state=state.state,
            to_state=state.state,
            status="allowed",
            reason="validation passed" if passed else "validation failed",
            action="validation",
            intent=intent,
            trigger=trigger,
            user_message=user_message,
        )
        self._save(state)
        return self._transition_response(state, decision, "validation")

    # ------------------------------------------------------------------
    # pause / resume (validated, state-preserving)
    # ------------------------------------------------------------------
    def pause(
        self,
        *,
        intent: str = "",
        trigger: str = "api",
        user_message: str = "",
    ) -> Day15TransitionResponse:
        """Pause the task, remembering the ACTIVE state in ``previous_state``."""
        state = self._require()
        if state.paused:
            decision = TransitionDecision(
                allowed=False,
                from_state=state.state,
                to_state="paused",
                reason="Task is already paused.",
                rule="paused",
            )
            self._record(
                state,
                from_state=state.state,
                to_state="paused",
                status="blocked",
                reason=decision.reason,
                action="pause",
                intent=intent,
                trigger=trigger,
                user_message=user_message,
            )
            self._save(state)
            return self._transition_response(state, decision, "pause")
        if state.state == "done":
            decision = TransitionDecision(
                allowed=False,
                from_state=state.state,
                to_state="paused",
                reason="A completed task cannot be paused.",
                rule="not_allowed",
            )
            self._record(
                state,
                from_state=state.state,
                to_state="paused",
                status="blocked",
                reason=decision.reason,
                action="pause",
                intent=intent,
                trigger=trigger,
                user_message=user_message,
            )
            self._save(state)
            return self._transition_response(state, decision, "pause")

        active = state.state
        state.paused = True
        state.previous_state = active
        decision = TransitionDecision(
            allowed=True,
            from_state=active,
            to_state="paused",
            reason=f"Paused at {active}. State preserved for resume.",
            rule="allowed_transitions",
        )
        self._record(
            state,
            from_state=active,
            to_state="paused",
            status="allowed",
            reason=f"paused from {active}",
            action="pause",
            intent=intent,
            trigger=trigger,
            user_message=user_message,
        )
        self._save(state)
        return self._transition_response(state, decision, "pause")

    def resume(
        self,
        *,
        intent: str = "",
        trigger: str = "api",
        user_message: str = "",
    ) -> Day15TransitionResponse:
        """Resume to EXACTLY the state the task was paused at."""
        state = self._require()
        if not state.paused:
            decision = TransitionDecision(
                allowed=False,
                from_state=state.state,
                to_state=state.state,
                reason="Task is not paused.",
                rule="paused",
            )
            self._record(
                state,
                from_state=state.state,
                to_state=state.state,
                status="blocked",
                reason=decision.reason,
                action="resume",
                intent=intent,
                trigger=trigger,
                user_message=user_message,
            )
            self._save(state)
            return self._transition_response(state, decision, "resume")

        target = state.previous_state or state.state
        # Resume NEVER moves the task to a different state than it was paused at.
        state.state = target
        state.paused = False
        state.previous_state = None
        state.expected_action = expected_action_for(state)
        decision = TransitionDecision(
            allowed=True,
            from_state="paused",
            to_state=target,
            reason=f"Resumed into {target} with the saved context.",
            rule="allowed_transitions",
        )
        self._record(
            state,
            from_state="paused",
            to_state=target,
            status="allowed",
            reason=f"resumed to {target}",
            action="resume",
            intent=intent,
            trigger=trigger,
            user_message=user_message,
        )
        self._save(state)
        return self._transition_response(state, decision, "resume")

    # ------------------------------------------------------------------
    # assistant (proposes; the machine decides)
    # ------------------------------------------------------------------
    def chat(
        self,
        *,
        message: str,
        proposed_transition: TaskStateName | None = None,
        model: str | None = None,
    ) -> Day15ChatResponse:
        """Assistant turn.

        The assistant never writes the lifecycle directly. If it proposes a
        transition, the SAME state machine evaluates it and the attempt is
        logged. A blocked proposal leaves the state untouched.
        """
        state = self._require()
        resolved_model = model or self.model
        decision: TransitionDecision | None = None

        if proposed_transition is not None:
            decision = self._machine.evaluate(state, proposed_transition)
            self._record(
                state,
                from_state=state.state,
                to_state=proposed_transition,
                status="allowed" if decision.allowed else "blocked",
                reason=decision.reason,
                action="chat",
            )
            if decision.allowed:
                state = self._machine.apply(state, proposed_transition)
                answer = (
                    f"The state machine ALLOWED {decision.from_state} -> "
                    f"{proposed_transition}.\n\n"
                    f"Current state: {state.state}\n"
                    f"Expected action: {state.expected_action}"
                )
            else:
                answer = (
                    f"Transition {decision.from_state} -> {proposed_transition} "
                    f"is BLOCKED.\nReason: {decision.reason}\n\n"
                    f"Current state: {state.state}\n"
                    f"Expected action: {state.expected_action}\n"
                    f"I cannot move the task further until the guard is satisfied."
                )
            self._save(state)
        else:
            allowed = self._machine.allowed_transitions(state.state)
            answer = (
                f"Current state: {state.state}\n"
                f"Expected action: {state.expected_action}\n"
                f"Allowed transitions: {', '.join(allowed) or '(none)'}\n\n"
                f"Note: I can only propose actions. The backend state machine "
                f"decides whether a transition is applied."
            )

        return Day15ChatResponse(
            answer=answer,
            model=resolved_model,
            state=state,
            decision=decision,
            allowed_transitions=self._machine.allowed_transitions(state.state),
            transition_options=self._machine.options(state),
            transition_history=list(state.transitions),
        )

    # ------------------------------------------------------------------
    # natural-language assistant (the MAIN Day 15 interaction)
    # ------------------------------------------------------------------
    def _chat_response(
        self,
        state: Day15TaskState,
        *,
        answer: str,
        decision: TransitionDecision | None,
        intent: str,
        confidence: float,
        error: str,
        since: int,
        model: str,
        trigger: str,
        user_message: str,
    ) -> Day15ChatResponse:
        return Day15ChatResponse(
            answer=answer,
            model=model,
            state=state,
            decision=decision,
            detected_intent=intent,
            intent_confidence=confidence,
            intent_error=error,
            messages=self._turn_slice(since),
            user_message=user_message,
            trigger=trigger,
            allowed_transitions=self._machine.allowed_transitions(state.state),
            transition_options=self._machine.options(state),
            transition_history=list(state.transitions),
        )

    async def start_task(
        self, goal: str, model: str | None = None
    ) -> Day15ChatResponse:
        """Create a task AND immediately do the PLANNING work.

        This is the entry point the UI uses for "Create task". A task is by
        definition created in PLANNING, so the user must not have to ask for a
        plan separately: the agent generates one, shows it in the chat and the
        state machine moves ``planning -> plan_approval`` (guard: plan exists).
        """
        resolved_model = model or self.model
        state = self.create_task(goal)
        since = len(state.chat_messages)
        self._emit(role="user", content=goal, kind="message")
        self._emit(
            role="assistant",
            content="Задача создана. Составляю план…",
            kind="info",
        )
        response = await self.prepare_plan(
            message=goal,
            model=resolved_model,
            intent="continue_planning",
            trigger="agent",
            user_message=goal,
        )
        decision = self._decision_from_transition(response)
        if not decision.allowed:
            self._emit(
                role="assistant",
                content=f"Составить план не удалось: {decision.reason}",
                kind="blocked",
            )
        current = self._require()
        self._emit_debug(
            intent="continue_planning", confidence=1.0, decision=decision
        )
        answer = self._last_assistant_text(since)
        return self._chat_response(
            self._require(),
            answer=answer,
            decision=decision,
            intent="continue_planning",
            confidence=1.0,
            error="",
            since=since,
            model=resolved_model,
            trigger="agent",
            user_message=goal,
        )

    async def converse(
        self,
        *,
        message: str,
        forced_intent: str | None = None,
        model: str | None = None,
    ) -> Day15ChatResponse:
        """One natural-language turn of the conversational task agent.

        Pipeline (the LLM never touches ``state`` directly)::

            user message
              -> LLM intent recognition   (state-aware)
              -> application action
              -> LifecycleMachine guards
              -> state transition
              -> agent does the work of the (new) current state
              -> assistant content in the chat journal

        A recognition FAILURE is reported through ``intent_error`` instead of
        being silently downgraded to a confident ``normal_message (0.00)``.
        A forced intent (debug/no-LLM) applies only the deterministic action.
        """
        state = self._require()
        resolved_model = model or self.model
        since = len(state.chat_messages)
        self._emit(role="user", content=message, kind="message")

        trigger = "user_message"
        intent_error = ""
        if forced_intent and forced_intent in VALID_INTENTS:
            intent_result = IntentResult(
                intent=forced_intent, confidence=1.0, raw="forced"
            )  # type: ignore[arg-type]
            trigger = "forced_intent"
        elif forced_intent:
            intent_error = f"unknown forced intent: {forced_intent!r}"
            intent_result = IntentResult(
                intent="normal_message",
                confidence=0.0,
                raw="invalid-forced",
                error=intent_error,
            )
            trigger = "forced_intent"
        else:
            intent_result, intent_error = await recognize_intent(
                self._deepseek, self._require(), message, resolved_model
            )

        # Ambiguity guard: a weak/unknown intent is treated as normal_message.
        resolved_intent = (
            intent_result.intent if intent_result.is_confident else "normal_message"
        )

        # A forced intent is a deterministic debug control (no LLM): it applies
        # ONLY the transition, so the demo can prove the state machine in
        # isolation. Natural language drives the FULL agent (work + follow-up
        # workflow transitions).
        auto = trigger != "forced_intent"
        decision = await self._run_intent(
            resolved_intent,
            message=message,
            model=resolved_model,
            trigger=trigger,
            user_message=message,
            auto=auto,
        )

        if resolved_intent == "normal_message":
            if intent_error:
                self._answer_recognition_failure(intent_error)
            else:
                await self._answer_normal(message, model=resolved_model)

        current = self._require()
        answer = self._last_assistant_text(since)
        self._emit_debug(
            intent=resolved_intent,
            confidence=intent_result.confidence,
            decision=decision,
            error=intent_error,
        )
        logger.info(
            "day15 converse intent=%s confidence=%.2f error=%s result=%s state=%s",
            resolved_intent,
            intent_result.confidence,
            intent_error or "-",
            "allowed"
            if (decision and decision.allowed)
            else ("blocked" if decision else "no-op"),
            self._require().state,
        )
        return self._chat_response(
            self._require(),
            answer=answer,
            decision=decision,
            intent=resolved_intent,
            confidence=intent_result.confidence,
            error=intent_error,
            since=since,
            model=resolved_model,
            trigger=trigger,
            user_message=message,
        )

    # ------------------------------------------------------------------
    # WORK methods: the agent performs the stage, then content goes to chat
    # ------------------------------------------------------------------
    async def _run_planning_stage(
        self,
        *,
        message: str,
        model: str,
        intent: str,
        trigger: str,
        user_message: str,
        intro: str = "",
    ) -> TransitionDecision:
        """Generate a real plan and move PLANNING -> PLAN_APPROVAL."""
        if intro:
            self._emit(role="assistant", content=intro, kind="info")
        response = await self.prepare_plan(
            message=message,
            model=model,
            intent=intent,
            trigger=trigger,
            user_message=user_message,
        )
        decision = self._decision_from_transition(response)
        if not decision.allowed:
            self._emit(
                role="assistant",
                content=f"Составить план сейчас нельзя: {decision.reason}",
                kind="blocked",
            )
        return decision

    async def _run_execution_stage(
        self, *, message: str, model: str, trigger: str
    ) -> None:
        """Produce the concrete result of the current plan and show it in chat."""
        state = self._require()
        if state.state != "execution" or state.paused:
            return
        messages, _ = build_execution_request(state, message)
        try:
            result, _, _ = await self._deepseek.generate(
                messages, model=model, max_tokens=900, thinking=False
            )
            if not result.strip():
                raise ValueError("empty execution result")
        except Exception as exc:  # noqa: BLE001 - keep the demo moving
            logger.warning(
                "day15 execution work failed (%s): %s", type(exc).__name__, exc
            )
            result = (
                "Не удалось получить результат от провайдера "
                f"({type(exc).__name__}). Состояние остаётся EXECUTION."
            )
        state = self._require()
        state.execution_result = result
        if state.plan:
            state.completed_steps = list(state.plan)
            state.current_step = len(state.plan)
        state.expected_action = expected_action_for(state)
        self._save(state)
        self._emit(role="assistant", content=result, kind="execution")
        self._emit(
            role="assistant",
            content=(
                "Когда будете готовы, напишите «проверь результат» — "
                "я запущу этап VALIDATION."
            ),
            kind="info",
        )

    async def _run_validation_stage(
        self, *, message: str, model: str, trigger: str
    ) -> TransitionDecision | None:
        """Validate the produced result and record the verdict as a guard."""
        state = self._require()
        if state.state != "validation":
            return None
        messages, _ = build_validation_request(state, message)
        try:
            content, _, _ = await self._deepseek.generate(
                messages, model=model, max_tokens=300, thinking=False
            )
            verdict, summary = parse_verdict(content)
        except Exception as exc:  # noqa: BLE001 - conservative on failure
            logger.warning(
                "day15 validation work failed (%s): %s", type(exc).__name__, exc
            )
            verdict, summary = "fail", f"Не удалось выполнить проверку: {exc}"
        passed = verdict == "pass"
        self.set_validation(
            passed,
            summary or ("Проверка пройдена." if passed else "Проверка не пройдена."),
            intent="request_validation",
            trigger=trigger,
            user_message=message,
        )
        if passed:
            self._emit(
                role="assistant",
                content=f"Проверка пройдена. {summary}".strip(),
                kind="validation",
            )
            response = self.transition(
                "done",
                intent="request_validation",
                trigger="agent",
                user_message=message,
            )
            decision = self._decision_from_transition(response)
            if decision.allowed:
                self._emit(
                    role="assistant",
                    content="Задача завершена. Все этапы пройдены.",
                    kind="transition",
                )
            return decision
        self._emit(
            role="assistant",
            content=(
                f"Проверка не пройдена. {summary}\n"
                "Возвращаюсь к этапу EXECUTION, чтобы доработать результат."
            ).strip(),
            kind="validation",
        )
        response = self.transition(
            "execution",
            intent="request_validation",
            trigger="agent",
            user_message=message,
        )
        return self._decision_from_transition(response)

    async def _run_approve(
        self,
        *,
        message: str,
        model: str,
        intent: str,
        trigger: str,
        user_message: str,
        auto: bool,
    ) -> TransitionDecision:
        """Approve the plan, enter EXECUTION and immediately do the work."""
        approve_response = self.approve_plan(
            intent=intent, trigger=trigger, user_message=user_message
        )
        approve_decision = self._decision_from_transition(approve_response)
        if not approve_decision.allowed:
            self._emit(
                role="assistant",
                content=f"Утвердить план сейчас нельзя: {approve_decision.reason}",
                kind="blocked",
            )
            return approve_decision
        transition_response = self.transition(
            "execution", intent=intent, trigger=trigger, user_message=user_message
        )
        decision = self._decision_from_transition(transition_response)
        if not decision.allowed:
            self._emit(
                role="assistant",
                content=f"Перейти к выполнению сейчас нельзя: {decision.reason}",
                kind="blocked",
            )
            return decision
        self._emit(
            role="assistant",
            content="План утверждён. Перехожу к выполнению.",
            kind="transition",
        )
        if auto:
            await self._run_execution_stage(
                message=message, model=model, trigger="agent"
            )
        return decision

    async def _run_rollback(
        self,
        *,
        message: str,
        model: str,
        intent: str,
        trigger: str,
        user_message: str,
        auto: bool,
        reason: str,
    ) -> TransitionDecision:
        """CONTROLLED ROLLBACK to PLANNING and (optionally) re-plan immediately."""
        response = self.transition(
            "planning", intent=intent, trigger=trigger, user_message=user_message
        )
        decision = self._decision_from_transition(response)
        if not decision.allowed:
            self._emit(
                role="assistant",
                content=f"Вернуться к планированию сейчас нельзя: {decision.reason}",
                kind="blocked",
            )
            return decision
        self._emit(
            role="assistant",
            content=(
                f"{reason} Возвращаюсь к этапу PLANNING; предыдущее утверждение "
                "плана сброшено."
            ),
            kind="transition",
        )
        if auto:
            await self._run_planning_stage(
                message=message,
                model=model,
                intent=intent,
                trigger=trigger,
                user_message=user_message,
            )
        return decision

    async def _run_start_execution(
        self,
        *,
        message: str,
        model: str,
        intent: str,
        trigger: str,
        user_message: str,
        auto: bool,
    ) -> TransitionDecision | None:
        state = self._require()
        if state.state == "execution":
            if auto:
                await self._run_execution_stage(
                    message=message, model=model, trigger="agent"
                )
            return self._last_decision()
        if state.state == "plan_approval" and not state.plan_approved:
            self.approve_plan(
                intent=intent, trigger=trigger, user_message=user_message
            )
        response = self.transition(
            "execution", intent=intent, trigger=trigger, user_message=user_message
        )
        decision = self._decision_from_transition(response)
        if not decision.allowed:
            self._emit(
                role="assistant",
                content=f"Начать выполнение сейчас нельзя: {decision.reason}",
                kind="blocked",
            )
            return decision
        if auto:
            await self._run_execution_stage(
                message=message, model=model, trigger="agent"
            )
        return decision

    async def _run_request_validation(
        self,
        *,
        message: str,
        model: str,
        intent: str,
        trigger: str,
        user_message: str,
        auto: bool,
    ) -> TransitionDecision | None:
        state = self._require()
        if state.state == "validation":
            decision = self._last_decision()
        else:
            response = self.transition(
                "validation", intent=intent, trigger=trigger, user_message=user_message
            )
            decision = self._decision_from_transition(response)
            if not decision.allowed:
                self._emit(
                    role="assistant",
                    content=f"Запустить проверку сейчас нельзя: {decision.reason}",
                    kind="blocked",
                )
                return decision
        if auto:
            await self._run_validation_stage(
                message=message, model=model, trigger="agent"
            )
        return decision

    def _run_validation_flag(
        self, passed: bool, *, intent: str, trigger: str, user_message: str
    ) -> TransitionDecision:
        response = self.set_validation(
            passed,
            "Зафиксировано пользователем.",
            intent=intent,
            trigger=trigger,
            user_message=user_message,
        )
        decision = self._decision_from_transition(response)
        if not decision.allowed:
            self._emit(
                role="assistant",
                content=f"Зафиксировать проверку сейчас нельзя: {decision.reason}",
                kind="blocked",
            )
            return decision
        if passed:
            self._emit(
                role="assistant",
                content="Проверка отмечена как успешная.",
                kind="validation",
            )
            done = self.transition(
                "done", intent=intent, trigger="agent", user_message=user_message
            )
            decision = self._decision_from_transition(done)
            if decision.allowed:
                self._emit(
                    role="assistant",
                    content="Задача завершена.",
                    kind="transition",
                )
            else:
                self._emit(
                    role="assistant",
                    content=f"Завершить задачу нельзя: {decision.reason}",
                    kind="blocked",
                )
        else:
            self._emit(
                role="assistant",
                content=(
                    "Проверка отмечена как неуспешная. "
                    "Возвращаюсь к выполнению."
                ),
                kind="validation",
            )
            back = self.transition(
                "execution", intent=intent, trigger="agent", user_message=user_message
            )
            decision = self._decision_from_transition(back)
        return decision

    def _run_complete(
        self, *, intent: str, trigger: str, user_message: str
    ) -> TransitionDecision:
        """Request DONE — the machine blocks it until validation has passed."""
        response = self.transition(
            "done", intent=intent, trigger=trigger, user_message=user_message
        )
        decision = self._decision_from_transition(response)
        if decision.allowed:
            self._emit(
                role="assistant", content="Задача завершена.", kind="transition"
            )
        else:
            self._emit(
                role="assistant",
                content=(
                    f"Завершить задачу сейчас нельзя: {decision.reason}\n"
                    "Состояние остаётся "
                    f"{self._require().state.upper()}."
                ),
                kind="blocked",
            )
        return decision

    def _run_pause(
        self, *, intent: str, trigger: str, user_message: str
    ) -> TransitionDecision:
        response = self.pause(
            intent=intent, trigger=trigger, user_message=user_message
        )
        decision = self._decision_from_transition(response)
        if decision.allowed:
            self._emit(
                role="assistant",
                content=(
                    "Задача поставлена на паузу. Текущий этап, шаг и контекст "
                    "сохранены. Напишите «продолжай», чтобы вернуться."
                ),
                kind="transition",
            )
        else:
            self._emit(
                role="assistant",
                content=f"Поставить задачу на паузу нельзя: {decision.reason}",
                kind="blocked",
            )
        return decision

    async def _run_resume(
        self,
        *,
        message: str,
        model: str,
        intent: str,
        trigger: str,
        user_message: str,
        auto: bool,
    ) -> TransitionDecision:
        response = self.resume(
            intent=intent, trigger=trigger, user_message=user_message
        )
        decision = self._decision_from_transition(response)
        if not decision.allowed:
            self._emit(
                role="assistant",
                content=f"Продолжить сейчас нельзя: {decision.reason}",
                kind="blocked",
            )
            return decision
        state = self._require()
        self._emit(
            role="assistant",
            content=(
                f"Продолжаю с этапа {state.state.upper()}. "
                f"Ожидаемое действие: {state.expected_action}."
            ),
            kind="transition",
        )
        if auto:
            if state.state == "planning":
                await self._run_planning_stage(
                    message=message,
                    model=model,
                    intent=intent,
                    trigger=trigger,
                    user_message=user_message,
                )
            elif state.state == "execution" and not state.execution_result:
                await self._run_execution_stage(
                    message=message, model=model, trigger="agent"
                )
        return decision

    async def _run_intent(
        self,
        intent: str,
        *,
        message: str,
        model: str,
        trigger: str = "user_message",
        user_message: str = "",
        auto: bool = True,
    ) -> TransitionDecision | None:
        """Map a semantic INTENT to an application ACTION.

        The application only ever REQUESTS transitions/flag changes; the
        ``LifecycleMachine`` still decides whether they are allowed. When
        ``auto`` is set (the natural-language path), the agent also performs
        the WORK of the resulting state and emits its result into the chat.
        """
        if intent == "normal_message":
            return None

        meta = {"intent": intent, "trigger": trigger, "user_message": user_message}

        if intent == "continue_planning":
            return await self._run_planning_stage(
                message=message, model=model, **meta
            )
        if intent == "approve_plan":
            return await self._run_approve(
                message=message, model=model, auto=auto, **meta
            )
        if intent == "request_plan_changes":
            return await self._run_rollback(
                message=message,
                model=model,
                auto=auto,
                reason="Принято: план нужно поправить.",
                **meta,
            )
        if intent == "revise_plan":
            return await self._run_rollback(
                message=message,
                model=model,
                auto=auto,
                reason="Принято: возвращаемся к пересмотру плана.",
                **meta,
            )
        if intent == "start_execution":
            return await self._run_start_execution(
                message=message, model=model, auto=auto, **meta
            )
        if intent == "request_validation":
            return await self._run_request_validation(
                message=message, model=model, auto=auto, **meta
            )
        if intent == "validation_passed":
            return self._run_validation_flag(True, **meta)
        if intent == "validation_failed":
            return self._run_validation_flag(False, **meta)
        if intent == "complete_task":
            return self._run_complete(**meta)
        if intent == "pause":
            return self._run_pause(**meta)
        if intent == "resume":
            return await self._run_resume(
                message=message, model=model, auto=auto, **meta
            )
        return None

    # ------------------------------------------------------------------
    # conversational answers (normal_message)
    # ------------------------------------------------------------------
    async def _answer_normal(self, message: str, *, model: str) -> str:
        """Answer a normal message in the context of the task (state unchanged)."""
        state = self._require()
        messages, _ = build_answer_request(state, message)
        try:
            content, _, _ = await self._deepseek.generate(
                messages, model=model, max_tokens=700, thinking=False
            )
            if not content.strip():
                raise ValueError("empty answer")
        except Exception as exc:  # noqa: BLE001 - never break the turn
            logger.warning(
                "day15 conversational answer failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            content = (
                "Не удалось получить ответ от провайдера "
                f"({type(exc).__name__}). Состояние не изменилось: "
                f"{state.state.upper()} — {state.expected_action}."
            )
        self._emit(role="assistant", content=content, kind="message")
        return content

    def _answer_recognition_failure(self, error: str) -> str:
        """Report a recognition FAILURE honestly; the state stays untouched."""
        content = (
            "Не удалось распознать намерение.\n"
            f"Причина: {error}\n"
            "Состояние не изменено. Проверьте подключение/модель и повторите "
            "сообщение."
        )
        self._emit(
            role="assistant", content=content, kind="error", error=error
        )
        return content

    # ------------------------------------------------------------------
    # demo scenarios
    # ------------------------------------------------------------------
    def setup_scenario(self, name: str) -> Day15TaskState:
        """Build a deterministic starting point for a demo scenario.

        The scenarios seed the state AND the transition history so the video
        can immediately show the interesting part (blocked moves, pause/resume)
        without clicking through the whole lifecycle first.
        """
        now = _now()
        plan = list(DEFAULT_PLAN)

        def base(state_name: str) -> Day15TaskState:
            task = Day15TaskState(
                task_id=_new_task_id(),
                goal="Build a REST API for a small notes service",
                state=state_name,  # type: ignore[arg-type]
                current_step=1,
                plan=list(plan),
                plan_approved=False,
                validation_passed=None,
                paused=False,
                transitions=[],
                created_at=now,
                updated_at=now,
            )
            task.expected_action = expected_action_for(task)
            return task

        if name in {"happy_path", "skip_planning"}:
            task = base("planning")
            # A plan is already prepared so PLAN_APPROVAL is reachable, but the
            # state deliberately stays in PLANNING for the skip demonstration.
            self._save(task)
            return task

        if name == "skip_validation":
            task = base("execution")
            task.plan_approved = True
            task.current_step = 2
            task.completed_steps = [plan[0]]
            task.expected_action = plan[1]
            self._record(
                task,
                from_state="planning",
                to_state="plan_approval",
                status="allowed",
                reason="plan prepared",
                action="prepare_plan",
            )
            self._record(
                task,
                from_state="plan_approval",
                to_state="execution",
                status="allowed",
                reason="plan approved",
                action="transition",
            )
            self._save(task)
            return task

        if name == "failed_validation":
            task = base("validation")
            task.plan_approved = True
            task.current_step = len(plan)
            task.completed_steps = list(plan)
            task.validation_passed = False
            task.validation_summary = "Missing error handling"
            task.expected_action = expected_action_for(task)
            for record in (
                ("planning", "plan_approval", "prepare_plan"),
                ("plan_approval", "execution", "transition"),
                ("execution", "validation", "transition"),
            ):
                self._record(
                    task,
                    from_state=record[0],
                    to_state=record[1],
                    status="allowed",
                    reason="scenario setup",
                    action=record[2],
                )
            self._record(
                task,
                from_state="validation",
                to_state="done",
                status="blocked",
                reason="Validation failed. Fix the result and validate again.",
                action="transition",
            )
            self._save(task)
            return task

        if name == "pause_resume":
            task = base("execution")
            task.plan_approved = True
            task.current_step = 2
            task.completed_steps = [plan[0]]
            task.expected_action = plan[1]
            self._record(
                task,
                from_state="planning",
                to_state="plan_approval",
                status="allowed",
                reason="plan prepared",
                action="prepare_plan",
            )
            self._record(
                task,
                from_state="plan_approval",
                to_state="execution",
                status="allowed",
                reason="plan approved",
                action="transition",
            )
            self._save(task)
            return task

        raise Day15Error(f"Unknown scenario: {name}", status_code=404)


__all__ = ["DEFAULT_PLAN", "Day15Error", "Day15LifecycleService", "STATES"]
