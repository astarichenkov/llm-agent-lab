"""Day 13 — Task State Machine service.

This service is the thin orchestration layer that ties together:

* the **TaskState** (structured data: stage / current_step / expected_action);
* the **TaskStateMachine** (code-enforced allowed transitions);
* the **TaskStore**    (persistence across HTTP requests / restarts);
* the existing **DeepSeekService** (the LLM).

The division of responsibility is strict:

    LLM  -> generates content (a plan, an answer, a verification verdict)
    code -> decides and applies every STAGE transition (planning->execution,
            execution->validation, validation->done, validation->execution)

The LLM can therefore never set an arbitrary stage: if it returns something
unexpected, the service simply does not transition (or falls back safely).

Pause is a flag, not a stage: ``paused=true`` preserves ``stage``,
``current_step`` and ``expected_action`` so a Resume continues exactly where
the task stopped, without re-explaining the original task.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.config import Settings
from app.schemas.day13 import (
    Day13ChatResponse,
    Day13StateResponse,
    TaskStage,
    TaskState,
)
from app.services.day13.store import TaskStore
from app.services.day13.task_state import (
    InvalidTransitionError,
    TaskStateMachine,
    expected_action_for,
)
from app.services.deepseek import DeepSeekService
from app.services.structured_json import parse_json_object

logger = logging.getLogger("app.services.day13")

# ----------------------------------------------------------------------
# Prompts (the ONLY place task-state -> prompt text happens)
# ----------------------------------------------------------------------
PLANNING_SYSTEM_PROMPT = (
    "You are a task PLANNER. The user gives one TASK GOAL. Produce a short, "
    "concrete plan of 3 to 5 actionable steps and return STRICT JSON:\n"
    '{"plan": ["step 1", "step 2", "step 3"]}\n\n'
    "Rules:\n"
    "- each step is ONE short imperative phrase;\n"
    "- do not number the steps inside the strings;\n"
    "- do not add markdown, prose or code fences;\n"
    "- output ONLY the JSON object."
)

EXECUTION_SYSTEM_PROMPT = (
    "You are an autonomous assistant WORKING on a task. You receive the "
    "formal TASK STATE (stage, current step, expected action, plan, completed "
    "steps). Perform ONLY the expected action for the current step. Do not "
    "redo completed steps and do not skip ahead unless asked. Answer in the "
    "same language the task is written in. Be concise and concrete."
)

VALIDATION_SYSTEM_PROMPT = (
    "You are a strict TASK VALIDATOR. Check whether the task result matches "
    "the goal and every plan step. Reply with STRICT JSON:\n"
    '{"verdict": "pass" | "fail", "summary": "one short sentence"}\n\n'
    "Use \"pass\" only when the result really satisfies the goal and all "
    "steps. Output ONLY the JSON object."
)

# Safety cap so a runaway model cannot create an enormous plan.
_MAX_PLAN_STEPS = 8

# Minimal natural-language pause/resume commands. Day 13 keeps this
# deliberately tiny: the button endpoints remain the primary control and these
# commands only map to the EXISTING pause/resume service methods.
_PAUSE_COMMANDS = {"пауза", "pause"}
_RESUME_COMMANDS = {"продолжай", "resume", "continue"}

# Fallback plan used when the planning model returns unparsable content.
_DEFAULT_PLAN = [
    "Проанализировать задачу и требования",
    "Выполнить основной объём работы",
    "Проверить полученный результат",
]


class TaskStateError(Exception):
    """Raised for invalid operations on the current task (no task / paused)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_task_id() -> str:
    return f"task-{uuid4().hex[:8]}"


# ----------------------------------------------------------------------
# Parsing / formatting (pure, unit-testable)
# ----------------------------------------------------------------------
def parse_plan(raw: str) -> list[str]:
    """Parse a model plan into a list of steps.

    Primary path: strict JSON ``{"plan": [...]}``. Fallback: a numbered or
    bulleted list. Final fallback: a small deterministic plan.
    """
    text = (raw or "").strip()
    if text:
        try:
            data = parse_json_object(text)
            steps = data.get("plan", data.get("steps"))
            if isinstance(steps, list):
                cleaned = [str(item).strip() for item in steps if str(item).strip()]
                if cleaned:
                    return cleaned[:_MAX_PLAN_STEPS]
        except ValueError:
            pass

        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for prefix in ("- ", "* ", "• "):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            else:
                # Strip a leading "1." / "1)" numbering.
                head = line.split(" ", 1)
                if head[0].rstrip(".)").isdigit() and len(head) == 2:
                    line = head[1].strip()
            if line:
                lines.append(line)
        if 2 <= len(lines) <= _MAX_PLAN_STEPS:
            return lines

    return list(_DEFAULT_PLAN)


def parse_verdict(raw: str) -> tuple[str, str]:
    """Return ``("pass"|"fail", summary)`` from a validator response."""
    text = (raw or "").strip()
    if text:
        try:
            data = parse_json_object(text)
            verdict = str(data.get("verdict", "")).strip().lower()
            summary = str(data.get("summary", "")).strip()
            if verdict in {"pass", "passed", "ok", "success", "valid"}:
                return "pass", summary
            if verdict in {"fail", "failed", "invalid", "error"}:
                return "fail", summary
        except ValueError:
            pass
        lowered = text.lower()
        if "pass" in lowered or "успешн" in lowered:
            return "pass", ""
    # Conservative default: do NOT auto-finish a task we could not verify.
    return "fail", ""


def format_plan(plan: list[str]) -> str:
    """Render a plan as a readable numbered list (used as the chat answer)."""
    if not plan:
        return "План пуст."
    lines = "\n".join(f"{index}. {step}" for index, step in enumerate(plan, 1))
    return f"План ({len(plan)} шага):\n{lines}"


def is_pause_command(message: str) -> bool:
    """True for a bare pause command (no NLP: exact short command match)."""
    return (message or "").strip().lower() in _PAUSE_COMMANDS


def is_resume_command(message: str) -> bool:
    """True for a bare resume command (no NLP: exact short command match)."""
    return (message or "").strip().lower() in _RESUME_COMMANDS


def build_task_state_block(state: TaskState) -> str:
    """Build the formal TASK STATE block injected into a model request."""
    total = len(state.plan) if state.plan else 0
    step_line = f"{state.current_step} из {total}" if total else str(state.current_step)
    lines = [
        "TASK STATE",
        "==========",
        "TASK:",
        state.goal or "(не указана)",
        "",
        "STAGE:",
        state.stage,
        "",
        "CURRENT STEP:",
        step_line,
        "",
        "EXPECTED ACTION:",
        state.expected_action or "(не указано)",
    ]
    if state.plan:
        lines += [
            "",
            "PLAN:",
            "\n".join(f"{i}. {step}" for i, step in enumerate(state.plan, 1)),
        ]
    if state.completed_steps:
        lines += [
            "",
            "COMPLETED STEPS:",
            "\n".join(f"- {step}" for step in state.completed_steps),
        ]
    lines += ["", "PAUSED:", "true" if state.paused else "false"]
    return "\n".join(lines)


def build_planning_request(goal: str, message: str) -> tuple[list[dict[str, str]], str]:
    """Messages for turning a goal into a plan."""
    user_content = (
        f"TASK GOAL:\n{goal}\n\n"
        f"USER MESSAGE:\n{message}\n\n"
        "Return the JSON plan now."
    )
    messages = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, user_content


def build_execution_request(
    state: TaskState, message: str
) -> tuple[list[dict[str, str]], str]:
    """Messages for executing the current plan step."""
    block = build_task_state_block(state)
    messages = [
        {"role": "system", "content": EXECUTION_SYSTEM_PROMPT + "\n\n" + block},
        {"role": "user", "content": message},
    ]
    return messages, block


def build_validation_request(
    state: TaskState, message: str
) -> tuple[list[dict[str, str]], str]:
    """Messages for validating the task result."""
    block = build_task_state_block(state)
    messages = [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT + "\n\n" + block},
        {"role": "user", "content": message},
    ]
    return messages, block


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------
class Day13TaskService:
    """Owns the current Task State and drives the FSM around the LLM."""

    def __init__(
        self,
        settings: Settings,
        deepseek: DeepSeekService | None = None,
        store: TaskStore | None = None,
        machine: TaskStateMachine | None = None,
    ) -> None:
        self._settings = settings
        self._deepseek = deepseek if deepseek is not None else DeepSeekService(settings)
        self._store = store or TaskStore(settings.day13_task_path)
        self._machine = machine or TaskStateMachine()
        # The task survives requests / restarts by being loaded from disk.
        self._state: TaskState | None = self._store.load()
        logger.info(
            "day13 task service initialised path=%s has_task=%s",
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
    def machine(self) -> TaskStateMachine:
        return self._machine

    def state(self) -> TaskState | None:
        return self._state

    def allowed_transitions(self) -> list[TaskStage]:
        if self._state is None:
            return []
        return self._machine.allowed_transitions(self._state.stage)

    def state_response(self) -> Day13StateResponse:
        return Day13StateResponse(
            model=self.model,
            has_task=self._state is not None,
            state=self._state,
            allowed_transitions=[str(s) for s in self.allowed_transitions()],
        )

    def _require(self) -> TaskState:
        if self._state is None:
            raise TaskStateError(
                "Задача ещё не создана. Сначала создайте задачу.", status_code=404
            )
        return self._state

    def _save(self, state: TaskState) -> TaskState:
        state.updated_at = _now()
        self._state = state
        self._store.save(state)
        return state

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def create_task(self, goal: str) -> TaskState:
        """Start a NEW task in the ``planning`` stage."""
        cleaned = (goal or "").strip()
        if not cleaned:
            raise TaskStateError("Цель задачи не может быть пустой.")
        now = _now()
        state = TaskState(
            task_id=_new_task_id(),
            goal=cleaned,
            stage="planning",
            current_step=1,
            plan=[],
            completed_steps=[],
            paused=False,
            created_at=now,
            updated_at=now,
        )
        state.expected_action = expected_action_for(state)
        logger.info("day13 task created id=%s", state.task_id)
        return self._save(state)

    def pause(self) -> TaskState:
        """Pause the task on ANY unfinished stage (state is preserved)."""
        state = self._require()
        if state.stage == "done":
            raise TaskStateError("Завершённую задачу нельзя поставить на паузу.")
        state.paused = True
        logger.info(
            "day13 task paused id=%s stage=%s step=%s",
            state.task_id,
            state.stage,
            state.current_step,
        )
        return self._save(state)

    def resume(self) -> TaskState:
        """Resume a paused task WITHOUT changing stage/step/expected_action."""
        state = self._require()
        state.paused = False
        logger.info(
            "day13 task resumed id=%s stage=%s step=%s expected=%r",
            state.task_id,
            state.stage,
            state.current_step,
            state.expected_action,
        )
        return self._save(state)

    def transition(self, to_stage: TaskStage) -> TaskState:
        """Apply an EXPLICIT, code-validated transition."""
        state = self._require()
        if state.paused:
            raise TaskStateError(
                "Задача на паузе. Сначала выполните Resume.", status_code=409
            )
        try:
            updated = self._machine.transition(state, to_stage)
        except InvalidTransitionError as exc:
            raise TaskStateError(str(exc)) from exc
        # When entering execution make sure the expected action points at the
        # current plan step (or a sensible generic phrase when the plan is empty).
        if to_stage == "execution":
            if updated.plan:
                index = max(1, min(updated.current_step, len(updated.plan))) - 1
                updated.expected_action = updated.plan[index]
            else:
                updated.expected_action = "Выполнить текущий шаг"
        logger.info(
            "day13 transition id=%s %s -> %s",
            updated.task_id,
            state.stage,
            updated.stage,
        )
        return self._save(updated)

    def reset(self) -> None:
        """Forget the current task (demo/tests convenience)."""
        self._state = None
        self._store.clear()

    # ------------------------------------------------------------------
    # chat / work
    # ------------------------------------------------------------------
    async def chat(
        self,
        *,
        message: str,
        model: str | None = None,
    ) -> Day13ChatResponse:
        """One turn on the current task, driven by the formal Task State.

        A bare ``пауза``/``pause`` command maps to :meth:`pause`; a bare
        ``продолжай``/``resume``/``continue`` command maps to :meth:`resume`
        ONLY when the task is currently paused (otherwise it is a normal work
        message). This keeps the text path equivalent to the Pause/Resume
        buttons without adding an NLP layer.
        """
        state = self._require()
        resolved_model = model or self.model

        if is_pause_command(message) and state.stage != "done":
            return self._pause_turn(resolved_model)
        if state.paused and is_resume_command(message):
            return self._resume_turn(resolved_model)
        if state.paused:
            raise TaskStateError(
                "Задача на паузе. Напишите «продолжай» или нажмите Resume.",
                status_code=409,
            )

        if state.stage == "planning":
            return await self._planning_turn(state, message, resolved_model)
        if state.stage == "execution":
            return await self._execution_turn(state, message, resolved_model)
        if state.stage == "validation":
            return await self._validation_turn(state, message, resolved_model)

        # done — no transitions left; answer without changing the state.
        return Day13ChatResponse(
            answer="Задача уже завершена. Создайте новую задачу, чтобы продолжить.",
            model=resolved_model,
            state=state,
            allowed_transitions=[],
            context_block=build_task_state_block(state),
        )

    def _pause_turn(self, model: str) -> Day13ChatResponse:
        """Pause via the chat text command (no LLM call)."""
        state = self.pause()
        return Day13ChatResponse(
            answer="Пауза принята. Останавливаюсь на текущем шаге.",
            model=model,
            state=state,
            allowed_transitions=[str(s) for s in self.allowed_transitions()],
            context_block=build_task_state_block(state),
            stage_event="pause",
        )

    def _resume_turn(self, model: str) -> Day13ChatResponse:
        """Resume via the chat text command (no LLM call, no re-explanation)."""
        state = self.resume()
        return Day13ChatResponse(
            answer="Продолжаю с сохранённого шага: "
            f"{state.expected_action or '—'}.",
            model=model,
            state=state,
            allowed_transitions=[str(s) for s in self.allowed_transitions()],
            context_block=build_task_state_block(state),
            stage_event="resume",
        )

    async def _planning_turn(
        self, state: TaskState, message: str, model: str
    ) -> Day13ChatResponse:
        messages, block = build_planning_request(state.goal, message)
        content, finish, usage = await self._deepseek.generate(
            messages, model=model, max_tokens=900, thinking=False
        )
        plan = parse_plan(content)
        state.plan = plan
        state.current_step = 1
        # The LLM returned content; CODE performs the only allowed transition.
        updated = self._machine.transition(state, "execution")
        updated.expected_action = plan[0]
        self._save(updated)
        logger.info(
            "day13 planning done id=%s steps=%s -> execution",
            updated.task_id,
            len(plan),
        )
        return Day13ChatResponse(
            answer=format_plan(plan),
            model=model,
            finish_reason=finish,
            state=updated,
            allowed_transitions=[str(s) for s in self.allowed_transitions()],
            context_block=block,
            stage_event="planning → execution",
            usage=usage,
        )

    async def _execution_turn(
        self, state: TaskState, message: str, model: str
    ) -> Day13ChatResponse:
        messages, block = build_execution_request(state, message)
        content, finish, usage = await self._deepseek.generate(
            messages, model=model, max_tokens=2048, thinking=False
        )
        # Code marks the current step as completed and advances the state.
        step_text = (
            state.plan[state.current_step - 1]
            if 0 <= state.current_step - 1 < len(state.plan)
            else state.expected_action
        )
        if step_text and step_text not in state.completed_steps:
            state.completed_steps.append(step_text)

        event: str | None = None
        if state.current_step < len(state.plan):
            state.current_step += 1
            state.expected_action = state.plan[state.current_step - 1]
        else:
            # All planned steps are done: CODE moves to validation.
            state = self._machine.transition(state, "validation")
            event = "execution → validation"
        self._save(state)
        logger.info(
            "day13 execution turn id=%s step=%s completed=%s stage=%s",
            state.task_id,
            state.current_step,
            len(state.completed_steps),
            state.stage,
        )
        return Day13ChatResponse(
            answer=content,
            model=model,
            finish_reason=finish,
            state=state,
            allowed_transitions=[str(s) for s in self.allowed_transitions()],
            context_block=block,
            stage_event=event,
            usage=usage,
        )

    async def _validation_turn(
        self, state: TaskState, message: str, model: str
    ) -> Day13ChatResponse:
        messages, block = build_validation_request(state, message)
        content, finish, usage = await self._deepseek.generate(
            messages, model=model, max_tokens=900, thinking=False
        )
        verdict, summary = parse_verdict(content)
        event: str | None
        if verdict == "pass":
            state = self._machine.transition(state, "done")
            event = "validation → done"
        else:
            state = self._machine.transition(state, "execution")
            # Re-run the last plan step to address the validation feedback.
            if state.plan:
                state.current_step = len(state.plan)
                state.expected_action = state.plan[-1]
            event = "validation → execution"
        self._save(state)
        answer = content
        if summary:
            answer = f"{content}\n\nИтог проверки: {summary}"
        logger.info(
            "day13 validation verdict=%s id=%s -> %s",
            verdict,
            state.task_id,
            state.stage,
        )
        return Day13ChatResponse(
            answer=answer,
            model=model,
            finish_reason=finish,
            state=state,
            allowed_transitions=[str(s) for s in self.allowed_transitions()],
            context_block=block,
            stage_event=event,
            usage=usage,
        )
