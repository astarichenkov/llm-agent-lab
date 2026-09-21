"""Day 13 — the Task State Machine (FSM).

The FSM is intentionally tiny and PURE: it knows the stages, the ALLOWED
transitions and the expected action for each stage. It never calls an LLM and
never touches storage, so it is trivially unit-testable.

The LLM is NOT allowed to pick a stage: a caller must go through
:meth:`TaskStateMachine.transition`, which rejects any transition that is not
in :data:`ALLOWED_TRANSITIONS`.

Diagram::

    planning ──► execution ──► validation ──► done
                     ▲              │
                     └──────────────┘   (validation -> execution)

``validation -> execution`` is a real necessity: when the check fails the task
must be able to go back and fix the result instead of being stuck.
"""
from __future__ import annotations

from app.schemas.day13 import TaskStage, TaskState

STAGES: tuple[TaskStage, ...] = ("planning", "execution", "validation", "done")

# The ONLY transitions the code allows. Anything else is rejected.
ALLOWED_TRANSITIONS: dict[TaskStage, set[TaskStage]] = {
    "planning": {"execution"},
    "execution": {"validation"},
    "validation": {"execution", "done"},
    "done": set(),
}

# Expected action per stage when there is no more specific step.
_STAGE_EXPECTED_ACTION: dict[TaskStage, str] = {
    "planning": "Составить план реализации",
    "execution": "Выполнить текущий шаг плана",
    "validation": "Проверить полученный результат",
    "done": "Задача завершена",
}


class InvalidTransitionError(ValueError):
    """Raised when a transition is not allowed by the FSM."""

    def __init__(self, from_stage: str, to_stage: str) -> None:
        allowed = sorted(ALLOWED_TRANSITIONS.get(from_stage, set()))
        super().__init__(
            "Недопустимый переход "
            f"{from_stage!r} -> {to_stage!r}. "
            f"Разрешено из {from_stage!r}: "
            + (", ".join(allowed) if allowed else "(нет переходов)")
        )
        self.from_stage = from_stage
        self.to_stage = to_stage


def expected_action_for(state: TaskState) -> str:
    """Return the expected action for the task's CURRENT stage/step.

    During ``execution`` the action is the current plan step (falling back to a
    generic phrase when the plan is empty); for every other stage it is fixed.
    """
    if state.stage == "execution" and state.plan:
        index = max(1, min(state.current_step, len(state.plan))) - 1
        return state.plan[index]
    return _STAGE_EXPECTED_ACTION[state.stage]


class TaskStateMachine:
    """Pure, stateless transition guard for :class:`~app.schemas.day13.TaskState`."""

    def allowed_transitions(self, stage: TaskStage) -> list[TaskStage]:
        """Return the stages reachable from ``stage`` (stable order)."""
        allowed = ALLOWED_TRANSITIONS.get(stage, set())
        return [candidate for candidate in STAGES if candidate in allowed]

    def can_transition(self, from_stage: TaskStage, to_stage: TaskStage) -> bool:
        """True only when the FSM explicitly allows ``from_stage -> to_stage``."""
        return to_stage in ALLOWED_TRANSITIONS.get(from_stage, set())

    def transition(self, state: TaskState, to_stage: TaskStage) -> TaskState:
        """Return a COPY of ``state`` moved to ``to_stage``.

        Raises :class:`InvalidTransitionError` when the transition is not
        allowed. ``expected_action`` is refreshed by the caller/service when a
        plan step is involved.
        """
        if not self.can_transition(state.stage, to_stage):
            raise InvalidTransitionError(state.stage, to_stage)
        updated = state.model_copy(deep=True)
        updated.stage = to_stage
        updated.expected_action = expected_action_for(updated)
        return updated
