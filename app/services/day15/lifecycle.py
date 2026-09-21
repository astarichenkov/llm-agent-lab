"""Day 15 — the controlled lifecycle (state machine + guards).

This module is the SINGLE source of truth for the task lifecycle. It is pure:
it never calls an LLM and never touches storage, so it is trivially testable.

Lifecycle::

    planning ──► plan_approval ──► execution ──► validation ──► done
                     │                  ▲              │
                     └──► planning      └──────────────┘
                        (request changes)  validation -> execution
                                           (validation failed)

``pause`` is NOT a state: it is a flag that preserves the current state and
remembers it in ``previous_state``. ``resume`` restores exactly that state, so
pause can never be abused to jump to another state.

Two rules protect the critical transitions (GUARDS):

* ``plan_approval -> execution`` requires ``plan_approved == True``;
* ``validation   -> done``      requires ``validation_passed == True``.

The LLM may *propose* an action, but only :meth:`LifecycleMachine.evaluate` /
:meth:`LifecycleMachine.apply` decide whether it is applied. A caller that does
not go through this engine cannot legitimately change ``state``.
"""
from __future__ import annotations

from typing import Callable

from app.schemas.day15 import (
    Day15TaskState,
    TaskStateName,
    TransitionDecision,
    TransitionOption,
)

# Stable, ordered list of states (used by the UI and the engine).
STATES: tuple[TaskStateName, ...] = (
    "planning",
    "plan_approval",
    "execution",
    "validation",
    "done",
)

# The ONLY transitions the code allows. Anything else is rejected.
#
# ``execution -> planning`` is a CONTROLLED ROLLBACK: during implementation the
# user may decide the PLAN itself must change. It is a normal row in the table
# (not an escape hatch) and it resets the stale ``plan_approved`` flag, so the
# revised plan must be approved again before execution.
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "planning": {"plan_approval"},
    "plan_approval": {"execution", "planning"},
    "execution": {"validation", "planning"},
    "validation": {"done", "execution"},
    "done": set(),
}

# Expected action per state (English, matching the assignment's UI wording).
EXPECTED_ACTION: dict[str, str] = {
    "planning": "Generate / prepare a plan",
    "plan_approval": "Approve the plan or request changes",
    "execution": "Execute the current plan step",
    "validation": "Validate the result",
    "done": "Task completed",
}

# Human-readable name of the guard attached to a critical transition.
GUARD_LABELS: dict[tuple[str, str], str] = {
    ("plan_approval", "execution"): "plan_approved",
    ("validation", "done"): "validation_passed",
    ("planning", "plan_approval"): "plan_not_empty",
}

# Transitions that semantically ROLL BACK the lifecycle (shown as ROLLBACK in
# the UI/log). Internally they are ordinary controlled transitions.
ROLLBACK_TRANSITIONS: set[tuple[str, str]] = {
    ("execution", "planning"),
}


def is_rollback(from_state: str, to_state: str) -> bool:
    """True when ``from_state -> to_state`` is a lifecycle rollback."""
    return (from_state, to_state) in ROLLBACK_TRANSITIONS

Guard = Callable[[Day15TaskState], tuple[bool, str]]


# ----------------------------------------------------------------------
# Guards — deterministic, code-side conditions for critical transitions
# ----------------------------------------------------------------------
def guard_plan_not_empty(state: Day15TaskState) -> tuple[bool, str]:
    if not state.plan:
        return False, "Plan is empty. Prepare a plan first."
    return True, ""


def guard_plan_approved(state: Day15TaskState) -> tuple[bool, str]:
    if not state.plan:
        return False, "Plan is empty. Prepare a plan first."
    if not state.plan_approved:
        return False, "Plan must be approved before execution."
    return True, ""


def guard_validation_passed(state: Day15TaskState) -> tuple[bool, str]:
    if state.validation_passed is True:
        return True, ""
    if state.validation_passed is False:
        return False, "Validation failed. Fix the result and validate again."
    return False, "Validation is required before completion."


GUARDS: dict[tuple[str, str], Guard] = {
    ("planning", "plan_approval"): guard_plan_not_empty,
    ("plan_approval", "execution"): guard_plan_approved,
    ("validation", "done"): guard_validation_passed,
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def expected_action_for(state: Day15TaskState) -> str:
    """Return the action expected in the task's current state/step."""
    if state.state == "execution" and state.plan:
        index = max(1, min(state.current_step, len(state.plan))) - 1
        return state.plan[index]
    return EXPECTED_ACTION[state.state]


def _contextual_reason(state: Day15TaskState, target: str) -> str:
    """Explain clearly WHY a structurally-disallowed transition is blocked."""
    if target == "execution" and state.state in {"planning", "plan_approval"}:
        if not state.plan_approved:
            return "Plan must be approved before execution."
    if target == "done":
        if state.validation_passed is not True:
            return "Validation is required before completion."
    if target == "plan_approval" and not state.plan:
        return "Plan is empty. Prepare a plan first."
    return (
        f"Transition {state.state} -> {target} is not allowed by the state machine."
    )


class InvalidTransitionError(ValueError):
    """Kept for callers that prefer an exception over a decision object."""

    def __init__(self, from_state: str, to_state: str) -> None:
        super().__init__(f"Transition {from_state} -> {to_state} is not allowed.")
        self.from_state = from_state
        self.to_state = to_state


class LifecycleMachine:
    """Pure transition guard + guard evaluation for :class:`Day15TaskState`."""

    # ------------------------------------------------------------------
    # structure
    # ------------------------------------------------------------------
    def allowed_transitions(self, from_state: str) -> list[str]:
        """Return the states reachable from ``from_state`` (stable order)."""
        allowed = ALLOWED_TRANSITIONS.get(from_state, set())
        return [candidate for candidate in STATES if candidate in allowed]

    def can_transition(self, from_state: str, to_state: str) -> bool:
        """True only when the table allows ``from_state -> to_state``."""
        return to_state in ALLOWED_TRANSITIONS.get(from_state, set())

    def guard_for(self, from_state: str, to_state: str) -> Guard | None:
        return GUARDS.get((from_state, to_state))

    # ------------------------------------------------------------------
    # evaluation (the CENTRAL mechanism)
    # ------------------------------------------------------------------
    def evaluate(self, state: Day15TaskState, target: str) -> TransitionDecision:
        """Decide whether ``state -> target`` is allowed, with a reason.

        Order of checks:

        1. a paused task cannot transition at all;
        2. the transition must exist in ``ALLOWED_TRANSITIONS``;
        3. its guard (if any) must return ``True``.
        """
        from_state = state.state

        if state.paused:
            return TransitionDecision(
                allowed=False,
                from_state=from_state,
                to_state=target,
                reason="Task is paused. Resume before transitioning.",
                rule="paused",
            )

        if not self.can_transition(from_state, target):
            return TransitionDecision(
                allowed=False,
                from_state=from_state,
                to_state=target,
                reason=_contextual_reason(state, target),
                rule="not_allowed",
            )

        guard = self.guard_for(from_state, target)
        if guard is not None:
            passed, reason = guard(state)
            if not passed:
                return TransitionDecision(
                    allowed=False,
                    from_state=from_state,
                    to_state=target,
                    reason=reason,
                    rule="guard",
                )

        return TransitionDecision(
            allowed=True,
            from_state=from_state,
            to_state=target,
            reason=f"Transition {from_state} -> {target} is allowed.",
            rule="allowed_transitions",
        )

    def apply(self, state: Day15TaskState, target: str) -> Day15TaskState:
        """Return a COPY moved to ``target`` (caller must have evaluated first)."""
        updated = state.model_copy(deep=True)
        previous = updated.state
        updated.state = target  # type: ignore[assignment]
        # Going back to planning (request changes OR execution->planning
        # rollback) invalidates the approval of the old plan version. The user
        # must approve the revised plan again before execution can resume.
        if target == "planning":
            updated.plan_approved = False
        # Entering validation starts a fresh validation round.
        if target == "validation":
            updated.validation_passed = None
            updated.validation_summary = ""
        updated.expected_action = expected_action_for(updated)
        if target == "execution" and updated.plan:
            index = max(1, min(updated.current_step, len(updated.plan))) - 1
            updated.expected_action = updated.plan[index]
        # ``previous_state`` is meaningful only while paused.
        if target != previous:
            updated.previous_state = None
        return updated

    def transition(self, state: Day15TaskState, target: str) -> Day15TaskState:
        """Evaluate and apply, raising :class:`InvalidTransitionError` if blocked."""
        decision = self.evaluate(state, target)
        if not decision.allowed:
            raise InvalidTransitionError(state.state, target)
        return self.apply(state, target)

    # ------------------------------------------------------------------
    # UI helper
    # ------------------------------------------------------------------
    def options(self, state: Day15TaskState) -> list[TransitionOption]:
        """Lifecycle transitions from the current state WITH lock status.

        Only transitions that exist in the table are returned; a structurally
        valid transition whose guard is unsatisfied is returned locked, so the
        UI can show ``EXECUTION — Locked: approve the plan first`` without
        offering the user a free choice of state.
        """
        options: list[TransitionOption] = []
        for candidate in self.allowed_transitions(state.state):
            decision = self.evaluate(state, candidate)
            requires = ""
            guard = self.guard_for(state.state, candidate)
            if guard is not None:
                requires = GUARD_LABELS.get((state.state, candidate), "")
            options.append(
                TransitionOption(
                    state=candidate,
                    allowed=decision.allowed,
                    locked_reason="" if decision.allowed else decision.reason,
                    requires=requires,
                )
            )
        return options


__all__ = [
    "ALLOWED_TRANSITIONS",
    "EXPECTED_ACTION",
    "GUARDS",
    "GUARD_LABELS",
    "ROLLBACK_TRANSITIONS",
    "STATES",
    "InvalidTransitionError",
    "LifecycleMachine",
    "expected_action_for",
    "guard_plan_approved",
    "guard_plan_not_empty",
    "guard_validation_passed",
    "is_rollback",
]
