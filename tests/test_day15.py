"""Day 15 — controlled state transitions tests.

No real DeepSeek call is made. The plan provider is deterministic. Tests focus
on the CONTRACT of the state machine: which transitions are allowed, which are
blocked (and why), that a blocked transition never mutates the state, that
pause/resume round-trips exactly, and that the transition history is kept.
"""
from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.schemas.day15 import Day15TaskState
from app.services.day15 import (
    ALLOWED_TRANSITIONS,
    Day15Error,
    Day15LifecycleService,
    LifecycleMachine,
    LifecycleStore,
    guard_plan_approved,
    guard_validation_passed,
)

PLAN = [
    "Define the data model",
    "Define the API endpoints",
    "Describe error handling",
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
class ScriptedProvider:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    async def generate(self, messages, *, model=None, max_tokens=1024, thinking=False):
        self.calls.append({"messages": [dict(m) for m in messages], "model": model})
        answer = self.responses.pop(0) if self.responses else _plan_json()
        return answer, "stop", {"total_tokens": 8}

    def last_messages(self):
        return self.calls[-1]["messages"] if self.calls else []


def _plan_json(plan: list[str] | None = None) -> str:
    return json.dumps({"plan": plan or PLAN}, ensure_ascii=False)


def _settings(tmp_path) -> Settings:
    return Settings(
        deepseek_api_key="test",
        environment="test",
        day15_task_path=str(tmp_path / "day15.json"),
    )


def _service(tmp_path, provider=None) -> Day15LifecycleService:
    settings = _settings(tmp_path)
    return Day15LifecycleService(
        settings,
        deepseek=provider or ScriptedProvider(),
        store=LifecycleStore(settings.day15_task_path),
    )


async def _planned_service(tmp_path) -> Day15LifecycleService:
    """Task already in PLAN_APPROVAL with a plan."""
    service = _service(tmp_path, ScriptedProvider([_plan_json()]))
    service.create_task("Build a REST API for a notes service")
    response = await service.prepare_plan()
    assert response.allowed
    assert response.state.state == "plan_approval"
    return service


async def _execution_service(tmp_path) -> Day15LifecycleService:
    service = await _planned_service(tmp_path)
    service.approve_plan()
    response = service.transition("execution")
    assert response.allowed
    return service


async def _validation_service(tmp_path) -> Day15LifecycleService:
    service = await _execution_service(tmp_path)
    response = service.transition("validation")
    assert response.allowed
    return service


# ----------------------------------------------------------------------
# 1) FSM table + guards
# ----------------------------------------------------------------------
def test_allowed_transitions_table():
    assert ALLOWED_TRANSITIONS == {
        "planning": {"plan_approval"},
        "plan_approval": {"execution", "planning"},
        "execution": {"validation", "planning"},
        "validation": {"done", "execution"},
        "done": set(),
    }


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        ("planning", "plan_approval"),
        ("plan_approval", "execution"),
        ("plan_approval", "planning"),
        ("execution", "validation"),
        ("execution", "planning"),
        ("validation", "done"),
        ("validation", "execution"),
    ],
)
def test_structurally_allowed(from_state, to_state):
    assert LifecycleMachine().can_transition(from_state, to_state)


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        ("planning", "execution"),
        ("planning", "done"),
        ("planning", "validation"),
        ("execution", "done"),
        ("validation", "planning"),
        ("done", "planning"),
        ("done", "execution"),
    ],
)
def test_structurally_forbidden(from_state, to_state):
    assert not LifecycleMachine().can_transition(from_state, to_state)


def test_guard_plan_approved_blocks_without_flag():
    state = Day15TaskState(task_id="t", state="plan_approval", plan=PLAN)
    passed, reason = guard_plan_approved(state)
    assert not passed
    assert "approved" in reason.lower()


def test_guard_validation_requires_pass():
    state = Day15TaskState(task_id="t", state="validation", validation_passed=None)
    assert guard_validation_passed(state)[0] is False
    assert "required" in guard_validation_passed(state)[1].lower()
    state.validation_passed = False
    assert guard_validation_passed(state)[0] is False
    state.validation_passed = True
    assert guard_validation_passed(state) == (True, "")


# ----------------------------------------------------------------------
# 2) explicit blocked / allowed transitions (service level)
# ----------------------------------------------------------------------
def test_planning_to_plan_approval_requires_a_plan(tmp_path):
    service = _service(tmp_path)
    service.create_task("task")
    # No plan yet -> guard fails, state stays in planning.
    response = service.transition("plan_approval")
    assert response.allowed is False
    assert response.state.state == "planning"


def test_planning_to_execution_is_blocked(tmp_path):
    service = _service(tmp_path)
    service.create_task("task")
    response = service.transition("execution")
    assert response.allowed is False
    assert response.state.state == "planning"
    assert response.reason == "Plan must be approved before execution."


def test_planning_to_done_is_blocked(tmp_path):
    service = _service(tmp_path)
    service.create_task("task")
    response = service.transition("done")
    assert response.allowed is False
    assert response.state.state == "planning"
    assert "validation" in response.reason.lower()


async def test_plan_approval_to_execution_without_approval_blocked(tmp_path):
    service = await _planned_service(tmp_path)
    response = service.transition("execution")
    assert response.allowed is False
    assert response.state.state == "plan_approval"
    assert response.reason == "Plan must be approved before execution."


async def test_plan_approval_to_execution_with_approval_allowed(tmp_path):
    service = await _planned_service(tmp_path)
    service.approve_plan()
    response = service.transition("execution")
    assert response.allowed is True
    assert response.state.state == "execution"


async def test_execution_to_done_is_blocked(tmp_path):
    service = await _execution_service(tmp_path)
    response = service.transition("done")
    assert response.allowed is False
    assert response.state.state == "execution"
    assert response.reason == "Validation is required before completion."


async def test_execution_to_validation_allowed(tmp_path):
    service = await _execution_service(tmp_path)
    response = service.transition("validation")
    assert response.allowed is True
    assert response.state.state == "validation"


async def test_execution_to_planning_rollback_allowed(tmp_path):
    """execution -> planning is a legitimate controlled rollback."""
    service = await _execution_service(tmp_path)
    assert service.state().plan_approved is True
    response = service.transition("planning")
    assert response.allowed is True
    assert response.state.state == "planning"
    # the stale approval MUST be reset by the rollback
    assert response.state.plan_approved is False


async def test_rollback_requires_fresh_approval(tmp_path):
    service = await _execution_service(tmp_path)
    service.transition("planning")
    assert service.state().plan_approved is False
    # cannot jump from planning to execution any more
    blocked = service.transition("execution")
    assert blocked.allowed is False
    assert blocked.state.state == "planning"
    # the full approval cycle must be repeated
    assert service.transition("plan_approval").allowed is True
    service.approve_plan()
    assert service.transition("execution").allowed is True
    assert service.state().plan_approved is True


async def test_revise_plan_intent_rolls_back(tmp_path):
    service = await _execution_service(tmp_path)
    response = await service.converse(
        message="нужно вернуться к плану и переделать его",
        forced_intent="revise_plan",
    )
    assert response.detected_intent == "revise_plan"
    assert response.decision is not None
    assert response.decision.allowed is True
    assert response.state.state == "planning"
    assert response.state.plan_approved is False
    assert response.transition_history[-1].intent == "revise_plan"


async def test_validation_to_done_without_pass_blocked(tmp_path):
    service = await _validation_service(tmp_path)
    response = service.transition("done")
    assert response.allowed is False
    assert response.state.state == "validation"


async def test_validation_failure_then_back_to_execution_allowed(tmp_path):
    service = await _validation_service(tmp_path)
    service.set_validation(False, "Missing error handling")
    assert service.transition("done").allowed is False
    back = service.transition("execution")
    assert back.allowed is True
    assert back.state.state == "execution"
    assert back.state.validation_passed is False


async def test_full_recovery_then_done(tmp_path):
    service = await _validation_service(tmp_path)
    service.set_validation(False)
    service.transition("execution")
    service.transition("validation")
    # Re-entering validation resets the previous result.
    assert service.state().validation_passed is None
    service.set_validation(True, "All good")
    done = service.transition("done")
    assert done.allowed is True
    assert done.state.state == "done"


async def test_validation_to_done_with_pass_allowed(tmp_path):
    service = await _validation_service(tmp_path)
    service.set_validation(True)
    response = service.transition("done")
    assert response.allowed is True
    assert response.state.state == "done"


# ----------------------------------------------------------------------
# 3) blocked transitions do NOT change the state
# ----------------------------------------------------------------------
async def test_blocked_transition_does_not_change_state(tmp_path):
    service = await _planned_service(tmp_path)
    before = service.state()
    snapshot = (
        before.state,
        before.current_step,
        before.expected_action,
        before.plan_approved,
        before.validation_passed,
        list(before.completed_steps),
    )
    try:
        service.transition("execution")
    except Day15Error:  # pragma: no cover - transition returns a decision
        pass
    after = service.state()
    assert (
        after.state,
        after.current_step,
        after.expected_action,
        after.plan_approved,
        after.validation_passed,
        list(after.completed_steps),
    ) == snapshot


# ----------------------------------------------------------------------
# 4) transition history (including blocked attempts)
# ----------------------------------------------------------------------
async def test_history_records_allowed_and_blocked(tmp_path):
    service = await _planned_service(tmp_path)
    # blocked attempt
    service.transition("execution")
    # approval + allowed transition
    service.approve_plan()
    allowed = service.transition("execution")

    history = allowed.transition_history
    statuses = [(r.from_state, r.to_state, r.status) for r in history]
    assert ("planning", "plan_approval", "allowed") in statuses
    assert ("plan_approval", "execution", "blocked") in statuses
    assert ("plan_approval", "execution", "allowed") in statuses
    blocked = [r for r in history if r.status == "blocked"][0]
    assert blocked.reason == "Plan must be approved before execution."
    assert blocked.timestamp
    # sequence numbers are strictly increasing
    assert [r.seq for r in history] == list(range(1, len(history) + 1))


# ----------------------------------------------------------------------
# 5) pause / resume
# ----------------------------------------------------------------------
async def test_pause_resume_execution(tmp_path):
    service = await _execution_service(tmp_path)
    before = service.state()
    paused = service.pause()
    assert paused.allowed is True
    assert paused.state.paused is True
    assert paused.state.state == "execution"
    assert paused.state.previous_state == "execution"
    assert paused.state.expected_action == before.expected_action

    resumed = service.resume()
    assert resumed.allowed is True
    assert resumed.state.paused is False
    assert resumed.state.state == "execution"
    assert resumed.state.previous_state is None
    assert resumed.state.expected_action == before.expected_action


async def test_pause_resume_validation(tmp_path):
    service = await _validation_service(tmp_path)
    paused = service.pause()
    assert paused.state.state == "validation"
    assert paused.state.previous_state == "validation"
    resumed = service.resume()
    assert resumed.state.state == "validation"
    assert resumed.state.paused is False


async def test_pause_resume_planning(tmp_path):
    service = _service(tmp_path)
    service.create_task("task")
    paused = service.pause()
    assert paused.state.state == "planning"
    assert paused.state.previous_state == "planning"
    resumed = service.resume()
    assert resumed.state.state == "planning"


async def test_pause_preserves_context_and_history(tmp_path):
    service = await _planned_service(tmp_path)
    service.approve_plan()
    service.transition("execution")
    history_len = len(service.state().transitions)
    paused = service.pause()
    assert paused.state.plan_approved is True
    assert len(paused.state.transitions) > history_len
    resumed = service.resume()
    assert resumed.state.plan_approved is True
    assert resumed.state.previous_state is None
    assert len(resumed.state.transitions) == history_len + 2


async def test_transition_while_paused_is_blocked(tmp_path):
    service = await _execution_service(tmp_path)
    service.pause()
    response = service.transition("validation")
    assert response.allowed is False
    assert "paused" in response.reason.lower()
    assert service.state().state == "execution"


async def test_cannot_pause_done(tmp_path):
    service = await _validation_service(tmp_path)
    service.set_validation(True)
    service.transition("done")
    response = service.pause()
    assert response.allowed is False
    assert service.state().state == "done"


# ----------------------------------------------------------------------
# 6) persistence across "requests" / restart
# ----------------------------------------------------------------------
async def test_state_and_history_survive_restart(tmp_path):
    service = await _execution_service(tmp_path)
    service.pause()
    expected = service.state()
    assert expected is not None

    restored = _service(tmp_path, ScriptedProvider())
    loaded = restored.state()
    assert loaded is not None
    assert loaded.state == "execution"
    assert loaded.paused is True
    assert loaded.previous_state == "execution"
    assert loaded.plan_approved is True
    assert len(loaded.transitions) == len(expected.transitions)


# ----------------------------------------------------------------------
# 7) options / lock status shown to the UI
# ----------------------------------------------------------------------
async def test_options_lock_execution_before_approval(tmp_path):
    service = await _planned_service(tmp_path)
    options = {o.state: o for o in service.machine.options(service.state())}
    assert options["execution"].allowed is False
    assert "approve" in options["execution"].locked_reason.lower()
    assert options["execution"].requires == "plan_approved"
    assert options["planning"].allowed is True


async def test_options_unlock_done_after_pass(tmp_path):
    service = await _validation_service(tmp_path)
    options = {o.state: o for o in service.machine.options(service.state())}
    assert options["done"].allowed is False
    service.set_validation(True)
    options = {o.state: o for o in service.machine.options(service.state())}
    assert options["done"].allowed is True


# ----------------------------------------------------------------------
# 8) assistant proposes; the machine decides
# ----------------------------------------------------------------------
async def test_chat_proposes_blocked_transition(tmp_path):
    service = await _planned_service(tmp_path)
    response = service.chat(
        message="Start implementation now.",
        proposed_transition="execution",
    )
    assert response.decision is not None
    assert response.decision.allowed is False
    assert response.state.state == "plan_approval"
    assert "BLOCKED" in response.answer
    # the attempt is still recorded
    assert response.transition_history[-1].status == "blocked"


async def test_chat_proposed_transition_respects_guards(tmp_path):
    service = await _execution_service(tmp_path)
    response = service.chat(message="Finish it.", proposed_transition="done")
    assert response.decision.allowed is False
    assert response.state.state == "execution"


# ----------------------------------------------------------------------
# 9) API level
# ----------------------------------------------------------------------
def test_api_state_starts_empty(client):
    body = client.get("/api/day15/state").json()
    assert body["has_task"] is False
    assert body["state"] is None


def test_api_create_and_blocked_skip(client):
    resp = client.post("/api/day15/task", json={"goal": "Build a REST API"})
    assert resp.status_code == 200
    assert resp.json()["state"]["state"] == "planning"

    blocked = client.post(
        "/api/day15/transition", json={"to_state": "execution"}
    )
    assert blocked.status_code == 200
    body = blocked.json()
    assert body["allowed"] is False
    assert body["reason"] == "Plan must be approved before execution."
    # state unchanged after the blocked transition
    assert client.get("/api/day15/state").json()["state"]["state"] == "planning"


def test_api_happy_path(client):
    client.post("/api/day15/task", json={"goal": "Build a REST API"})
    plan = client.post("/api/day15/plan", json={"message": ""}).json()
    assert plan["allowed"] is True
    assert plan["state"]["state"] == "plan_approval"

    blocked = client.post(
        "/api/day15/transition", json={"to_state": "execution"}
    ).json()
    assert blocked["allowed"] is False

    assert client.post("/api/day15/approve").json()["state"]["plan_approved"] is True
    assert client.post(
        "/api/day15/transition", json={"to_state": "execution"}
    ).json()["allowed"] is True
    assert client.post(
        "/api/day15/transition", json={"to_state": "validation"}
    ).json()["allowed"] is True

    # validation -> done blocked until validation passes
    assert client.post(
        "/api/day15/transition", json={"to_state": "done"}
    ).json()["allowed"] is False

    client.post("/api/day15/validation", json={"passed": True})
    done = client.post(
        "/api/day15/transition", json={"to_state": "done"}
    ).json()
    assert done["allowed"] is True
    assert done["state"]["state"] == "done"


def test_api_failed_validation_recovery(client):
    client.post("/api/day15/scenario/failed_validation")
    state = client.get("/api/day15/state").json()
    assert state["state"]["state"] == "validation"
    assert state["state"]["validation_passed"] is False

    assert client.post(
        "/api/day15/transition", json={"to_state": "done"}
    ).json()["allowed"] is False
    assert client.post(
        "/api/day15/transition", json={"to_state": "execution"}
    ).json()["allowed"] is True
    assert client.post(
        "/api/day15/transition", json={"to_state": "validation"}
    ).json()["allowed"] is True
    client.post("/api/day15/validation", json={"passed": True})
    assert client.post(
        "/api/day15/transition", json={"to_state": "done"}
    ).json()["allowed"] is True


def test_api_pause_resume(client):
    client.post("/api/day15/scenario/pause_resume")
    before = client.get("/api/day15/state").json()["state"]
    assert before["state"] == "execution"

    paused = client.post("/api/day15/pause").json()
    assert paused["allowed"] is True
    assert paused["state"]["paused"] is True
    assert paused["state"]["previous_state"] == "execution"
    assert paused["state"]["expected_action"] == before["expected_action"]

    # transitions are blocked while paused
    assert client.post(
        "/api/day15/transition", json={"to_state": "validation"}
    ).json()["allowed"] is False

    resumed = client.post("/api/day15/resume").json()
    assert resumed["state"]["state"] == "execution"
    assert resumed["state"]["paused"] is False
    assert resumed["state"]["expected_action"] == before["expected_action"]


def test_api_history_endpoint(client):
    client.post("/api/day15/task", json={"goal": "task"})
    client.post("/api/day15/transition", json={"to_state": "execution"})
    history = client.get("/api/day15/history").json()
    assert history["history"]
    assert history["history"][-1]["status"] == "blocked"
    assert history["history"][-1]["reason"] == (
        "Plan must be approved before execution."
    )


def test_api_scenario_unknown_is_404(client):
    assert client.post("/api/day15/scenario/nope").status_code == 404


def test_api_reset(client):
    client.post("/api/day15/task", json={"goal": "task"})
    client.delete("/api/day15/task")
    assert client.get("/api/day15/state").json()["has_task"] is False


def test_api_chat_without_task_is_404(client):
    assert client.post("/api/day15/chat", json={"message": "hi"}).status_code == 404


def test_api_day15_does_not_touch_day13_or_day7(client):
    client.post("/api/day15/task", json={"goal": "task"})
    client.post("/api/day15/transition", json={"to_state": "execution"})
    assert client.get("/api/day13/state").json()["has_task"] is False
    assert client.get("/api/chat/history").json()["count"] == 0
