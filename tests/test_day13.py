"""Day 13 — Task State Machine tests.

No real DeepSeek call is made: a deterministic scripted provider returns canned
plan/answer/verdict content. Tests assert on the FSM, the structured Task
State, persistence and the pause/resume continuation — never on free-form LLM
text.
"""
from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.schemas.day13 import TaskState
from app.services.day13 import (
    ALLOWED_TRANSITIONS,
    Day13TaskService,
    InvalidTransitionError,
    TaskStateError,
    TaskStateMachine,
    TaskStore,
    build_task_state_block,
    parse_plan,
    parse_verdict,
)

PLAN = [
    "Определить структуру данных",
    "Определить API endpoints",
    "Описать обработку ошибок",
    "Проверить итоговое решение",
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
class ScriptedProvider:
    """Returns queued responses (JSON plan, answer, JSON verdict)."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    async def generate(self, messages, *, model=None, max_tokens=1024, thinking=False):
        self.calls.append(
            {"messages": [dict(m) for m in messages], "model": model}
        )
        if self.responses:
            answer = self.responses.pop(0)
        else:
            answer = "Продолжение работы."
        return answer, "stop", {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

    def last_messages(self) -> list[dict[str, str]]:
        return self.calls[-1]["messages"] if self.calls else []


def _settings(tmp_path) -> Settings:
    return Settings(
        deepseek_api_key="test",
        environment="test",
        day13_task_path=str(tmp_path / "day13_task_state.json"),
    )


def _service(provider: ScriptedProvider, settings: Settings) -> Day13TaskService:
    return Day13TaskService(
        settings,
        deepseek=provider,
        store=TaskStore(settings.day13_task_path),
    )


def _plan_json(plan: list[str] | None = None) -> str:
    return json.dumps({"plan": plan or PLAN}, ensure_ascii=False)


def _pass_json() -> str:
    return json.dumps({"verdict": "pass", "summary": "Всё выполнено"}, ensure_ascii=False)


def _fail_json() -> str:
    return json.dumps({"verdict": "fail", "summary": "Не хватает деталей"}, ensure_ascii=False)


async def _reach_execution_step_two(service: Day13TaskService) -> TaskState:
    service.create_task("Спроектировать REST API для сервиса заметок")
    # planning -> execution
    await service.chat(message="Составь план")
    # execute step 1 -> current_step 2
    await service.chat(message="Выполни шаг 1")
    return service.state()


# ----------------------------------------------------------------------
# 1) FSM unit tests
# ----------------------------------------------------------------------
def test_allowed_transitions_table():
    assert ALLOWED_TRANSITIONS["planning"] == {"execution"}
    assert ALLOWED_TRANSITIONS["execution"] == {"validation"}
    assert ALLOWED_TRANSITIONS["validation"] == {"execution", "done"}
    assert ALLOWED_TRANSITIONS["done"] == set()


@pytest.mark.parametrize(
    "from_stage,to_stage",
    [
        ("planning", "execution"),
        ("execution", "validation"),
        ("validation", "done"),
        ("validation", "execution"),
    ],
)
def test_allowed_transitions_from_planning_to_done(from_stage, to_stage):
    assert TaskStateMachine().can_transition(from_stage, to_stage)


def test_forbidden_transition_planning_to_done_is_blocked():
    machine = TaskStateMachine()
    state = TaskState(task_id="t1", goal="g", stage="planning")
    assert not machine.can_transition("planning", "done")
    with pytest.raises(InvalidTransitionError):
        machine.transition(state, "done")
    # the original state is untouched
    assert state.stage == "planning"


@pytest.mark.parametrize(
    "from_stage,to_stage",
    [
        ("done", "execution"),
        ("done", "planning"),
        ("planning", "validation"),
        ("execution", "done"),
        ("validation", "planning"),
    ],
)
def test_other_forbidden_transitions_are_blocked(from_stage, to_stage):
    machine = TaskStateMachine()
    state = TaskState(task_id="t1", goal="g", stage=from_stage)
    with pytest.raises(InvalidTransitionError):
        machine.transition(state, to_stage)


def test_transition_returns_copy_and_sets_expected_action():
    machine = TaskStateMachine()
    state = TaskState(task_id="t1", goal="g", stage="planning")
    updated = machine.transition(state, "execution")
    assert updated is not state
    assert updated.stage == "execution"
    assert updated.expected_action == "Выполнить текущий шаг плана"
    assert state.stage == "planning"


# ----------------------------------------------------------------------
# 2) Plan / verdict parsing
# ----------------------------------------------------------------------
def test_parse_plan_from_json():
    assert parse_plan(_plan_json()) == PLAN


def test_parse_plan_fallback_numbered_list():
    raw = "1. Первый шаг\n2. Второй шаг\n3. Третий шаг"
    assert parse_plan(raw) == ["Первый шаг", "Второй шаг", "Третий шаг"]


def test_parse_plan_fallback_default_on_garbage():
    plan = parse_plan("не могу составить план")
    assert len(plan) >= 2


def test_parse_verdict_json_and_fallback():
    assert parse_verdict(_pass_json())[0] == "pass"
    assert parse_verdict(_fail_json())[0] == "fail"
    assert parse_verdict("Проверка pass")[0] == "pass"
    assert parse_verdict("непонятно")[0] == "fail"


# ----------------------------------------------------------------------
# 3) Store persistence
# ----------------------------------------------------------------------
def test_store_roundtrip(tmp_path):
    store = TaskStore(tmp_path / "task.json")
    assert store.load() is None
    state = TaskState(task_id="t1", goal="goal", stage="execution", current_step=2)
    store.save(state)
    loaded = TaskStore(tmp_path / "task.json").load()
    assert loaded == state


# ----------------------------------------------------------------------
# 4) create / planning / execution
# ----------------------------------------------------------------------
async def test_create_task_starts_in_planning(tmp_path):
    settings = _settings(tmp_path)
    service = _service(ScriptedProvider(), settings)
    state = service.create_task("Спроектировать REST API для заметок")
    assert state.stage == "planning"
    assert state.current_step == 1
    assert state.expected_action == "Составить план реализации"
    assert state.paused is False
    assert state.plan == []
    assert state.task_id.startswith("task-")


async def test_planning_turn_builds_plan_and_moves_to_execution(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json()])
    service = _service(provider, settings)
    service.create_task("Спроектировать REST API для заметок")

    response = await service.chat(message="Составь план")

    assert response.stage_event == "planning → execution"
    assert response.state.stage == "execution"
    assert response.state.plan == PLAN
    assert response.state.current_step == 1
    assert response.state.expected_action == PLAN[0]
    # the planning request carried the formal goal
    assert "Спроектировать REST API для заметок" in response.context_block


async def test_execution_turn_advances_current_step(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json(), "Шаг 1 выполнен."])
    service = _service(provider, settings)
    state = await _reach_execution_step_two(service)

    assert state.stage == "execution"
    assert state.current_step == 2
    assert state.expected_action == PLAN[1]
    assert state.completed_steps == [PLAN[0]]


async def test_execution_completes_and_moves_to_validation(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider(
        [_plan_json()] + ["ответ"] * len(PLAN)
    )
    service = _service(provider, settings)
    service.create_task("Спроектировать REST API")
    await service.chat(message="Составь план")
    # execute every plan step (step 1 already advanced in the last call)
    for _ in range(len(PLAN)):
        response = await service.chat(message="Продолжай")

    assert response.state.stage == "validation"
    assert response.state.completed_steps == PLAN
    assert response.stage_event == "execution → validation"
    assert response.state.expected_action == "Проверить полученный результат"


# ----------------------------------------------------------------------
# 5) Pause / Resume
# ----------------------------------------------------------------------
async def test_pause_preserves_stage_step_and_action(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json(), "ответ"])
    service = _service(provider, settings)
    await _reach_execution_step_two(service)
    before = service.state()
    assert (before.stage, before.current_step, before.expected_action) == (
        "execution",
        2,
        PLAN[1],
    )

    paused = service.pause()

    assert paused.paused is True
    assert paused.stage == "execution"
    assert paused.current_step == 2
    assert paused.expected_action == before.expected_action


async def test_resume_restores_exactly(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json(), "ответ"])
    service = _service(provider, settings)
    await _reach_execution_step_two(service)
    service.pause()
    expected_action = service.state().expected_action

    resumed = service.resume()

    assert resumed.paused is False
    assert resumed.stage == "execution"
    assert resumed.current_step == 2
    assert resumed.expected_action == expected_action


async def test_pause_blocks_work_but_text_resume_resumes(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json(), "ответ"])
    service = _service(provider, settings)
    await _reach_execution_step_two(service)
    service.pause()
    expected_action = service.state().expected_action

    # While paused, normal work is blocked...
    with pytest.raises(TaskStateError) as exc:
        await service.chat(message="Выполни шаг")
    assert exc.value.status_code == 409

    # ...but the text resume command resumes without re-explaining the task.
    response = await service.chat(message="Продолжай")
    assert response.state.paused is False
    assert response.state.stage == "execution"
    assert response.state.current_step == 2
    assert response.state.expected_action == expected_action


async def test_text_pause_resume_on_planning_stage(tmp_path):
    settings = _settings(tmp_path)
    service = _service(ScriptedProvider(), settings)
    service.create_task("Спроектировать REST API")
    assert service.state().stage == "planning"

    paused = await service.chat(message="пауза")
    assert paused.state.paused is True
    assert paused.state.stage == "planning"
    assert paused.state.current_step == 1

    resumed = await service.chat(message="продолжай")
    assert resumed.state.paused is False
    assert resumed.state.stage == "planning"
    assert resumed.state.current_step == 1


async def test_text_pause_resume_on_validation_stage(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json()] + ["ответ"] * len(PLAN))
    service = _service(provider, settings)
    service.create_task("task")
    await service.chat(message="план")
    for _ in range(len(PLAN)):
        await service.chat(message="дальше")
    assert service.state().stage == "validation"

    paused = await service.chat(message="пауза")
    assert paused.state.paused is True
    assert paused.state.stage == "validation"

    resumed = await service.chat(message="продолжай")
    assert resumed.state.paused is False
    assert resumed.state.stage == "validation"


async def test_pause_on_validation_stage(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json()] + ["ответ"] * len(PLAN))
    service = _service(provider, settings)
    service.create_task("task")
    await service.chat(message="план")
    for _ in range(len(PLAN)):
        await service.chat(message="дальше")
    assert service.state().stage == "validation"

    paused = service.pause()

    assert paused.paused is True
    assert paused.stage == "validation"


# ----------------------------------------------------------------------
# 6) Resume WITHOUT re-explaining the task (the key scenario)
# ----------------------------------------------------------------------
async def test_resume_continues_without_repeating_the_task(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider(
        [_plan_json(), "ответ 1", "ответ 2"]
    )
    service = _service(provider, settings)
    await _reach_execution_step_two(service)
    service.pause()
    saved_action = service.state().expected_action

    # A brand-new service instance simulates a separate HTTP request (and a
    # restart): it only has the persisted file, not the original goal text.
    restored = _service(ScriptedProvider(["ответ после resume"]), settings)
    restored_state = restored.state()
    assert restored_state is not None
    assert restored_state.paused is True
    assert restored_state.stage == "execution"
    assert restored_state.current_step == 2
    assert restored_state.expected_action == saved_action

    resumed = restored.resume()
    assert resumed.paused is False
    assert resumed.current_step == 2
    assert resumed.expected_action == saved_action

    # The user's message does NOT repeat the task/plan/step.
    response = await restored.chat(message="Продолжай")
    assert response.state.current_step == 3
    assert response.state.completed_steps == [PLAN[0], PLAN[1]]
    # The backend still injected the full formal state into the request.
    assert "TASK STATE" in response.context_block
    assert saved_action in response.context_block
    assert "Продолжай" == restored._deepseek.calls[-1]["messages"][-1]["content"]


# ----------------------------------------------------------------------
# 7) Validation outcomes are decided by CODE
# ----------------------------------------------------------------------
async def test_validation_pass_moves_to_done(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json()] + ["ответ"] * len(PLAN) + [_pass_json()])
    service = _service(provider, settings)
    service.create_task("task")
    await service.chat(message="план")
    for _ in range(len(PLAN)):
        await service.chat(message="дальше")
    assert service.state().stage == "validation"

    response = await service.chat(message="Проверь результат")

    assert response.stage_event == "validation → done"
    assert response.state.stage == "done"


async def test_validation_fail_returns_to_execution(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json()] + ["ответ"] * len(PLAN) + [_fail_json()])
    service = _service(provider, settings)
    service.create_task("task")
    await service.chat(message="план")
    for _ in range(len(PLAN)):
        await service.chat(message="дальше")

    response = await service.chat(message="Проверь результат")

    assert response.stage_event == "validation → execution"
    assert response.state.stage == "execution"
    assert response.state.current_step == len(PLAN)


# ----------------------------------------------------------------------
# 8) Explicit transitions are validated by code
# ----------------------------------------------------------------------
async def test_explicit_transition_allowed_and_forbidden(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json()])
    service = _service(provider, settings)
    service.create_task("task")
    await service.chat(message="план")
    assert service.state().stage == "execution"

    with pytest.raises(TaskStateError):
        service.transition("planning")  # not allowed from execution

    # execution -> validation is allowed explicitly
    updated = service.transition("validation")
    assert updated.stage == "validation"


def test_transition_without_task_raises(tmp_path):
    settings = _settings(tmp_path)
    service = _service(ScriptedProvider(), settings)
    with pytest.raises(TaskStateError) as exc:
        service.transition("execution")
    assert exc.value.status_code == 404


# ----------------------------------------------------------------------
# 9) Task-state block + LLM integration
# ----------------------------------------------------------------------
async def test_task_state_block_is_injected_into_request(tmp_path):
    settings = _settings(tmp_path)
    provider = ScriptedProvider([_plan_json(), "выполняю шаг"])
    service = _service(provider, settings)
    service.create_task("Спроектировать REST API для заметок")
    await service.chat(message="Составь план")

    await service.chat(message="Выполни шаг 1")

    sent = "\n".join(m["content"] for m in provider.last_messages())
    assert "TASK STATE" in sent
    assert "execution" in sent
    assert "CURRENT STEP" in sent
    assert PLAN[0] in sent  # expected action of the step being executed
    # the user message itself never repeats the goal
    assert provider.last_messages()[-1]["content"] == "Выполни шаг 1"


def test_build_task_state_block_contents():
    state = TaskState(
        task_id="t1",
        goal="Спроектировать REST API",
        stage="execution",
        current_step=2,
        expected_action=PLAN[1],
        plan=PLAN,
        completed_steps=[PLAN[0]],
    )
    block = build_task_state_block(state)
    assert "Спроектировать REST API" in block
    assert "execution" in block
    assert "2 из 4" in block
    assert PLAN[1] in block
    assert PLAN[0] in block


# ----------------------------------------------------------------------
# 10) API level
# ----------------------------------------------------------------------
def test_api_state_starts_empty(client):
    body = client.get("/api/day13/state").json()
    assert body["has_task"] is False
    assert body["state"] is None


def test_api_create_and_state(client):
    resp = client.post("/api/day13/task", json={"goal": "Спроектировать REST API"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_task"] is True
    assert body["state"]["stage"] == "planning"
    assert body["state"]["expected_action"] == "Составить план реализации"
    assert body["allowed_transitions"] == ["execution"]
    # persisted across requests
    assert client.get("/api/day13/state").json()["state"]["stage"] == "planning"


def test_api_planning_moves_to_execution(client):
    client.post("/api/day13/task", json={"goal": "Спроектировать REST API"})
    resp = client.post("/api/day13/chat", json={"message": "Составь план"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"]["stage"] == "execution"
    assert body["stage_event"] == "planning → execution"
    assert body["state"]["plan"]
    # planning context really carries the formal goal
    assert "Спроектировать REST API" in body["context_block"]


def test_api_pause_resume_flow(client):
    client.post("/api/day13/task", json={"goal": "Спроектировать REST API"})
    client.post("/api/day13/chat", json={"message": "Составь план"})
    client.post("/api/day13/chat", json={"message": "Шаг 1"})

    executed = client.get("/api/day13/state").json()["state"]
    assert executed["stage"] == "execution"
    assert executed["current_step"] == 2

    paused = client.post("/api/day13/pause").json()["state"]
    assert paused["paused"] is True
    assert paused["stage"] == "execution"
    assert paused["current_step"] == 2
    assert paused["expected_action"] == executed["expected_action"]

    # normal work is blocked while paused (the text resume command is tested
    # separately in test_api_text_pause_resume_flow)
    blocked = client.post("/api/day13/chat", json={"message": "Выполни шаг"})
    assert blocked.status_code == 409

    resumed = client.post("/api/day13/resume").json()["state"]
    assert resumed["paused"] is False
    assert resumed["stage"] == "execution"
    assert resumed["current_step"] == 2
    assert resumed["expected_action"] == executed["expected_action"]

    # continues from the saved step without re-explaining the task
    continued = client.post("/api/day13/chat", json={"message": "Продолжай"})
    assert continued.status_code == 200
    assert continued.json()["state"]["current_step"] == 3


def test_api_text_pause_resume_flow(client):
    client.post("/api/day13/task", json={"goal": "Спроектировать REST API"})
    client.post("/api/day13/chat", json={"message": "Составь план"})
    client.post("/api/day13/chat", json={"message": "Шаг 1"})

    executed = client.get("/api/day13/state").json()["state"]
    assert executed["stage"] == "execution"
    assert executed["current_step"] == 2

    # Text command "пауза" must set paused=true WITHOUT moving stage/step/action.
    paused_resp = client.post("/api/day13/chat", json={"message": "пауза"})
    assert paused_resp.status_code == 200
    paused = paused_resp.json()["state"]
    assert paused["paused"] is True
    assert paused["stage"] == "execution"
    assert paused["current_step"] == 2
    assert paused["expected_action"] == executed["expected_action"]

    # Text command "продолжай" must resume to the SAME stage/step/action.
    resumed_resp = client.post("/api/day13/chat", json={"message": "продолжай"})
    assert resumed_resp.status_code == 200
    resumed = resumed_resp.json()["state"]
    assert resumed["paused"] is False
    assert resumed["stage"] == "execution"
    assert resumed["current_step"] == 2
    assert resumed["expected_action"] == executed["expected_action"]


def test_api_forbidden_transition_blocked(client):
    client.post("/api/day13/task", json={"goal": "task"})
    resp = client.post("/api/day13/transition", json={"to_stage": "done"})
    assert resp.status_code == 400
    # state stayed in planning
    assert client.get("/api/day13/state").json()["state"]["stage"] == "planning"


def test_api_transition_allowed(client):
    client.post("/api/day13/task", json={"goal": "task"})
    resp = client.post("/api/day13/transition", json={"to_stage": "execution"})
    assert resp.status_code == 200
    assert resp.json()["state"]["stage"] == "execution"


def test_api_chat_without_task_is_404(client):
    resp = client.post("/api/day13/chat", json={"message": "привет"})
    assert resp.status_code == 404


def test_api_day13_does_not_touch_day7_history(client):
    client.post("/api/day13/task", json={"goal": "task"})
    client.post("/api/day13/chat", json={"message": "Составь план"})
    assert client.get("/api/chat/history").json()["count"] == 0


def test_api_old_endpoints_still_work(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/day12/profile").status_code == 200
    assert client.get("/api/day11/state").status_code == 200
