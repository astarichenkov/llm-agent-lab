"""Unit + API tests for Week 5 / Day 25 — Mini Chat with RAG + Task Memory.

No test touches a real Ollama/DeepSeek: the Day 25 chat service is wired to
the same fakes used by Day 22-24 (deterministic retrieval, a scripted
grounded generation and a rule-based task-state updater).
"""
import json

import pytest

from app.schemas.day25 import (
    MSG_STATUS_ANSWERED,
    MSG_STATUS_INSUFFICIENT,
    StateOperation,
    TaskState,
)
from app.services.day25.chat_service import Day25ChatService
from app.services.day25.contextual_query import build_contextual_query
from app.services.day25.repository import ChatRepository
from app.services.day25.task_state import (
    apply_operations,
    default_task_state,
    state_contains,
    state_to_prompt_text,
)
from app.services.day25.updater import (
    RuleBasedTaskStateExtractor,
    TaskStateUpdater,
    classify_turn,
)
from app.services.day25.evaluation import Day25EvaluationService


# ----------------------------------------------------------------------
# 1-5: sessions, messages, history, persistence
# ----------------------------------------------------------------------
def test_create_chat_session(day25_repository):
    session = day25_repository.create_session()
    assert session.id
    assert session.title == "Новый чат"
    assert day25_repository.get_session(session.id).id == session.id


def test_save_user_message(day25_repository):
    session = day25_repository.create_session()
    message = day25_repository.add_message(session.id, "user", "Пробег 82 тысячи.")
    assert message.role == "user"
    messages = day25_repository.list_messages(session.id)
    assert [m.content for m in messages] == ["Пробег 82 тысячи."]


def test_save_assistant_message(day25_repository):
    session = day25_repository.create_session()
    day25_repository.add_message(session.id, "user", "Вопрос?")
    assistant = day25_repository.add_message(
        session.id, "assistant", "Ответ", status=MSG_STATUS_ANSWERED
    )
    assert assistant.status == MSG_STATUS_ANSWERED
    assert day25_repository.list_messages(session.id)[1].role == "assistant"


def test_load_history_order(day25_repository):
    session = day25_repository.create_session()
    day25_repository.add_message(session.id, "user", "1")
    day25_repository.add_message(session.id, "assistant", "2")
    day25_repository.add_message(session.id, "user", "3")
    assert [m.content for m in day25_repository.list_messages(session.id)] == ["1", "2", "3"]


def test_persistence_after_reopen(tmp_path, settings):
    path = settings.day25_chat_db_path
    repo = ChatRepository(path)
    session = repo.create_session("Диагностика")
    repo.add_message(session.id, "user", "Пробег 82 тысячи.")
    repo.save_state(
        session.id,
        apply_operations(
            default_task_state(),
            [StateOperation(op="update_vehicle", field="mileage_km", value=82000)],
        ),
    )

    reopened = ChatRepository(path)
    assert reopened.get_session(session.id).title == "Диагностика"
    assert [m.content for m in reopened.list_messages(session.id)] == ["Пробег 82 тысячи."]
    assert reopened.get_state(session.id).vehicle.mileage_km == 82000


# ----------------------------------------------------------------------
# 6-13: task state operations
# ----------------------------------------------------------------------
def test_create_initial_task_state():
    state = default_task_state()
    assert state.goal is None
    assert state.vehicle.model == "Mitsubishi Xpander"
    assert state.known_facts == []
    assert state.hypotheses == []


def test_add_fact():
    state = apply_operations(
        default_task_state(), [StateOperation(op="add_fact", text="вибрация на D")]
    )
    assert state_contains(state, "вибрация на D")


def test_update_existing_fact_by_key():
    state = apply_operations(
        default_task_state(),
        [
            StateOperation(op="add_fact", text="пробег 80", key="mileage"),
            StateOperation(op="add_fact", text="пробег 92", key="mileage"),
        ],
    )
    matching = [f for f in state.known_facts if f.key == "mileage"]
    assert len(matching) == 1
    assert matching[0].text == "пробег 92"


def test_correction_supersedes_mileage():
    state = apply_operations(
        default_task_state(),
        [
            StateOperation(op="update_vehicle", field="mileage_km", value=80000),
            StateOperation(op="update_vehicle", field="mileage_km", value=92000),
        ],
    )
    assert state.vehicle.mileage_km == 92000
    facts = [f.text for f in state.known_facts if f.key == "vehicle_mileage_km"]
    assert facts == ["пробег: 92000"]


def test_hypothesis_is_not_confirmed_fact():
    state = apply_operations(
        default_task_state(),
        [StateOperation(op="add_hypothesis", text="engine mount", origin="user")],
    )
    assert [h.text for h in state.hypotheses] == ["engine mount"]
    assert all("engine mount" not in f.text for f in state.known_facts)


def test_checks_performed():
    state = apply_operations(
        default_task_state(),
        [StateOperation(op="add_check", text="right engine mount replaced")],
    )
    assert [c.text for c in state.checks_performed] == ["right engine mount replaced"]


def test_result():
    state = apply_operations(
        default_task_state(),
        [StateOperation(op="add_result", text="vibration remained")],
    )
    assert [r.text for r in state.results] == ["vibration remained"]


def test_goal_retention():
    state = default_task_state()
    state = apply_operations(state, [StateOperation(op="set_goal", text="diagnose D vibration")])
    state = apply_operations(state, [StateOperation(op="add_fact", text="cold")])
    assert state.goal == "diagnose D vibration"


# ----------------------------------------------------------------------
# 14-16: ContextualQueryBuilder
# ----------------------------------------------------------------------
def test_contextual_query_uses_current_message():
    state = default_task_state()
    query = build_contextual_query("А что проверить первым?", state)
    assert "А что проверить первым?" in query


def test_contextual_query_uses_task_state():
    state = apply_operations(
        default_task_state(),
        [
            StateOperation(op="set_goal", text="diagnose vibration"),
            StateOperation(op="update_vehicle", field="transmission", value="CVT"),
            StateOperation(op="update_vehicle", field="mileage_km", value=92000),
        ],
    )
    query = build_contextual_query("follow up", state)
    assert "diagnose vibration" in query
    assert "CVT" in query
    assert "92000" in query


def test_followup_becomes_self_contained():
    state = apply_operations(
        default_task_state(),
        [
            StateOperation(op="update_vehicle", field="transmission", value="CVT"),
            StateOperation(op="update_vehicle", field="mileage_km", value=92000),
        ],
    )
    query = build_contextual_query("А эта рекомендация подходит?", state)
    assert "CVT" in query and "92000" in query


# ----------------------------------------------------------------------
# 17-25: pipeline reuse, evidence, recovery, isolation
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rag_runs_for_new_technical_question(
    day25_service, day22_rag_service
):
    session = day25_service.create_session()
    await day25_service.send_message(session.id, "Пробег 82 тысячи.")
    searches_after_fact = len(day22_rag_service.search_calls)
    result = await day25_service.send_message(
        session.id, "А что проверить первым?"
    )
    assert len(day22_rag_service.search_calls) > searches_after_fact
    assert result.grounded is not None


@pytest.mark.asyncio
async def test_uses_day23_improved_retrieval(day25_service, day22_rag_service):
    session = day25_service.create_session()
    await day25_service.send_message(session.id, "Как проверить аккумулятор зимой?")
    # Day 23 improved retrieval uses retrieval_top_k=20 (baseline would be 5).
    assert day22_rag_service.search_calls[-1][1] == 20


@pytest.mark.asyncio
async def test_uses_day24_grounding(day25_service):
    session = day25_service.create_session()
    result = await day25_service.send_message(
        session.id, "Какое давление в шинах 205/55R16?"
    )
    assert result.grounded is not None
    assert result.grounded.retrieval.gate_reason
    assert result.grounded.status == "answered"
    assert result.grounded.grounding_valid is True


@pytest.mark.asyncio
async def test_grounded_answer_saves_evidence(day25_service):
    session = day25_service.create_session()
    result = await day25_service.send_message(
        session.id, "Какое давление в шинах 205/55R16?"
    )
    assert result.assistant_message.status == MSG_STATUS_ANSWERED
    assert len(result.assistant_message.evidence) >= 1
    evidence = result.assistant_message.evidence[0]
    assert evidence.chunk_id
    assert evidence.quote_valid is True
    assert evidence.source


@pytest.mark.asyncio
async def test_evidence_restored_after_reload(settings, day25_service):
    session = day25_service.create_session()
    await day25_service.send_message(session.id, "Какое давление в шинах?")
    reopened = Day25ChatService(
        settings,
        repository=ChatRepository(settings.day25_chat_db_path),
        grounded_service=day25_service.grounded,
        updater=TaskStateUpdater(RuleBasedTaskStateExtractor()),
    )
    detail = reopened.get_detail(session.id)
    assistant = [m for m in detail.messages if m.role == "assistant"]
    assert assistant and assistant[0].evidence
    assert assistant[0].evidence[0].quote_valid is True


@pytest.mark.asyncio
async def test_insufficient_context_saved_correctly(
    day25_service, day22_rag_service
):
    day22_rag_service.force_low_score = True
    session = day25_service.create_session()
    result = await day25_service.send_message(
        session.id, "Как адаптировать вариатор сканером?"
    )
    assert result.assistant_message.status == MSG_STATUS_INSUFFICIENT
    assert result.assistant_message.evidence == []
    assert "недостаточно" in result.assistant_message.content.lower()


@pytest.mark.asyncio
async def test_rejected_answer_creates_no_fake_sources(
    day25_service, day22_rag_service
):
    day22_rag_service.force_low_score = True
    session = day25_service.create_session()
    result = await day25_service.send_message(
        session.id, "Сколько масла в двигателе Hyundai Creta?"
    )
    assert result.assistant_message.status == MSG_STATUS_INSUFFICIENT
    assert day25_service.repository.list_evidence(result.assistant_message.id) == []


@pytest.mark.asyncio
async def test_recent_history_limit(day24_generation, day25_service):
    session = day25_service.create_session()
    for index in range(8):
        await day25_service.send_message(session.id, f"Факт номер {index}.")
    await day25_service.send_message(session.id, "А что дальше делать?")
    prompt = day24_generation.calls[-1]["messages"][-1]["content"]
    recent_block = prompt.split("RECENT CONVERSATION:")[-1].split("RAG EVIDENCE:")[0]
    role_lines = [
        line for line in recent_block.splitlines()
        if line.startswith("USER:") or line.startswith("ASSISTANT:")
    ]
    assert len(role_lines) <= day25_service.recent_messages


@pytest.mark.asyncio
async def test_old_important_fact_survives_in_task_state(day25_service):
    session = day25_service.create_session()
    await day25_service.send_message(session.id, "Коробка CVT.")
    for index in range(8):
        await day25_service.send_message(session.id, f"Ещё деталь {index}.")
    state = day25_service.get_state(session.id)
    assert state.vehicle.transmission == "CVT"
    assert state_contains(state, "CVT")


@pytest.mark.asyncio
async def test_session_isolation(day25_service):
    first = day25_service.create_session()
    second = day25_service.create_session()
    await day25_service.send_message(first.id, "Пробег 82 тысячи.")
    assert day25_service.repository.message_count(second.id) == 0
    assert day25_service.get_state(second.id).vehicle.mileage_km is None
    assert day25_service.get_state(first.id).vehicle.mileage_km == 82000


@pytest.mark.asyncio
async def test_state_does_not_leak_between_sessions(day25_service):
    first = day25_service.create_session()
    second = day25_service.create_session()
    await day25_service.send_message(first.id, "Хочу разобраться с вариатором.")
    assert day25_service.get_state(second.id).goal is None


# ----------------------------------------------------------------------
# 28-33: API
# ----------------------------------------------------------------------
def test_api_create_session(client):
    response = client.post("/api/week5/day25/sessions", json={})
    assert response.status_code == 201
    body = response.json()
    assert body["id"]
    assert body["title"] == "Новый чат"


def test_api_list_sessions(client):
    created = client.post("/api/week5/day25/sessions", json={"title": "Тест"}).json()
    response = client.get("/api/week5/day25/sessions")
    assert response.status_code == 200
    ids = [s["id"] for s in response.json()["sessions"]]
    assert created["id"] in ids


def test_api_get_session(client):
    created = client.post("/api/week5/day25/sessions", json={}).json()
    response = client.get(f"/api/week5/day25/sessions/{created['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["session"]["id"] == created["id"]
    assert body["messages"] == []
    assert body["state"]["goal"] is None


def test_api_send_message(client):
    created = client.post("/api/week5/day25/sessions", json={}).json()
    response = client.post(
        f"/api/week5/day25/sessions/{created['id']}/messages",
        json={"content": "Какое давление в шинах 205/55R16?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user_message"]["content"].startswith("Какое давление")
    assert body["assistant_message"]["role"] == "assistant"
    assert body["grounded"]["status"] == "answered"


def test_api_get_state(client):
    created = client.post("/api/week5/day25/sessions", json={}).json()
    response = client.get(f"/api/week5/day25/sessions/{created['id']}/state")
    assert response.status_code == 200
    assert response.json()["state"]["goal"] is None


def test_api_unknown_session(client):
    response = client.get("/api/week5/day25/sessions/does-not-exist")
    assert response.status_code == 404


def test_api_persistence_across_app_restart(client, settings):
    created = client.post("/api/week5/day25/sessions", json={"title": "Persist"}).json()
    client.post(
        f"/api/week5/day25/sessions/{created['id']}/messages",
        json={"content": "Пробег 92 тысячи."},
    )
    # A fresh repository over the same file simulates an application restart.
    reopened = ChatRepository(settings.day25_chat_db_path)
    assert reopened.get_session(created["id"]).title == "Persist"
    assert reopened.get_state(created["id"]).vehicle.mileage_km == 92000


def test_existing_day22_24_endpoints_still_work(client):
    assert client.get("/api/week5/day22/status").status_code == 200
    assert client.get("/api/week5/day23/status").status_code == 200
    assert client.get("/api/week5/day24/status").status_code == 200


# ----------------------------------------------------------------------
# Evaluation runner
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_metrics(day25_evaluation_service):
    result = await day25_evaluation_service.run_scenario("scenario_01_diagnostics")
    metrics = result.metrics
    assert metrics.turns_completed == metrics.turns_total
    assert metrics.goal_retained is True
    assert metrics.corrections_expected == 1
    assert metrics.corrections_applied == 1
    assert metrics.facts_retained >= 1
    assert metrics.contextual_followups == 1
    assert metrics.contextual_followups_resolved == 1


def test_state_to_prompt_text_and_classification():
    state = apply_operations(
        default_task_state(),
        [StateOperation(op="set_goal", text="battery")],
    )
    text = state_to_prompt_text(state)
    assert "GOAL: battery" in text
    assert classify_turn("Наверное, это батарея", [
        StateOperation(op="add_hypothesis", text="x")
    ]) == "fact_update"
    assert classify_turn("Пробег 82 тысячи.", [
        StateOperation(op="update_vehicle", field="mileage_km", value=82000)
    ]) == "fact_update"
    # With no operations we cannot know the intent; the fallback is question.
    assert classify_turn("Пробег 82 тысячи.", []) == "question"


# ----------------------------------------------------------------------
# Integration smoke test (real Day 21 index + real generation provider)
# ----------------------------------------------------------------------
@pytest.mark.integration
def test_day25_integration_smoke(tmp_path, monkeypatch) -> None:
    """Real chat: 6-7 messages, memory persists across a repository reopen.

    Run explicitly::

        pytest -m integration tests/test_day25.py

    Requires the Day 21 index, Ollama/bge-m3 and a reachable generation
    provider with a usable chat model.
    """
    import asyncio

    from app.config import get_settings
    from app.services.day25 import ChatRepository, Day25ChatService

    settings = get_settings().model_copy(
        update={"day25_chat_db_path": str(tmp_path / "day25_integration.sqlite3")}
    )
    service = Day25ChatService(settings)
    session = service.create_session()

    messages = [
        "Хочу разобраться, почему машина плохо заводится зимой.",
        "Пробег 82 тысячи, коробка CVT.",
        "Аккумулятор родной, ему 4 года.",
        "Что проверить первым?",
        "Как часто заряжать аккумулятор при длительном простое?",
        "А на морозе ёмкость аккумулятора падает?",
        "Подведи итог.",
    ]
    for message in messages:
        result = asyncio.run(service.send_message(session.id, message))
        assert result.user_message.content == message

    state = service.get_state(session.id)
    assert state.vehicle.mileage_km == 82000
    assert state.vehicle.transmission == "CVT"

    detail = service.get_detail(session.id)
    answered = [
        m for m in detail.messages
        if m.role == "assistant" and m.status == MSG_STATUS_ANSWERED
    ]
    assert answered, "at least one grounded answer was expected"
    assert any(message.evidence for message in answered)
    for message in answered:
        for evidence in message.evidence:
            assert evidence.chunk_id

    # Simulate a backend restart: a brand-new repository/service pair.
    reopened = Day25ChatService(
        settings,
        repository=ChatRepository(settings.day25_chat_db_path),
        grounded_service=service.grounded,
        updater=service.updater,
    )
    resumed = reopened.get_detail(session.id)
    assert resumed.session.id == session.id
    assert resumed.state.vehicle.mileage_km == 82000
    assert len(resumed.messages) >= len(messages)
    assert any(m.evidence for m in resumed.messages if m.role == "assistant")
