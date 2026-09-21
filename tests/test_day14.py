"""Day 14 — invariants and state constraints tests.

No real DeepSeek call is made: a deterministic scripted provider records the
exact messages it receives. Conflict detection and the refusal text are
produced by code, so the conflict tests never depend on LLM wording.
"""
from __future__ import annotations

import json

from app.config import Settings
from app.schemas.day14 import Invariant
from app.services.day14 import (
    DEFAULT_INVARIANTS,
    Day14InvariantService,
    InvariantStore,
    build_invariants_block,
    detect_conflicts,
)

ALLOWED_REQUEST = "Добавь endpoint /health в существующий FastAPI backend."
ONE_CONFLICT_REQUEST = "Перепиши backend на Go."
MULTI_CONFLICT_REQUEST = (
    "Перепиши backend на Go, разбей его на микросервисы и замени PostgreSQL "
    "на MongoDB."
)
BUSINESS_CONFLICT_REQUEST = "Удаляй аккаунт пользователя сразу без подтверждения."


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
class ScriptedProvider:
    """Records the messages passed to the model, returns a canned answer."""

    def __init__(self, answer: str = "Готово: endpoint /health добавлен.") -> None:
        self.answer = answer
        self.calls: list[dict] = []

    async def generate(self, messages, *, model=None, max_tokens=1024, thinking=False):
        self.calls.append({"messages": [dict(m) for m in messages], "model": model})
        return self.answer, "stop", {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

    def last_messages(self) -> list[dict[str, str]]:
        return self.calls[-1]["messages"] if self.calls else []


def _settings(tmp_path) -> Settings:
    return Settings(
        deepseek_api_key="test",
        environment="test",
        day14_invariants_path=str(tmp_path / "day14_invariants.json"),
    )


def _service(provider: ScriptedProvider, settings: Settings) -> Day14InvariantService:
    return Day14InvariantService(
        settings,
        deepseek=provider,
        store=InvariantStore(settings.day14_invariants_path),
    )


# ----------------------------------------------------------------------
# 1) invariants are stored separately from the chat history
# ----------------------------------------------------------------------
def test_default_invariants_cover_required_categories():
    by_id = {inv.id: inv for inv in DEFAULT_INVARIANTS}
    assert by_id["architecture"].category == "architecture"
    assert by_id["database"].category == "technical_decision"
    assert by_id["stack"].category == "stack"
    assert by_id["business_rule"].category == "business"


def test_store_roundtrip_keeps_invariants_separate(tmp_path):
    settings = _settings(tmp_path)
    store = InvariantStore(settings.day14_invariants_path)
    invariants = [Invariant(id="custom", category="stack", rule="Только Python")]
    store.save(invariants)

    # The file contains ONLY invariants — never conversation messages.
    raw = json.loads(
        (tmp_path / "day14_invariants.json").read_text(encoding="utf-8")
    )
    assert "invariants" in raw
    assert "messages" not in raw
    assert [inv.id for inv in store.load()] == ["custom"]


async def test_invariants_are_not_part_of_chat_history(tmp_path):
    provider = ScriptedProvider()
    service = _service(provider, _settings(tmp_path))

    assert service.messages == []
    assert len(service.invariants) == 4

    await service.chat(message=ALLOWED_REQUEST)

    # Conversation has exactly the user + assistant turn.
    assert [m.role for m in service.messages] == ["user", "assistant"]
    rules = [inv.rule for inv in service.invariants]
    for message in service.messages:
        for rule in rules:
            assert rule not in message.content

    # The two state views are exposed as separate fields.
    state = service.state_response()
    assert len(state.invariants) == 4
    assert len(state.messages) == 2
    assert state.count == 2


# ----------------------------------------------------------------------
# 2) allowed request
# ----------------------------------------------------------------------
async def test_allowed_request_is_processed(tmp_path):
    provider = ScriptedProvider()
    service = _service(provider, _settings(tmp_path))

    response = await service.chat(message=ALLOWED_REQUEST)

    assert response.allowed is True
    assert response.conflicts == []
    assert response.answer == provider.answer
    assert len(provider.calls) == 1


# ----------------------------------------------------------------------
# 3) a single conflict
# ----------------------------------------------------------------------
def test_single_conflict_is_detected():
    conflicts = detect_conflicts(ONE_CONFLICT_REQUEST, DEFAULT_INVARIANTS)
    assert [c.invariant_id for c in conflicts] == ["stack"]


async def test_single_conflict_is_refused_and_named(tmp_path):
    provider = ScriptedProvider()
    service = _service(provider, _settings(tmp_path))

    response = await service.chat(message=ONE_CONFLICT_REQUEST)

    assert response.allowed is False
    assert [c.invariant_id for c in response.conflicts] == ["stack"]
    assert "Backend реализуется на Python" in response.answer
    assert "Stack" in response.answer
    # The model is NOT asked to agree with a violating request.
    assert provider.calls == []


# ----------------------------------------------------------------------
# 4) several conflicts
# ----------------------------------------------------------------------
def test_multiple_conflicts_are_all_detected():
    conflicts = detect_conflicts(MULTI_CONFLICT_REQUEST, DEFAULT_INVARIANTS)
    assert {c.invariant_id for c in conflicts} == {
        "architecture",
        "database",
        "stack",
    }


async def test_multiple_conflicts_are_all_reported(tmp_path):
    provider = ScriptedProvider()
    service = _service(provider, _settings(tmp_path))

    response = await service.chat(message=MULTI_CONFLICT_REQUEST)

    assert response.allowed is False
    ids = {c.invariant_id for c in response.conflicts}
    assert ids == {"architecture", "database", "stack"}
    assert "монолитным FastAPI-приложением" in response.answer
    assert "PostgreSQL" in response.answer
    assert "Backend реализуется на Python" in response.answer
    assert provider.calls == []


# ----------------------------------------------------------------------
# 5) business rule
# ----------------------------------------------------------------------
def test_business_rule_conflict_is_detected():
    conflicts = detect_conflicts(BUSINESS_CONFLICT_REQUEST, DEFAULT_INVARIANTS)
    assert [c.invariant_id for c in conflicts] == ["business_rule"]
    assert conflicts[0].category == "business"


async def test_business_rule_is_enforced(tmp_path):
    provider = ScriptedProvider()
    service = _service(provider, _settings(tmp_path))

    response = await service.chat(message=BUSINESS_CONFLICT_REQUEST)

    assert response.allowed is False
    assert [c.invariant_id for c in response.conflicts] == ["business_rule"]
    assert "явного подтверждения" in response.answer
    assert provider.calls == []


# ----------------------------------------------------------------------
# 6) invariants really reach the model request
# ----------------------------------------------------------------------
def test_build_invariants_block_contains_every_rule():
    block = build_invariants_block(DEFAULT_INVARIANTS)
    assert "ACTIVE INVARIANTS" in block
    assert "cannot be violated" in block
    for invariant in DEFAULT_INVARIANTS:
        assert invariant.rule in block


async def test_invariants_block_is_injected_into_model_request(tmp_path):
    provider = ScriptedProvider()
    service = _service(provider, _settings(tmp_path))

    response = await service.chat(message=ALLOWED_REQUEST)

    messages = provider.last_messages()
    system_messages = [m for m in messages if m["role"] == "system"]
    assert len(system_messages) == 2
    block = system_messages[1]["content"]
    assert "ACTIVE INVARIANTS" in block
    for invariant in DEFAULT_INVARIANTS:
        assert invariant.rule in block
    # The response exposes the SAME block that was sent to the model.
    assert response.context_block == block
    # The user request appears EXACTLY ONCE, as the last message.
    user_messages = [m for m in messages if m["role"] == "user"]
    assert user_messages == [{"role": "user", "content": ALLOWED_REQUEST}]
    assert messages[-1] == {"role": "user", "content": ALLOWED_REQUEST}


def test_build_context_is_deterministic(tmp_path):
    service = _service(ScriptedProvider(), _settings(tmp_path))
    messages, block = service.build_context(ALLOWED_REQUEST)
    assert messages[0]["content"].startswith("You are a helpful software")
    assert "ACTIVE INVARIANTS" in messages[1]["content"]
    assert messages[-1]["content"] == ALLOWED_REQUEST
    assert block == messages[1]["content"]


# ----------------------------------------------------------------------
# API tests
# ----------------------------------------------------------------------
def test_api_state_returns_separate_invariants_and_messages(client):
    res = client.get("/api/day14/state")
    assert res.status_code == 200
    body = res.json()
    assert len(body["invariants"]) == 4
    assert body["messages"] == []


def test_api_allowed_request(client):
    res = client.post("/api/day14/chat", json={"message": ALLOWED_REQUEST})
    assert res.status_code == 200
    body = res.json()
    assert body["allowed"] is True
    assert body["conflicts"] == []
    assert "ACTIVE INVARIANTS" in body["context_block"]

    state = client.get("/api/day14/state").json()
    assert len(state["messages"]) == 2


def test_api_conflict_request(client):
    res = client.post("/api/day14/chat", json={"message": MULTI_CONFLICT_REQUEST})
    assert res.status_code == 200
    body = res.json()
    assert body["allowed"] is False
    assert {c["invariant_id"] for c in body["conflicts"]} == {
        "architecture",
        "database",
        "stack",
    }


def test_api_clear_history_keeps_invariants(client):
    client.post("/api/day14/chat", json={"message": ALLOWED_REQUEST})
    res = client.delete("/api/day14/history")
    assert res.status_code == 200

    state = client.get("/api/day14/state").json()
    assert state["messages"] == []
    assert len(state["invariants"]) == 4


def test_api_rejects_blank_message(client):
    res = client.post("/api/day14/chat", json={"message": "   "})
    assert res.status_code == 422


def test_day14_does_not_touch_day13_or_day7(client):
    # Day 13 has no task yet and Day 14 must not create one.
    assert client.get("/api/day13/state").json()["has_task"] is False
    client.post("/api/day14/chat", json={"message": ALLOWED_REQUEST})
    assert client.get("/api/day13/state").json()["has_task"] is False

    # Day 7 history is untouched by Day 14.
    history = client.get("/api/chat/history").json()
    assert history["count"] == 0
