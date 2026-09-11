"""Deterministic tests for Day 6 — the first (stateless) agent.

The key acceptance criterion is that Day 6 uses the same ``Agent`` abstraction
as Day 7 but does NOT accumulate context between requests. These tests inspect
the exact ``messages`` list handed to the fake LLM.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.llm import DeepSeekLLMClient
from app.agents.manager import AgentManager
from app.agents.repository import InMemoryContextRepository, SQLiteContextRepository
from app.api.routes import get_agent_manager
from app.config import Settings
from app.main import create_app
from app.services.deepseek import DeepSeekAuthenticationError


# ----------------------------------------------------------------------
# pure Agent / manager behaviour
# ----------------------------------------------------------------------
async def test_transient_agent_uses_in_memory_repository(agent_manager):
    agent = agent_manager.create_transient()
    assert isinstance(agent._context_repository, InMemoryContextRepository)
    assert agent.agent_id == "day6-agent"


async def test_transient_agent_does_not_persist_and_is_fresh(agent_manager):
    first = agent_manager.create_transient()
    await first.chat("pervyy zapros")
    assert len(await first.get_history()) == 2  # in-memory only, within the call

    # A new transient agent starts empty (that is what the route does per request)
    second = agent_manager.create_transient()
    assert await second.get_history() == []

    # Day 7 persistent storage was never touched
    assert await agent_manager.repository.load_messages("default") == []


async def test_transient_agent_sends_only_system_and_current_user(
    agent_manager, fake_service, settings
):
    agent = agent_manager.create_transient()
    await agent.chat("privet")
    assert fake_service.generate_calls[0]["messages"] == [
        {"role": "system", "content": settings.system_prompt},
        {"role": "user", "content": "privet"},
    ]


# ----------------------------------------------------------------------
# API behaviour
# ----------------------------------------------------------------------
def test_day6_chat_success_returns_agent_metadata(client):
    response = client.post(
        "/api/day6/agent/chat",
        json={"message": "Explain REST API in one sentence."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Mocked answer from DeepSeek."
    assert body["agent_id"] == "day6-agent"
    assert body["provider"] == "deepseek"
    assert body["model"]
    assert body["stateless"] is True


def test_day6_is_stateless_between_requests(client, fake_service, settings):
    first = client.post(
        "/api/day6/agent/chat",
        json={"message": "Запомни кодовое слово капибара"},
    )
    second = client.post(
        "/api/day6/agent/chat",
        json={"message": "Какое кодовое слово я называл?"},
    )
    assert first.status_code == 200 and second.status_code == 200

    calls = fake_service.generate_calls
    assert len(calls) == 2

    # First request: system + current user only.
    assert [m["content"] for m in calls[0]["messages"]] == [
        settings.system_prompt,
        "Запомни кодовое слово капибара",
    ]
    # Second request MUST NOT contain the previous user/assistant turns.
    assert [m["content"] for m in calls[1]["messages"]] == [
        settings.system_prompt,
        "Какое кодовое слово я называл?",
    ]
    # no trace of the previous run anywhere in the second payload
    assert all(
        "капибара" not in m["content"].lower() for m in calls[1]["messages"]
    )


def test_day6_does_not_write_day7_history(client):
    client.post("/api/day6/agent/chat", json={"message": "hello"})
    history = client.get("/api/chat/history").json()
    assert history["count"] == 0


def test_day6_provider_override_uses_provider_default_model(client, settings):
    response = client.post(
        "/api/day6/agent/chat",
        json={"message": "hi", "provider": "openrouter"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openrouter"
    assert body["model"] == settings.openrouter_model


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "\t\n"},
        {},
    ],
)
def test_day6_rejects_invalid_messages(client, payload):
    response = client.post("/api/day6/agent/chat", json=payload)
    assert response.status_code == 422


def test_day6_validation_causes_zero_llm_calls(client, fake_service):
    client.post("/api/day6/agent/chat", json={"message": ""})
    assert fake_service.generate_calls == []


def test_day6_maps_provider_errors_to_http(client, fake_service):
    fake_service.raise_error = DeepSeekAuthenticationError()
    response = client.post("/api/day6/agent/chat", json={"message": "hello"})
    assert response.status_code == 401
    assert response.json()["detail"]


# ----------------------------------------------------------------------
# metadata endpoint (GET /api/day6/agent) — regression for the UI bug where
# Provider/Model stayed empty until the first chat request.
# ----------------------------------------------------------------------
def test_day6_agent_metadata_defaults(client, settings):
    response = client.get("/api/day6/agent")
    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == "day6-agent"
    assert body["name"] == "Simple Agent (transient)"
    assert body["provider"] == settings.agent_default_provider
    assert body["model"] == settings.deepseek_model  # resolved default model
    assert body["stateless"] is True


def test_day6_metadata_does_not_call_llm(client, fake_service):
    client.get("/api/day6/agent")
    assert fake_service.generate_calls == []


def test_day6_metadata_model_matches_chat_model(client, fake_service):
    meta = client.get("/api/day6/agent").json()
    client.post("/api/day6/agent/chat", json={"message": "hi"})
    assert fake_service.generate_calls[-1]["model"] == meta["model"]


def test_day6_metadata_follows_agent_default_provider(tmp_path, fake_service):
    """Metadata must come from config resolution, not hardcoded values."""
    settings = Settings(
        deepseek_api_key="test-key",
        agent_default_provider="openrouter",
        openrouter_model="openai/gpt-4o-mini",
    )
    llm = DeepSeekLLMClient(fake_service)
    manager = AgentManager(
        settings,
        SQLiteContextRepository(tmp_path / "agents.db"),
        {"deepseek": llm, "openrouter": llm},
    )
    app = create_app(settings=settings)
    app.dependency_overrides[get_agent_manager] = lambda: manager
    with TestClient(app) as c:
        body = c.get("/api/day6/agent").json()
    assert body["provider"] == "openrouter"
    assert body["model"] == "openai/gpt-4o-mini"
