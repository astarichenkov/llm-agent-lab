"""Deterministic tests for the persistent Agent layer (Day 7).

No test here touches a real provider: the LLM is a ``FakeLLMClient`` and the
context is stored in a temporary SQLite database.
"""
from __future__ import annotations

from collections import deque

import pytest
from fastapi.testclient import TestClient

from app.agents.agent import Agent
from app.agents.config import AgentConfig
from app.agents.llm import LLMClient, LLMResult
from app.agents.manager import AgentManager
from app.agents.repository import SQLiteContextRepository
from app.api.routes import get_agent_manager
from app.config import Settings
from app.main import create_app


class FakeLLMClient(LLMClient):
    """Deterministic stand-in that records every request it receives."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self._replies = deque(replies or [])

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: str | None = None,
        thinking: bool = False,
    ) -> LLMResult:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                "thinking": thinking,
            }
        )
        reply = self._replies.popleft() if self._replies else f"reply-{len(self.calls)}"
        return LLMResult(
            content=reply,
            finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )


def make_manager(settings: Settings, db_path, llm: FakeLLMClient) -> AgentManager:
    repository = SQLiteContextRepository(db_path)
    return AgentManager(
        settings,
        repository,
        {"deepseek": llm, "openrouter": llm},
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        deepseek_api_key="test-key",
        system_prompt="You are a test assistant.",
        deepseek_model="deepseek-test-model",
    )


# ----------------------------------------------------------------------
# persistence / restart
# ----------------------------------------------------------------------
async def test_restart_restores_history_and_uses_it_in_next_request(settings, tmp_path):
    db_path = tmp_path / "agents.db"

    # --- first process ---
    llm1 = FakeLLMClient(replies=["answer-1", "answer-2"])
    manager1 = make_manager(settings, db_path, llm1)
    agent1 = await manager1.get_or_create("test-agent")
    await agent1.chat("pervoe soobshenie")
    await agent1.chat("vtoroe soobshenie")

    stored = await manager1.repository.load_messages("test-agent")
    assert [m.role for m in stored] == ["user", "assistant", "user", "assistant"]
    assert [m.content for m in stored] == [
        "pervoe soobshenie",
        "answer-1",
        "vtoroe soobshenie",
        "answer-2",
    ]

    # --- simulate a restart: brand new manager + repository + LLM ---
    llm2 = FakeLLMClient(replies=["answer-3"])
    manager2 = make_manager(settings, db_path, llm2)
    agent2 = await manager2.get_or_create("test-agent")

    restored = await agent2.get_history()
    assert [m.content for m in restored] == [
        "pervoe soobshenie",
        "answer-1",
        "vtoroe soobshenie",
        "answer-2",
    ]

    await agent2.chat("tretye soobshenie")

    # The LLM request MUST contain the previous run's messages.
    sent = llm2.calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": "You are a test assistant."}
    assert [m["content"] for m in sent[1:]] == [
        "pervoe soobshenie",
        "answer-1",
        "vtoroe soobshenie",
        "answer-2",
        "tretye soobshenie",
    ]
    assert llm2.calls[0]["model"] == "deepseek-test-model"


async def test_system_prompt_is_not_stored_as_dialog(settings, tmp_path):
    llm = FakeLLMClient()
    manager = make_manager(settings, tmp_path / "agents.db", llm)
    agent = await manager.get_or_create("sys-agent")

    await agent.chat("hello")
    history = await agent.get_history()

    assert [m.role for m in history] == ["user", "assistant"]
    assert all("test assistant" not in m.content for m in history)
    # ...but it IS sent to the model.
    assert llm.calls[0]["messages"][0]["role"] == "system"


async def test_clear_history_does_not_forget_agent_config(settings, tmp_path):
    llm = FakeLLMClient()
    manager = make_manager(settings, tmp_path / "agents.db", llm)
    agent = await manager.create_agent(
        "clear-me", AgentConfig(name="Clear Me", system_prompt="custom")
    )
    await agent.chat("data")

    await agent.clear_history()
    assert await agent.get_history() == []
    assert await manager.repository.load_messages("clear-me") == []

    # config still available after a "restart"
    manager2 = make_manager(settings, tmp_path / "agents.db", FakeLLMClient())
    restored = await manager2.get_or_create("clear-me")
    assert restored.config.name == "Clear Me"
    assert restored.config.system_prompt == "custom"
    assert await restored.get_history() == []


# ----------------------------------------------------------------------
# multi-agent isolation
# ----------------------------------------------------------------------
async def test_multiple_agents_have_independent_histories(settings, tmp_path):
    db_path = tmp_path / "agents.db"
    llm = FakeLLMClient(replies=["a-reply", "b-reply", "a-reply-2", "b-reply-2"])
    manager = make_manager(settings, db_path, llm)

    agent_a = await manager.get_or_create(
        "agent-a", AgentConfig(name="Agent A", system_prompt="A prompt")
    )
    agent_b = await manager.get_or_create(
        "agent-b", AgentConfig(name="Agent B", system_prompt="B prompt")
    )
    assert agent_a is not agent_b
    assert agent_a.agent_id == "agent-a"
    assert agent_b.agent_id == "agent-b"

    await agent_a.chat("A message 1")
    await agent_b.chat("B message 1")

    # restart and reload both
    manager2 = make_manager(settings, db_path, FakeLLMClient(replies=["x", "y"]))
    a2 = await manager2.get_or_create("agent-a")
    b2 = await manager2.get_or_create("agent-b")

    a_hist = [m.content for m in await a2.get_history()]
    b_hist = [m.content for m in await b2.get_history()]
    assert a_hist == ["A message 1", "a-reply"]
    assert b_hist == ["B message 1", "b-reply"]
    assert "B message 1" not in a_hist
    assert "A message 1" not in b_hist

    # a third turn for A does not pull B's context
    await a2.chat("A message 2")
    sent = manager2._llm_clients["deepseek"].calls[-1]["messages"]
    contents = [m["content"] for m in sent]
    assert "B message 1" not in contents
    assert contents == [
        "A prompt",
        "A message 1",
        "a-reply",
        "A message 2",
    ]


async def test_agent_config_is_persisted_and_restored(settings, tmp_path):
    db_path = tmp_path / "agents.db"
    manager = make_manager(settings, db_path, FakeLLMClient())
    await manager.create_agent(
        "configured",
        AgentConfig(
            name="Configured Agent",
            provider="openrouter",
            model="openai/gpt-4o-mini",
            system_prompt="Configured prompt",
            temperature=0.2,
            max_tokens=321,
        ),
    )

    manager2 = make_manager(settings, db_path, FakeLLMClient())
    restored = await manager2.get_or_create("configured")
    assert restored.config.provider == "openrouter"
    assert restored.config.model == "openai/gpt-4o-mini"
    assert restored.config.system_prompt == "Configured prompt"
    assert restored.config.temperature == 0.2
    assert restored.config.max_tokens == 321


async def test_get_or_create_returns_same_live_instance(settings, tmp_path):
    manager = make_manager(settings, tmp_path / "agents.db", FakeLLMClient())
    first = await manager.get_or_create("same")
    second = await manager.get_or_create("same")
    assert first is second


async def test_default_config_is_created_for_unknown_agent(settings, tmp_path):
    manager = make_manager(settings, tmp_path / "agents.db", FakeLLMClient())
    agent = await manager.get_or_create("brand-new")
    assert agent.config.provider == "deepseek"
    assert agent.config.resolved_model(settings) == "deepseek-test-model"
    assert agent.config.system_prompt == settings.system_prompt
    assert await manager.repository.load_config("brand-new") is not None


async def test_list_agent_ids_includes_stored_agents(settings, tmp_path):
    db_path = tmp_path / "agents.db"
    manager = make_manager(settings, db_path, FakeLLMClient())
    await manager.get_or_create("one")
    await manager.get_or_create("two")

    manager2 = make_manager(settings, db_path, FakeLLMClient())
    assert await manager2.list_agent_ids() == ["one", "two"]


# ----------------------------------------------------------------------
# API integration
# ----------------------------------------------------------------------
def test_api_chat_persists_and_returns_history(client, agent_manager):
    first = client.post("/api/chat", json={"message": "hello"})
    assert first.status_code == 200
    assert first.json()["answer"] == "Mocked answer from DeepSeek."

    second = client.post("/api/chat", json={"message": "again"})
    assert second.status_code == 200

    history = client.get("/api/chat/history")
    assert history.status_code == 200
    body = history.json()
    assert body["count"] == 4
    assert [m["role"] for m in body["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert body["agent"]["agent_id"] == "default"

    cleared = client.delete("/api/chat/history")
    assert cleared.status_code == 200
    assert client.get("/api/chat/history").json()["count"] == 0


def test_api_llm_request_contains_restored_context(client, fake_service, agent_manager):
    client.post("/api/chat", json={"message": "remember capybara"})
    client.post("/api/chat", json={"message": "what was it?"})

    last_call = fake_service.generate_calls[-1]
    contents = [m["content"] for m in last_call["messages"]]
    assert "remember capybara" in contents
    assert "Mocked answer from DeepSeek." in contents
    assert contents[-1] == "what was it?"
    assert contents[0] == agent_manager.settings.system_prompt


def test_api_multi_agent_endpoints_are_isolated(client):
    r1 = client.post("/api/agents/agent-a/chat", json={"message": "alpha"})
    r2 = client.post("/api/agents/agent-b/chat", json={"message": "beta"})
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["agent_id"] == "agent-a"

    hist_a = client.get("/api/agents/agent-a/history").json()
    hist_b = client.get("/api/agents/agent-b/history").json()
    assert [m["content"] for m in hist_a["messages"]] == [
        "alpha",
        "Mocked answer from DeepSeek.",
    ]
    assert [m["content"] for m in hist_b["messages"]] == [
        "beta",
        "Mocked answer from DeepSeek.",
    ]

    assert client.delete("/api/agents/agent-a/history").status_code == 200
    assert client.get("/api/agents/agent-a/history").json()["count"] == 0
    assert client.get("/api/agents/agent-b/history").json()["count"] == 2


def test_api_agents_list(client):
    client.post("/api/agents/list-a/chat", json={"message": "x"})
    client.post("/api/agents/list-b/chat", json={"message": "y"})
    body = client.get("/api/agents").json()
    ids = {a["agent_id"] for a in body["agents"]}
    assert {"list-a", "list-b"} <= ids


def test_api_history_survives_new_manager_process(settings, tmp_path, fake_service):
    """Full restart simulation at the HTTP layer: a new app/manager over the
    same SQLite file restores the dialog and feeds it to the next LLM call."""
    db_path = tmp_path / "agents.db"

    llm1 = FakeLLMClient(replies=["noted"])
    manager1 = make_manager(settings, db_path, llm1)
    app1 = create_app(settings=settings)
    app1.dependency_overrides[get_agent_manager] = lambda: manager1
    with TestClient(app1) as c1:
        assert c1.post("/api/chat", json={"message": "code word: capybara"}).status_code == 200
        assert c1.get("/api/chat/history").json()["count"] == 2

    # new process: new app, new manager, same DB
    llm2 = FakeLLMClient(replies=["capybara"])
    manager2 = make_manager(settings, db_path, llm2)
    app2 = create_app(settings=settings)
    app2.dependency_overrides[get_agent_manager] = lambda: manager2
    with TestClient(app2) as c2:
        history = c2.get("/api/chat/history").json()
        assert history["count"] == 2
        assert history["messages"][0]["content"] == "code word: capybara"

        assert c2.post("/api/chat", json={"message": "what was it?"}).status_code == 200
        sent = llm2.calls[0]["messages"]
        assert [m["content"] for m in sent] == [
            settings.system_prompt,
            "code word: capybara",
            "noted",
            "what was it?",
        ]


async def test_agent_chat_rejects_empty_message(settings, tmp_path):
    manager = make_manager(settings, tmp_path / "agents.db", FakeLLMClient())
    agent = await manager.get_or_create("empty")
    with pytest.raises(ValueError):
        await agent.chat("   ")
