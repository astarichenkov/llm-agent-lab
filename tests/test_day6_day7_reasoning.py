"""Day 6 / Day 7 — reasoning must be OFF in the ACTUAL provider request.

These tests do not stop at ``AgentConfig.thinking``: they drive the real
``DeepSeekService`` through the ``DeepSeekLLMClient`` adapter and inspect the
kwargs that reach ``chat.completions.create``. They also cover the persisted
``default`` agent config migration (old ``max_tokens``/missing ``thinking``).

Days 2-5 are covered by their own suites; this file additionally asserts the
Agent layer is the *only* place the override is introduced.
"""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from app.agents.config import AgentConfig, CONFIG_VERSION, DEFAULT_MAX_TOKENS
from app.agents.llm import DeepSeekLLMClient
from app.agents.manager import AgentManager
from app.agents.repository import SQLiteContextRepository
from app.config import Settings
from app.services.deepseek import DeepSeekOutputLimitError, DeepSeekService

THINKING_DISABLED = {"thinking": {"type": "disabled"}}


# ----------------------------------------------------------------------
# Fake AsyncOpenAI that records chat.completions.create kwargs
# ----------------------------------------------------------------------
class _FakeMessage:
    content = "recorded answer"


class _FakeChoice:
    def __init__(self) -> None:
        self.message = _FakeMessage()
        self.finish_reason = "stop"


class _FakeUsage:
    prompt_tokens = 3
    completion_tokens = 4
    total_tokens = 7


class _FakeResponse:
    def __init__(self) -> None:
        self.choices = [_FakeChoice()]
        self.usage = _FakeUsage()


class _RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse()


class _RecordingAsyncOpenAI:
    instances: list["_RecordingAsyncOpenAI"] = []

    def __init__(self, **kwargs) -> None:
        self.constructor_kwargs = kwargs
        self.completions = _RecordingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        _RecordingAsyncOpenAI.instances.append(self)


def _real_deepseek_manager(monkeypatch, settings: Settings, db_path) -> AgentManager:
    """AgentManager whose DeepSeek adapter wraps a REAL DeepSeekService whose
    underlying AsyncOpenAI client is a recording fake (no network)."""
    _RecordingAsyncOpenAI.instances = []
    monkeypatch.setattr("app.services.deepseek.AsyncOpenAI", _RecordingAsyncOpenAI)
    service = DeepSeekService(settings)
    manager = AgentManager(
        settings,
        SQLiteContextRepository(db_path),
        {"deepseek": DeepSeekLLMClient(service)},
    )
    return manager


def _last_create_kwargs() -> dict:
    assert _RecordingAsyncOpenAI.instances, "AsyncOpenAI was never constructed"
    calls = _RecordingAsyncOpenAI.instances[-1].completions.calls
    assert calls, "chat.completions.create was never called"
    return calls[-1]


# ----------------------------------------------------------------------
# Day 7 — persistent default agent
# ----------------------------------------------------------------------
async def test_day7_default_agent_sends_max_tokens_and_thinking_disabled(
    monkeypatch, settings, tmp_path
):
    manager = _real_deepseek_manager(monkeypatch, settings, tmp_path / "agents.db")
    agent = await manager.get_or_create("default")
    await agent.chat("remember the code word: capybara")

    call = _last_create_kwargs()
    assert call["max_tokens"] == DEFAULT_MAX_TOKENS == 8192
    assert call["extra_body"] == THINKING_DISABLED
    assert call["model"] == settings.deepseek_model


async def test_day7_context_is_preserved_and_every_call_disables_thinking(
    monkeypatch, settings, tmp_path
):
    manager = _real_deepseek_manager(monkeypatch, settings, tmp_path / "agents.db")
    agent = await manager.get_or_create("default")
    await agent.chat("first")
    await agent.chat("second")

    calls = _RecordingAsyncOpenAI.instances[-1].completions.calls
    assert len(calls) == 2
    for call in calls:
        assert call["max_tokens"] == 8192
        assert call["extra_body"] == THINKING_DISABLED
    # restored context is actually part of the second request
    second_contents = [m["content"] for m in calls[1]["messages"]]
    assert "first" in second_contents
    assert second_contents[-1] == "second"


# ----------------------------------------------------------------------
# Day 6 — transient agent
# ----------------------------------------------------------------------
async def test_day6_transient_agent_sends_max_tokens_and_thinking_disabled(
    monkeypatch, settings, tmp_path
):
    manager = _real_deepseek_manager(monkeypatch, settings, tmp_path / "agents.db")
    agent = manager.create_transient()
    await agent.chat("hello")

    call = _last_create_kwargs()
    assert call["max_tokens"] == 8192
    assert call["extra_body"] == THINKING_DISABLED


def test_day6_agent_config_defaults(settings):
    config = AgentConfig.default(settings)
    assert config.max_tokens == 8192
    assert config.thinking is False
    assert config.config_version == CONFIG_VERSION


# ----------------------------------------------------------------------
# API level: the agent endpoints carry the new parameters to the adapter
# ----------------------------------------------------------------------
def test_api_day6_records_max_tokens_and_thinking(client, fake_service):
    response = client.post("/api/day6/agent/chat", json={"message": "hi"})
    assert response.status_code == 200
    call = fake_service.generate_calls[-1]
    assert call["max_tokens"] == 8192
    assert call["thinking"] is False


def test_api_day7_records_max_tokens_and_thinking(client, fake_service):
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 200
    call = fake_service.generate_calls[-1]
    assert call["max_tokens"] == 8192
    assert call["thinking"] is False


def test_api_day6_metadata_exposes_actual_config(client):
    body = client.get("/api/day6/agent").json()
    assert body["max_tokens"] == 8192
    assert body["thinking"] is False


def test_api_day7_history_exposes_actual_config(client):
    body = client.get("/api/chat/history").json()
    assert body["agent"]["max_tokens"] == 8192
    assert body["agent"]["thinking"] is False


# ----------------------------------------------------------------------
# Regression for the real bug: a normal short answer must NOT be turned into
# the max_tokens error. The raw provider says finish_reason="stop".
# ----------------------------------------------------------------------
def test_api_day7_arithmetic_stop_is_not_max_tokens_error(client, fake_service):
    fake_service.answer = "48"
    response = client.post("/api/chat", json={"message": "46+2 ="})
    assert response.status_code == 200
    assert response.json()["answer"] == "48"


def test_api_day7_real_length_is_still_reported_as_output_limit(client, fake_service):
    """We must NOT simply delete the length handling: a genuine length still
    surfaces as the user-facing max_tokens error (HTTP 502)."""
    fake_service.raise_error = DeepSeekOutputLimitError()
    response = client.post("/api/chat", json={"message": "46+2 ="})
    assert response.status_code == 502
    assert "max_tokens" in response.json()["detail"]


# ----------------------------------------------------------------------
# Failed turn must not leave an orphaned user message in persistent history
# ----------------------------------------------------------------------
def test_failed_llm_call_does_not_persist_user_turn(client, fake_service):
    fake_service.raise_error = DeepSeekOutputLimitError()
    assert client.post("/api/chat", json={"message": "46+2 ="}).status_code == 502
    # nothing was committed
    body = client.get("/api/chat/history").json()
    assert body["count"] == 0


def test_repeated_failed_calls_do_not_accumulate_user_messages(client, fake_service):
    fake_service.raise_error = DeepSeekOutputLimitError()
    for _ in range(3):
        client.post("/api/chat", json={"message": "46+2 ="})
    assert client.get("/api/chat/history").json()["count"] == 0


def test_api_day7_sequential_turns_do_not_duplicate_history(client, fake_service):
    fake_service.answer = "48"
    client.post("/api/chat", json={"message": "first"})
    client.post("/api/chat", json={"message": "second"})

    body = client.get("/api/chat/history").json()
    assert [m["content"] for m in body["messages"]] == [
        "first",
        "48",
        "second",
        "48",
    ]
    # The second provider request carries each prior message exactly once.
    second_payload = [m["content"] for m in fake_service.generate_calls[1]["messages"]]
    assert second_payload.count("first") == 1
    assert second_payload.count("second") == 1


# ----------------------------------------------------------------------
# Persisted default config migration (messages must survive)
# ----------------------------------------------------------------------
def _seed_old_default_agent(db_path) -> None:
    """Simulate a database written by the previous release: no ``thinking``
    and no ``config_version`` fields, max_tokens=1024, plus a dialog."""
    old_config = {
        "name": "Default Agent",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "system_prompt": "old prompt",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO agents (agent_id, config_json, updated_at) VALUES (?, ?, ?)",
            ("default", json.dumps(old_config), "2020-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO messages (agent_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            ("default", "user", "old user message", "2020-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO messages (agent_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            ("default", "assistant", "old assistant message", "2020-01-01T00:00:00+00:00"),
        )
        conn.commit()


async def test_persisted_default_config_is_migrated_without_losing_messages(
    settings, tmp_path
):
    db_path = tmp_path / "agents.db"
    repo = SQLiteContextRepository(db_path)
    # initialise the schema through the repository, then seed old data
    await repo.load_config("default")
    _seed_old_default_agent(db_path)

    manager = AgentManager(settings, SQLiteContextRepository(db_path))
    agent = await manager.get_or_create("default")

    # migrated in memory ...
    assert agent.config.max_tokens == 8192
    assert agent.config.thinking is False
    assert agent.config.config_version == CONFIG_VERSION
    # ... and persisted
    restored = await SQLiteContextRepository(db_path).load_config("default")
    assert restored.max_tokens == 8192
    assert restored.thinking is False
    assert restored.config_version == CONFIG_VERSION

    # history was NOT destroyed
    history = await agent.get_history()
    assert [m.content for m in history] == [
        "old user message",
        "old assistant message",
    ]
    assert await SQLiteContextRepository(db_path).load_messages("default") != []


async def test_migration_does_not_touch_non_default_agents(settings, tmp_path):
    db_path = tmp_path / "agents.db"
    repo = SQLiteContextRepository(db_path)
    await repo.save_config(
        "custom", AgentConfig(name="Custom", max_tokens=321, thinking=True)
    )
    manager = AgentManager(settings, SQLiteContextRepository(db_path))
    agent = await manager.get_or_create("custom")
    # an explicitly configured agent keeps its own values
    assert agent.config.max_tokens == 321
    assert agent.config.thinking is True


# ----------------------------------------------------------------------
# Days 2-5 must stay untouched: the generic generate() only overrides when the
# Agent layer explicitly asks for it.
# ----------------------------------------------------------------------
async def test_generate_without_explicit_thinking_sends_no_override(
    monkeypatch, settings
):
    _RecordingAsyncOpenAI.instances = []
    monkeypatch.setattr("app.services.deepseek.AsyncOpenAI", _RecordingAsyncOpenAI)
    service = DeepSeekService(settings)
    await service.generate(
        [{"role": "user", "content": "hi"}], max_tokens=1024
    )
    call = _last_create_kwargs()
    assert "extra_body" not in call
    assert call["max_tokens"] == 1024
