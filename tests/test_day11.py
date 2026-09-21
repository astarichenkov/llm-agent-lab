"""Day 11 — agent memory layers tests.

No real DeepSeek call is made. A deterministic provider distinguishes the
MemoryClassifier call (its system prompt contains "MEMORY MANAGER") from the
main answer call, so every memory decision is scripted and observable.
"""
from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.schemas.day11 import LongTermMemory, WorkingMemory
from app.services.day11.classifier import (
    CLASSIFIER_SYSTEM_PROMPT,
    MemoryClassifier,
    build_classifier_prompt,
    parse_decision,
)
from app.services.day11.service import Day11MemoryService
from app.services.day11.store import (
    LongTermMemoryStore,
    apply_long_term_update,
    apply_working_update,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
class RecordingProvider:
    """Classifier calls and answer calls are separated by a marker."""

    def __init__(
        self,
        classifier_responses=None,
        answer: str = "OK-answer",
        classifier_error: Exception | None = None,
    ) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.classifier_responses = list(classifier_responses or [])
        self.answer = answer
        self.classifier_error = classifier_error
        self.classifier_calls = 0
        self.answer_calls = 0

    async def generate(self, messages, *, model=None, max_tokens=1024, thinking=False):
        self.calls.append([dict(m) for m in messages])
        is_classifier = any(
            "MEMORY MANAGER" in (m.get("content") or "") for m in messages
        )
        if is_classifier:
            self.classifier_calls += 1
            if self.classifier_error is not None:
                raise self.classifier_error
            if self.classifier_responses:
                payload = self.classifier_responses.pop(0)
            else:
                payload = {"nothing_to_save": True}
            content = payload if isinstance(payload, str) else json.dumps(payload)
            return content, "stop", {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            }
        self.answer_calls += 1
        return self.answer, "stop", {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        }

    def last_answer_messages(self) -> list[dict[str, str]]:
        for call in reversed(self.calls):
            if not any("MEMORY MANAGER" in (m.get("content") or "") for m in call):
                return call
        return []


def _settings(tmp_path) -> Settings:
    return Settings(
        deepseek_api_key="test",
        environment="test",
        day11_long_term_path=str(tmp_path / "long_term.json"),
    )


def _service(provider, tmp_path) -> Day11MemoryService:
    settings = _settings(tmp_path)
    return Day11MemoryService(
        settings,
        deepseek=provider,
        long_term_store=LongTermMemoryStore(settings.day11_long_term_path),
    )


WORKING_RESPONSE = {
    "nothing_to_save": False,
    "working_memory": {
        "goal": "Спроектировать сервис бронирования",
        "constraints": ["Не использовать PostgreSQL"],
    },
    "long_term_memory": {},
}

LONG_TERM_RESPONSE = {
    "nothing_to_save": False,
    "working_memory": {},
    "long_term_memory": {"preferred_language": "Python", "answer_style": "concise"},
}

# ONE classifier decision that carries BOTH a working-memory delta and a
# long-term-memory delta at the same time. It must not be treated as an
# either/or choice: both layers are updated in the same turn.
COMBINED_RESPONSE = {
    "nothing_to_save": False,
    "working_memory": {
        "goal": "Спроектировать сервис бронирования",
        "constraints": ["Не использовать PostgreSQL"],
    },
    "long_term_memory": {"preferred_language": "Python", "answer_style": "concise"},
}


# ----------------------------------------------------------------------
# 1) Short-term memory
# ----------------------------------------------------------------------
async def test_short_term_stores_current_session_messages(tmp_path):
    provider = RecordingProvider()
    svc = _service(provider, tmp_path)
    await svc.chat(message="Привет", classify=False)
    state = svc.state()
    assert state.short_term_count == 2
    assert state.short_term[0].role == "user"
    assert state.short_term[0].content == "Привет"
    assert state.short_term[1].role == "assistant"


# ----------------------------------------------------------------------
# 2) Working memory stored separately
# ----------------------------------------------------------------------
async def test_working_memory_stored_separately(tmp_path):
    provider = RecordingProvider(classifier_responses=[WORKING_RESPONSE])
    svc = _service(provider, tmp_path)
    await svc.chat(message="Сейчас проектируем сервис бронирования, PostgreSQL нельзя")
    state = svc.state()
    assert state.working.goal == "Спроектировать сервис бронирования"
    assert "Не использовать PostgreSQL" in state.working.constraints
    # working memory is NOT a copy of the messages
    assert state.short_term_count == 2
    assert all(
        "Спроектировать сервис бронирования" != m.content for m in state.short_term
    )
    # it is returned as its own layer
    assert state.working.count() == 2


# ----------------------------------------------------------------------
# 3) Long-term memory stored separately
# ----------------------------------------------------------------------
async def test_one_decision_applies_working_and_long_term_together(tmp_path):
    """A single classifier decision updates BOTH memory layers at once."""
    provider = RecordingProvider(classifier_responses=[COMBINED_RESPONSE])
    svc = _service(provider, tmp_path)
    response = await svc.chat(
        message="Проектируем сервис бронирования без PostgreSQL; я пишу на Python, отвечай кратко"
    )

    decision = response.last_decision
    assert decision is not None
    assert decision.performed is True
    assert decision.nothing_to_save is False
    # both deltas were parsed from the SAME payload
    assert decision.working_memory.goal == "Спроектировать сервис бронирования"
    assert "Не использовать PostgreSQL" in decision.working_memory.constraints
    assert decision.long_term_memory == {
        "preferred_language": "Python",
        "answer_style": "concise",
    }
    assert decision.summary_lines()  # reports updates, not nothing_to_save

    # ... and both were applied within the same turn/decision.
    state = svc.state()
    assert state.working.goal == "Спроектировать сервис бронирования"
    assert "Не использовать PostgreSQL" in state.working.constraints
    assert state.long_term.entries["preferred_language"] == "Python"
    assert state.long_term.entries["answer_style"] == "concise"
    # long-term was persisted immediately, once, for that decision.
    assert (tmp_path / "long_term.json").exists()
    assert provider.classifier_calls == 1


async def test_long_term_memory_stored_separately(tmp_path):
    provider = RecordingProvider(classifier_responses=[LONG_TERM_RESPONSE])
    svc = _service(provider, tmp_path)
    await svc.chat(message="Я обычно пишу на Python, отвечай кратко")
    state = svc.state()
    assert state.long_term.entries["preferred_language"] == "Python"
    assert state.long_term.entries["answer_style"] == "concise"
    assert state.working.is_empty()
    # persisted to the JSON file immediately
    assert (tmp_path / "long_term.json").exists()


# ----------------------------------------------------------------------
# 4-5) Independent clears
# ----------------------------------------------------------------------
async def test_clear_short_term_does_not_clear_long_term(tmp_path):
    provider = RecordingProvider(classifier_responses=[LONG_TERM_RESPONSE])
    svc = _service(provider, tmp_path)
    await svc.chat(message="Python, кратко")
    before = svc.state().long_term.count()
    svc.reset_short_term()
    state = svc.state()
    assert state.short_term_count == 0
    assert state.long_term.count() == before
    assert state.long_term.entries["preferred_language"] == "Python"


async def test_clear_working_does_not_clear_long_term(tmp_path):
    provider = RecordingProvider(classifier_responses=[WORKING_RESPONSE, LONG_TERM_RESPONSE])
    svc = _service(provider, tmp_path)
    await svc.chat(message="задача и ограничение")
    await svc.chat(message="Python, кратко")
    assert not svc.state().working.is_empty()
    svc.reset_working()
    state = svc.state()
    assert state.working.is_empty()
    assert state.long_term.entries["preferred_language"] == "Python"


# ----------------------------------------------------------------------
# 6) New session does not inherit short-term memory
# ----------------------------------------------------------------------
async def test_new_session_does_not_get_old_short_term(tmp_path):
    provider = RecordingProvider()
    svc = _service(provider, tmp_path)
    await svc.chat(message="secret-in-old-session", classify=False)
    old_id = svc.state().session_id
    state = svc.new_session()
    assert state.session_id != old_id
    assert state.short_term_count == 0
    assert state.working.is_empty()


# ----------------------------------------------------------------------
# 7) Long-term memory survives a new session (and a reload)
# ----------------------------------------------------------------------
async def test_long_term_survives_new_session_and_reload(tmp_path):
    provider = RecordingProvider(classifier_responses=[LONG_TERM_RESPONSE])
    svc = _service(provider, tmp_path)
    await svc.chat(message="Python, кратко")
    svc.new_session()
    assert svc.state().long_term.entries["preferred_language"] == "Python"

    # A brand-new service instance reads the SAME file: long-term persists.
    reloaded = _service(RecordingProvider(), tmp_path)
    assert reloaded.state().long_term.entries["preferred_language"] == "Python"
    # ... while short-term is empty in the new process/session.
    assert reloaded.state().short_term_count == 0


# ----------------------------------------------------------------------
# 8-10) MemoryClassifier decisions
# ----------------------------------------------------------------------
async def test_classifier_returns_working_update():
    provider = RecordingProvider(classifier_responses=[WORKING_RESPONSE])
    classifier = MemoryClassifier()
    decision, usage = await classifier.classify(
        user_message="Сейчас проектируем сервис бронирования, PostgreSQL нельзя",
        working=WorkingMemory(),
        long_term=LongTermMemory(),
        provider=provider,
        model="m",
    )
    assert decision.performed is True
    assert decision.nothing_to_save is False
    assert decision.working_memory.goal == "Спроектировать сервис бронирования"
    assert "Не использовать PostgreSQL" in decision.working_memory.constraints
    assert usage["total_tokens"] == 5


async def test_classifier_returns_long_term_update():
    provider = RecordingProvider(classifier_responses=[LONG_TERM_RESPONSE])
    classifier = MemoryClassifier()
    decision, _ = await classifier.classify(
        user_message="Я обычно пишу на Python",
        working=WorkingMemory(),
        long_term=LongTermMemory(),
        provider=provider,
        model="m",
    )
    assert decision.long_term_memory == {
        "preferred_language": "Python",
        "answer_style": "concise",
    }


async def test_classifier_returns_nothing_to_save():
    provider = RecordingProvider(
        classifier_responses=[{"nothing_to_save": True}]
    )
    classifier = MemoryClassifier()
    decision, _ = await classifier.classify(
        user_message="Спасибо",
        working=WorkingMemory(),
        long_term=LongTermMemory(entries={"preferred_language": "Python"}),
        provider=provider,
        model="m",
    )
    assert decision.performed is True
    assert decision.nothing_to_save is True
    assert decision.working_memory.is_empty()
    assert decision.long_term_memory == {}


async def test_classifier_returns_working_and_long_term_update_in_one_decision():
    provider = RecordingProvider(classifier_responses=[COMBINED_RESPONSE])
    classifier = MemoryClassifier()
    decision, _ = await classifier.classify(
        user_message="Проектируем сервис бронирования без PostgreSQL; Python, кратко",
        working=WorkingMemory(),
        long_term=LongTermMemory(),
        provider=provider,
        model="m",
    )
    assert decision.performed is True
    assert decision.nothing_to_save is False
    assert decision.working_memory.goal == "Спроектировать сервис бронирования"
    assert "Не использовать PostgreSQL" in decision.working_memory.constraints
    assert decision.long_term_memory["preferred_language"] == "Python"
    assert decision.long_term_memory["answer_style"] == "concise"
    assert decision.has_updates() is True
    assert provider.classifier_calls == 1


def test_parse_decision_keeps_both_updates_with_nothing_to_save_false():
    decision = parse_decision(json.dumps(COMBINED_RESPONSE))
    assert decision.nothing_to_save is False
    assert decision.working_memory.goal == "Спроектировать сервис бронирования"
    assert "Не использовать PostgreSQL" in decision.working_memory.constraints
    assert decision.long_term_memory == {
        "preferred_language": "Python",
        "answer_style": "concise",
    }


def test_parse_decision_forces_nothing_to_save_when_empty():
    decision = parse_decision('{"nothing_to_save": false, "working_memory": {}}')
    assert decision.nothing_to_save is True
    assert not decision.has_updates()


def test_parse_decision_tolerates_code_fences():
    raw = '```json\n{"long_term_memory": {"answer_style": "concise"}}\n```'
    decision = parse_decision(raw)
    assert decision.long_term_memory["answer_style"] == "concise"


def test_build_classifier_prompt_contains_state_and_message():
    payload = build_classifier_prompt(
        WorkingMemory(goal="G"),
        LongTermMemory(entries={"preferred_language": "Python"}),
        "NEW MESSAGE",
    )
    assert payload[0]["content"] == CLASSIFIER_SYSTEM_PROMPT
    assert "CURRENT WORKING MEMORY" in payload[1]["content"]
    assert "preferred_language" in payload[1]["content"]
    assert "NEW MESSAGE" in payload[1]["content"]


# ----------------------------------------------------------------------
# 11) Classifier error does not break the main chat
# ----------------------------------------------------------------------
async def test_classifier_error_does_not_break_chat(tmp_path):
    provider = RecordingProvider(
        answer="Основной ответ",
        classifier_error=RuntimeError("classifier down"),
    )
    svc = _service(provider, tmp_path)
    response = await svc.chat(message="Запомни важное требование")
    assert response.answer == "Основной ответ"
    assert response.classifier_error
    # the turn is still in short-term memory
    assert response.memory.short_term_count == 2
    # working / long-term were NOT polluted
    assert response.memory.working.is_empty()
    assert response.memory.long_term.is_empty()
    # the main answer call still happened
    assert provider.answer_calls == 1


async def test_classifier_invalid_json_does_not_break_chat(tmp_path):
    provider = RecordingProvider(
        classifier_responses=["not json at all"], answer="fine"
    )
    svc = _service(provider, tmp_path)
    response = await svc.chat(message="hello")
    assert response.answer == "fine"
    assert response.classifier_error


# ----------------------------------------------------------------------
# 12) Context Builder adds working / long-term memory to the request
# ----------------------------------------------------------------------
async def test_context_builder_includes_memory_blocks(tmp_path):
    provider = RecordingProvider(answer="A")
    svc = _service(provider, tmp_path)
    svc.set_working(
        WorkingMemory(
            goal="Спроектировать сервис бронирования",
            constraints=["Не использовать PostgreSQL"],
        )
    )
    svc.set_long_term(
        LongTermMemory(entries={"preferred_language": "Python", "answer_style": "concise"})
    )
    response = await svc.chat(message="Предложи архитектуру", classify=False)

    assert response.context.long_term_included is True
    assert response.context.working_included is True
    assert "long_term" in response.context.included_layers
    assert "working" in response.context.included_layers

    sent = provider.last_answer_messages()
    joined = "\n".join(m["content"] for m in sent)
    assert "LONG-TERM MEMORY" in joined
    assert "preferred_language = Python" in joined
    assert "WORKING MEMORY" in joined
    assert "Не использовать PostgreSQL" in joined
    # the current user message appears exactly once
    assert [m["content"] for m in sent].count("Предложи архитектуру") == 1


async def test_disabled_memory_layers_are_not_included(tmp_path):
    provider = RecordingProvider(answer="A")
    svc = _service(provider, tmp_path)
    svc.set_working(WorkingMemory(goal="G", constraints=["C"]))
    svc.set_long_term(LongTermMemory(entries={"preferred_language": "Python"}))
    response = await svc.chat(message="x", classify=False, use_memory=False)
    assert response.context.long_term_included is False
    assert response.context.working_included is False
    joined = "\n".join(m["content"] for m in provider.last_answer_messages())
    assert "LONG-TERM MEMORY" not in joined
    assert "WORKING MEMORY" not in joined


# ----------------------------------------------------------------------
# 13-14) Sliding Window + Memory Layers together
# ----------------------------------------------------------------------
async def test_sliding_window_and_memory_layers_work_together(tmp_path):
    provider = RecordingProvider(answer="A")
    svc = _service(provider, tmp_path)
    history = []
    for i in range(6):
        history.append({"role": "user", "content": f"u{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    svc.seed_short_term(history)
    response = await svc.chat(message="final", window_size=4, classify=False)
    assert response.context.sent_short_term_messages == 4
    assert response.context.dropped_messages == 8
    assert response.context.total_short_term_messages == 14  # +user +assistant
    assert "short_term" in response.context.included_layers


async def test_fact_dropped_by_window_still_influences_prompt(tmp_path):
    provider = RecordingProvider(answer="A")
    svc = _service(provider, tmp_path)
    opening = "PostgreSQL в этой задаче использовать нельзя."
    history = [{"role": "user", "content": opening}]
    for i in range(8):
        history.append({"role": "user", "content": f"filler user {i}"})
        history.append({"role": "assistant", "content": f"filler answer {i}"})
    svc.seed_short_term(history)
    # The constraint survived the raw message in WORKING memory.
    svc.set_working(WorkingMemory(constraints=["Не использовать PostgreSQL"]))

    response = await svc.chat(
        message="Предложи архитектуру хранения данных",
        window_size=4,
        classify=False,
    )

    # 1) the opening message is NOT in the short-term window any more
    sent_contents = [m.content for m in response.context.sent_preview]
    assert opening not in sent_contents
    dropped_contents = [m.content for m in response.context.dropped_preview]
    assert opening in dropped_contents

    # 2) ...but the constraint is STILL in the answer prompt via working memory
    joined = "\n".join(m["content"] for m in provider.last_answer_messages())
    assert opening not in joined
    assert "Не использовать PostgreSQL" in joined


# ----------------------------------------------------------------------
# merge helpers + explicit clear of long-term
# ----------------------------------------------------------------------
def test_apply_updates_merge_and_dedupe():
    working = apply_working_update(
        WorkingMemory(goal="old", constraints=["A"]),
        WorkingMemory(goal="new", constraints=["a", "B"], data={"db": "MySQL"}),
    )
    assert working.goal == "new"
    assert working.constraints == ["A", "B"]
    assert working.data == {"db": "MySQL"}

    long_term = apply_long_term_update(
        LongTermMemory(entries={"answer_style": "verbose"}),
        {"answer_style": "concise", "preferred_language": "Python"},
    )
    assert long_term.entries == {
        "answer_style": "concise",
        "preferred_language": "Python",
    }


async def test_clear_long_term_keeps_short_term_and_working(tmp_path):
    provider = RecordingProvider(classifier_responses=[WORKING_RESPONSE, LONG_TERM_RESPONSE])
    svc = _service(provider, tmp_path)
    await svc.chat(message="task")
    await svc.chat(message="Python, кратко")
    assert svc.state().short_term_count == 4
    svc.reset_long_term()
    state = svc.state()
    assert state.long_term.is_empty()
    assert state.short_term_count == 4
    assert not state.working.is_empty()
    assert not (tmp_path / "long_term.json").exists()


# ----------------------------------------------------------------------
# Observability: the four save-routing outcomes are explicit
# ----------------------------------------------------------------------
async def test_observable_save_routing_covers_all_four_cases(tmp_path):
    """From the existing response you can tell exactly where a turn was saved.

    Short-term is always deterministic; working / long-term are determined by
    the MemoryClassifier decision already returned in ``last_decision``.
    """
    cases = [
        ({"nothing_to_save": True}, "short-term-only"),
        (WORKING_RESPONSE, "short-term+working"),
        (LONG_TERM_RESPONSE, "short-term+long-term"),
        (COMBINED_RESPONSE, "short-term+working+long-term"),
    ]
    for payload, expected in cases:
        provider = RecordingProvider(classifier_responses=[payload])
        svc = _service(provider, tmp_path)
        response = await svc.chat(message="observable routing")

        decision = response.last_decision
        assert decision is not None
        # Short-term is always written for the current turn.
        assert response.memory.short_term_count == 2

        working_saved = (
            decision.performed
            and not decision.nothing_to_save
            and decision.working_memory.count() > 0
        )
        long_term_saved = (
            decision.performed
            and not decision.nothing_to_save
            and bool(decision.long_term_memory)
        )

        if expected == "short-term-only":
            assert working_saved is False
            assert long_term_saved is False
        elif expected == "short-term+working":
            assert working_saved is True
            assert long_term_saved is False
        elif expected == "short-term+long-term":
            assert working_saved is False
            assert long_term_saved is True
        else:
            assert working_saved is True
            assert long_term_saved is True


# ----------------------------------------------------------------------
# API level
# ----------------------------------------------------------------------
def test_api_day11_state(client):
    resp = client.get("/api/day11/state")
    assert resp.status_code == 200
    body = resp.json()
    assert "model" in body
    assert body["memory"]["short_term_count"] == 0


def test_api_day11_chat_graceful_classifier(client):
    # The default fake provider returns prose for the classifier call, so the
    # classifier fails to parse — the chat must still succeed.
    resp = client.post(
        "/api/day11/chat",
        json={"message": "hello", "classify": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Mocked answer from DeepSeek."
    assert body["memory"]["short_term_count"] == 2
    assert body["classifier_error"]


def test_api_day11_clears_are_independent(client):
    # seed demo memory (no provider call) then clear layers one by one
    seeded = client.post("/api/day11/memory/seed")
    assert seeded.status_code == 200
    memory = seeded.json()["memory"]
    assert memory["short_term_count"] > 0
    assert memory["working"]["constraints"]
    assert memory["long_term"]["entries"]

    cleared_st = client.delete("/api/day11/memory/short-term").json()["memory"]
    assert cleared_st["short_term_count"] == 0
    assert cleared_st["working"]["constraints"]
    assert cleared_st["long_term"]["entries"]

    cleared_working = client.delete("/api/day11/memory/working").json()["memory"]
    assert cleared_working["working"] == {
        "goal": None, "constraints": [], "requirements": [], "decisions": [], "data": {}
    }
    assert cleared_working["long_term"]["entries"]

    cleared_lt = client.delete("/api/day11/memory/long-term").json()["memory"]
    assert cleared_lt["long_term"]["entries"] == {}


def test_api_day11_scenario_and_evaluate(client):
    scenario = client.get("/api/day11/scenario")
    assert scenario.status_code == 200
    assert len(scenario.json()["messages"]) >= 5
    assert "Python" in scenario.json()["expected"]

    ev = client.post(
        "/api/day11/evaluate",
        json={"answer": "Python, краткие ответы, сервис бронирования, без PostgreSQL"},
    )
    assert ev.status_code == 200
    assert ev.json()["score"] == ev.json()["total"]


def test_api_day11_does_not_touch_day7_history(client):
    client.post("/api/day11/memory/seed")
    assert client.get("/api/chat/history").json()["count"] == 0


def test_api_old_endpoints_still_work(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/day10/state", params={"strategy": "sliding_window"}).status_code == 200
    assert client.get("/api/day9/diagnostics").status_code == 200
