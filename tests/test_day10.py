"""Day 10 — context-management strategies tests.

No real DeepSeek call is made: strategies are tested both in isolation (pure)
and through a deterministic provider that only "knows" what is present in the
messages it receives. That makes the sliding-window loss vs sticky-facts
retention objectively visible.
"""
from __future__ import annotations

import json
import re

import pytest

from app.api.routes import get_day10_service
from app.config import Settings
from app.services.day10.facts import build_facts_prompt, parse_facts_json
from app.services.day10.models import Conversation
from app.services.day10.scenario import (
    BRANCH_CONTROL_QUESTION,
    EXPECTED_FACTS_LABELS,
    SCENARIO_MESSAGES,
)
from app.services.day10.service import Day10ContextService
from app.services.day10.strategies import (
    BranchingStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
_TOKENS: list[tuple[str, list[str]]] = [
    ("Python", ["python"]),
    ("FastAPI", ["fastapi"]),
    ("PostgreSQL", ["postgresql"]),
    ("JWT", ["jwt"]),
    ("Docker", ["docker"]),
    ("4 недели", ["4 недели"]),
    ("Учёт заявок", ["учёт заявок", "учёта заявок"]),
]


class ScriptedProvider:
    """Fake provider returning a queue of extraction JSONs and a fixed answer."""

    def __init__(self, facts_queue=None, answer: str = "ANSWER") -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.facts_queue = list(facts_queue or [])
        self.answer = answer
        self.raise_on_extract: Exception | None = None

    async def generate(self, messages, *, model=None, max_tokens=1024, thinking=False):
        self.calls.append([dict(m) for m in messages])
        is_extraction = any("CURRENT MEMORY" in (m.get("content") or "") for m in messages)
        if is_extraction:
            if self.raise_on_extract is not None:
                raise self.raise_on_extract
            payload = self.facts_queue.pop(0) if self.facts_queue else {}
            content = payload if isinstance(payload, str) else json.dumps(payload)
            return content, "stop", {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        return self.answer, "stop", {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}


class ContextEchoProvider:
    """Simulates a model that only knows what is present in its input.

    * extraction: merges tokens found in the new user message into memory;
    * answer: lists only the tokens present in the received messages.
    """

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages, *, model=None, max_tokens=1024, thinking=False):
        self.calls.append([dict(m) for m in messages])
        prompt = "\n".join(m.get("content", "") for m in messages)
        low = prompt.casefold()
        if "current memory (json):" in low:
            return self._extract(prompt), "stop", {
                "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8
            }
        # A realistic assistant ACKs most messages without restating every
        # fact; only the final question asks for the collected spec.
        last_user = (messages[-1].get("content") or "").casefold()
        if "итоговое тз" not in last_user:
            return "Принято.", "stop", {
                "prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24
            }
        found = [label for label, patterns in _TOKENS if any(p in low for p in patterns)]
        return "Итоговое ТЗ: " + ", ".join(found), "stop", {
            "prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26
        }

    @staticmethod
    def _extract(prompt: str) -> str:
        match = re.search(
            r"CURRENT MEMORY \(JSON\):\s*(\{.*\})\s*NEW USER MESSAGE:",
            prompt,
            re.DOTALL,
        )
        facts = json.loads(match.group(1)) if match else {}
        new_message = prompt.split("NEW USER MESSAGE:")[-1].casefold()
        decisions = facts.setdefault("decisions", [])
        for label, patterns in _TOKENS:
            if any(p in new_message for p in patterns) and label not in decisions:
                decisions.append(label)
        return json.dumps(facts)


def _history(n: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"u{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out


def _service(fake) -> Day10ContextService:
    settings = Settings(deepseek_api_key="test", environment="test")
    return Day10ContextService(settings, deepseek=fake)


# ----------------------------------------------------------------------
# 1) Sliding Window
# ----------------------------------------------------------------------
def test_sliding_keeps_only_last_n_in_order():
    strat = SlidingWindowStrategy(window_size=6)
    strat.full_history = _history(8)  # 16 messages
    built = strat.build_context(system_prompt="SYS", user_message="NEW")
    assert built.messages[0] == {"role": "system", "content": "SYS"}
    assert built.messages[-1] == {"role": "user", "content": "NEW"}
    sent = built.messages[1:-1]
    assert sent == strat.full_history[-6:]
    assert len(sent) == 6
    # order is preserved
    assert sent == sorted(sent, key=strat.full_history.index)


def test_sliding_drops_old_messages():
    strat = SlidingWindowStrategy(window_size=6)
    strat.full_history = _history(8)
    built = strat.build_context(system_prompt="SYS", user_message="NEW")
    assert built.dropped_messages == strat.full_history[:-6]
    assert built.total_history_messages == 16
    assert built.sent_history_messages == 6
    # the dropped messages are NOT in the payload
    for message in built.dropped_messages:
        assert message not in built.messages


def test_sliding_system_prompt_always_present_and_not_counted():
    strat = SlidingWindowStrategy(window_size=2)
    strat.full_history = _history(3)
    built = strat.build_context(system_prompt="SYSTEM-PROMPT", user_message="NEW")
    assert built.messages[0] == {"role": "system", "content": "SYSTEM-PROMPT"}
    # system prompt is not part of the window count
    assert built.sent_history_messages == 2


def test_sliding_edge_window_of_one():
    strat = SlidingWindowStrategy(window_size=1)
    strat.full_history = _history(5)
    built = strat.build_context(system_prompt="SYS", user_message="NEW")
    assert built.messages[1:-1] == [strat.full_history[-1]]
    assert built.sent_history_messages == 1
    assert built.dropped_messages == strat.full_history[:-1]


def test_sliding_edge_window_larger_than_history():
    strat = SlidingWindowStrategy(window_size=100)
    strat.full_history = _history(3)
    built = strat.build_context(system_prompt="SYS", user_message="NEW")
    assert built.messages[1:-1] == strat.full_history
    assert built.dropped_messages == []
    assert built.total_history_messages == 6


# ----------------------------------------------------------------------
# 2) Sticky Facts
# ----------------------------------------------------------------------
async def test_facts_added_after_extraction():
    provider = ScriptedProvider(facts_queue=[{"goal": "Построить сервис", "constraints": ["Python"]}])
    strat = StickyFactsStrategy(recent_messages_limit=4)
    result = await strat.before_build(
        user_message="Нужно на Python", provider=provider, model="m"
    )
    assert result.performed is True
    assert strat.facts.goal == "Построить сервис"
    assert strat.facts.constraints == ["Python"]


async def test_fact_can_be_updated_and_old_value_replaced():
    provider = ScriptedProvider(
        facts_queue=[
            {"decisions": ["database=PostgreSQL"]},
            {"decisions": ["database=MySQL"]},
        ]
    )
    strat = StickyFactsStrategy()
    await strat.before_build(user_message="База PostgreSQL", provider=provider, model="m")
    assert strat.facts.decisions == ["database=PostgreSQL"]
    await strat.before_build(user_message="Передумали, MySQL", provider=provider, model="m")
    assert strat.facts.decisions == ["database=MySQL"]
    # the old value is gone: only the current one remains
    assert "database=PostgreSQL" not in strat.facts.decisions


async def test_facts_are_passed_into_built_context():
    provider = ScriptedProvider(facts_queue=[{"goal": "Orion", "decisions": ["FastAPI"]}])
    strat = StickyFactsStrategy(recent_messages_limit=2)
    await strat.before_build(user_message="FastAPI", provider=provider, model="m")
    strat.record_turn("u1", "a1")
    strat.record_turn("u2", "a2")
    strat.record_turn("u3", "a3")
    built = strat.build_context(system_prompt="SYS", user_message="NEW")
    assert built.facts_message is not None
    assert "Orion" in built.facts_message["content"]
    assert "FastAPI" in built.facts_message["content"]
    # facts block sits right after the system prompt and before the window
    assert built.messages[1] == built.facts_message
    assert len(built.history_messages) == 2


async def test_facts_recent_messages_limited_to_n():
    provider = ScriptedProvider(facts_queue=[{"goal": "G"}])
    strat = StickyFactsStrategy(recent_messages_limit=3)
    await strat.before_build(user_message="x", provider=provider, model="m")
    strat.full_history = _history(10)  # 20 messages
    built = strat.build_context(system_prompt="SYS", user_message="NEW")
    assert len(built.history_messages) == 3
    assert built.history_messages == strat.full_history[-3:]
    assert len(built.dropped_messages) == 17


async def test_facts_invalid_json_is_handled_gracefully():
    provider = ScriptedProvider(facts_queue=["this is not json"])
    strat = StickyFactsStrategy()
    result = await strat.before_build(user_message="x", provider=provider, model="m")
    assert result.performed is False
    assert result.error is not None
    assert strat.facts.is_empty()


async def test_failed_extraction_keeps_previous_memory():
    provider = ScriptedProvider(facts_queue=[{"goal": "Keep me"}])
    strat = StickyFactsStrategy()
    await strat.before_build(user_message="x", provider=provider, model="m")
    assert strat.facts.goal == "Keep me"
    provider.raise_on_extract = RuntimeError("provider down")
    result = await strat.before_build(user_message="y", provider=provider, model="m")
    assert result.performed is False
    assert result.error is not None
    # one failed extraction did NOT destroy the previous memory
    assert strat.facts.goal == "Keep me"


async def test_facts_parsing_tolerates_code_fences_and_prose():
    raw = '```json\n{"goal": "G", "decisions": ["A", "A", "B"]}\n```'
    facts = parse_facts_json(raw)
    assert facts.goal == "G"
    # duplicates removed
    assert facts.decisions == ["A", "B"]


def test_build_facts_prompt_contains_memory_and_new_message():
    from app.schemas.day10 import Facts

    payload = build_facts_prompt(Facts(goal="G"), "NEW TEXT")
    assert payload[0]["role"] == "system"
    assert "MEMORY" in payload[1]["content"]
    assert "NEW TEXT" in payload[1]["content"]


# ----------------------------------------------------------------------
# 3) Branching
# ----------------------------------------------------------------------
def _seeded_branching() -> tuple[BranchingStrategy, str, str, str]:
    strat = BranchingStrategy()
    conv = strat.conversation
    main = conv.active_branch
    for i in range(4):
        conv.add_message("user", f"common-u{i}")
        conv.add_message("assistant", f"common-a{i}")
    checkpoint = conv.branch_history(main.id)[-1].id
    a = strat.create_branch(name="A", parent_branch_id=main.id, checkpoint_message_id=checkpoint)
    b = strat.create_branch(name="B", parent_branch_id=main.id, checkpoint_message_id=checkpoint)
    return strat, main.id, a.id, b.id


def test_branch_created_from_checkpoint():
    strat, main_id, a_id, b_id = _seeded_branching()
    branch_a = strat.conversation.get_branch(a_id)
    assert branch_a.parent_branch_id == main_id
    assert branch_a.checkpoint_message_id is not None
    assert strat.conversation.checkpoint_index(a_id) == 8


def test_both_branches_share_the_prefix():
    strat, _main, a_id, b_id = _seeded_branching()
    hist_a = strat.conversation.branch_history(a_id)
    hist_b = strat.conversation.branch_history(b_id)
    assert hist_a == hist_b  # empty branches share the whole prefix
    assert len(hist_a) == 8


def test_messages_after_checkpoint_are_independent():
    strat, _main, a_id, b_id = _seeded_branching()
    strat.activate(a_id)
    strat.record_turn("A-only-user", "A-only-assistant")
    strat.activate(b_id)
    strat.record_turn("B-only-user", "B-only-assistant")

    hist_a = strat.conversation.branch_history(a_id)
    hist_b = strat.conversation.branch_history(b_id)
    # common prefix preserved
    assert hist_a[:8] == hist_b[:8]
    # Branch A message is absent in Branch B and vice versa
    assert all("A-only-user" != m.content for m in hist_b)
    assert all("B-only-user" != m.content for m in hist_a)
    assert any("A-only-user" == m.content for m in hist_a)
    assert any("B-only-user" == m.content for m in hist_b)


def test_switching_back_preserves_history():
    strat, _main, a_id, b_id = _seeded_branching()
    strat.activate(a_id)
    strat.record_turn("A1", "A1r")
    snapshot = [m.content for m in strat.conversation.branch_history(a_id)]
    strat.activate(b_id)
    strat.record_turn("B1", "B1r")
    strat.activate(a_id)
    assert [m.content for m in strat.conversation.branch_history(a_id)] == snapshot


def test_branch_context_contains_only_its_own_messages():
    strat, _main, a_id, b_id = _seeded_branching()
    strat.activate(a_id)
    strat.record_turn("A-only", "A-answer")
    strat.activate(b_id)
    built = strat.build_context(system_prompt="SYS", user_message="NEW")
    contents = [m["content"] for m in built.messages]
    assert "A-only" not in contents
    assert built.extra["active_branch_name"] == "B"


def test_branch_create_rejects_foreign_checkpoint():
    strat = BranchingStrategy()
    conv = strat.conversation
    conv.add_message("user", "hello")
    with pytest.raises(ValueError):
        conv.create_branch(checkpoint_message_id="does-not-exist")


def test_conversation_reset_creates_fresh_main():
    conv = Conversation.create_root()
    conv.add_message("user", "hello")
    conv.reset()
    assert len(conv.messages) == 0
    assert len(conv.branches) == 1
    assert conv.active_branch.name == "Main"


# ----------------------------------------------------------------------
# Service / scenario (deterministic provider)
# ----------------------------------------------------------------------
async def test_service_sliding_reports_dropped_and_sent():
    svc = _service(ScriptedProvider(answer="ok"))
    for i in range(8):
        await svc.chat(strategy="sliding_window", message=f"m{i}", window_size=4)
    response = await svc.chat(strategy="sliding_window", message="last", window_size=4)
    assert response.context.total_history_messages == 16
    assert response.context.sent_history_messages == 4
    assert response.context.dropped_messages == 12
    assert response.usage.prompt_tokens == 10


async def test_service_sticky_runs_extraction_and_counts_tokens():
    provider = ScriptedProvider(facts_queue=[{"goal": "G"}], answer="ok")
    svc = _service(provider)
    response = await svc.chat(strategy="sticky_facts", message="hello", recent_messages_limit=2)
    assert response.facts_extraction_performed is True
    assert response.usage.facts_extraction_total_tokens == 8
    assert response.usage.total_tokens == 14
    assert response.context.facts.goal == "G"


async def test_service_invalid_branch_id_raises_value_error():
    svc = _service(ScriptedProvider())
    with pytest.raises(ValueError):
        await svc.chat(strategy="branching", message="x", branch_id="nope")


async def test_run_scenario_sliding_loses_early_facts_sticky_keeps():
    settings = Settings(deepseek_api_key="test", environment="test")

    sliding_provider = ContextEchoProvider()
    sliding_svc = Day10ContextService(settings, deepseek=sliding_provider)
    sliding = await sliding_svc.run_scenario(strategy="sliding_window", window_size=6)

    sticky_provider = ContextEchoProvider()
    sticky_svc = Day10ContextService(settings, deepseek=sticky_provider)
    sticky = await sticky_svc.run_scenario(strategy="sticky_facts", recent_messages_limit=6)

    # The last scenario message is the control question; only context matters.
    assert sliding.evaluation.score < len(EXPECTED_FACTS_LABELS)
    assert sticky.evaluation.score == len(EXPECTED_FACTS_LABELS)
    assert "Python" in sticky.evaluation.matched
    assert sticky.usage.facts_extraction_total_tokens is not None

    # Sticky Facts really retained the early requirement even though the raw
    # message left the recent window.
    assert sticky.context.facts.count() >= 5


# ----------------------------------------------------------------------
# API level
# ----------------------------------------------------------------------
def test_api_state_valid_strategies(client):
    for strategy in ("sliding_window", "sticky_facts", "branching"):
        resp = client.get("/api/day10/state", params={"strategy": strategy})
        assert resp.status_code == 200, strategy
        assert resp.json()["strategy"] == strategy


def test_api_invalid_strategy_is_4xx(client):
    resp = client.get("/api/day10/state", params={"strategy": "bogus"})
    assert resp.status_code == 422
    resp = client.post("/api/day10/chat", json={"strategy": "bogus", "message": "x"})
    assert resp.status_code == 422


def test_api_sliding_chat(client):
    resp = client.post(
        "/api/day10/chat",
        json={"strategy": "sliding_window", "message": "hello", "window_size": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "sliding_window"
    assert body["usage"]["window_size"] == 2
    assert "context" in body


def test_api_scenario_endpoint(client):
    resp = client.get("/api/day10/scenario")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["messages"]) >= 10
    assert "Python" in body["expected"]
    assert "branches" in body["branching"]


def test_api_evaluate_endpoint(client):
    resp = client.post(
        "/api/day10/evaluate",
        json={"answer": "Python, FastAPI, PostgreSQL, JWT, Docker, 4 недели, учёт заявок"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] == body["total"] == len(EXPECTED_FACTS_LABELS)


def test_api_branch_seed_create_and_activate(client):
    seeded = client.post("/api/day10/branches/seed")
    assert seeded.status_code == 200
    branches = seeded.json()["context"]["branches"]
    assert len(branches) == 3  # Main + PostgreSQL + MongoDB
    active = seeded.json()["context"]["active_branch_id"]

    # activate another branch
    other = next(b["id"] for b in branches if b["id"] != active)
    resp = client.post(
        "/api/day10/branches/activate",
        json={"strategy": "branching", "branch_id": other},
    )
    assert resp.status_code == 200
    assert resp.json()["context"]["active_branch_id"] == other

    # create a new branch from the active one
    created = client.post("/api/day10/branches", json={"strategy": "branching", "name": "Extra"})
    assert created.status_code == 200
    assert created.json()["branch"]["name"] == "Extra"


def test_api_activate_unknown_branch_is_400(client):
    resp = client.post(
        "/api/day10/branches/activate",
        json={"strategy": "branching", "branch_id": "nope"},
    )
    assert resp.status_code == 400


def test_api_branch_chat_isolation(client):
    client.post("/api/day10/branches/seed")
    state = client.get("/api/day10/state", params={"strategy": "branching"}).json()
    branches = state["context"]["branches"]
    postgres = next(b for b in branches if b["name"] == "PostgreSQL")
    mongo = next(b for b in branches if b["name"] == "MongoDB")

    client.post(
        "/api/day10/branches/activate",
        json={"branch_id": postgres["id"]},
    )
    client.post(
        "/api/day10/chat",
        json={"strategy": "branching", "message": "decision-A", "branch_id": postgres["id"]},
    )
    client.post(
        "/api/day10/branches/activate",
        json={"branch_id": mongo["id"]},
    )
    mongo_state = client.post(
        "/api/day10/chat",
        json={"strategy": "branching", "message": "decision-B", "branch_id": mongo["id"]},
    ).json()

    for message in mongo_state["context"]["sent_preview"]:
        assert message["content"] != "decision-A"
    mongo_after = client.get("/api/day10/state", params={"strategy": "branching"}).json()
    mongo_contents = [m["content"] for m in mongo_after["context"]["sent_preview"]]
    assert "decision-B" in mongo_contents
    assert "decision-A" not in mongo_contents


def test_api_day10_does_not_touch_day7_history(client):
    client.post("/api/day10/branches/seed")
    client.post("/api/day10/chat", json={"strategy": "sliding_window", "message": "x"})
    assert client.get("/api/chat/history").json()["count"] == 0


def test_api_old_endpoints_still_work(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/day9/diagnostics").status_code == 200
    assert client.get("/api/day8/history").status_code == 200


def test_api_sticky_facts_with_real_extraction(client, settings):
    provider = ScriptedProvider(facts_queue=[{"goal": "Day10 goal"}], answer="ok")
    svc = Day10ContextService(settings, deepseek=provider)
    client.app.dependency_overrides[get_day10_service] = lambda: svc
    resp = client.post(
        "/api/day10/chat",
        json={"strategy": "sticky_facts", "message": "goal please", "update_facts": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["facts_extraction_performed"] is True
    assert body["context"]["facts"]["goal"] == "Day10 goal"


def test_api_facts_error_is_reported_not_crash(client, settings):
    provider = ScriptedProvider(facts_queue=["not json"], answer="ok")
    svc = Day10ContextService(settings, deepseek=provider)
    client.app.dependency_overrides[get_day10_service] = lambda: svc
    resp = client.post(
        "/api/day10/chat",
        json={"strategy": "sticky_facts", "message": "x", "update_facts": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["facts_extraction_performed"] is False
    assert body["facts_error"]
    assert body["answer"] == "ok"
