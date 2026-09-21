"""Day 9 — context compression tests.

Pure policy tests run without any provider. Service tests use the shared
deterministic fake; no real API call is required for the default suite.
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.schemas.day9 import Day9ChatRequest
from app.services.context_compression import (
    CompressionConfig,
    CompressionState,
    build_summary_prompt,
)
from app.services.day9 import Day9CompressionService
from app.services.deepseek import DeepSeekError


def _msg(role: str, text: str) -> dict[str, str]:
    return {"role": role, "content": text}


def _history(n: int) -> list[dict[str, str]]:
    out = []
    for i in range(n):
        out.append(_msg("user", f"u{i}"))
        out.append(_msg("assistant", f"a{i}"))
    return out


# ----------------------------------------------------------------------
# pure policy
# ----------------------------------------------------------------------
def test_short_history_does_not_compress():
    cfg = CompressionConfig(recent_messages_limit=6, compression_batch_size=10)
    state = CompressionState(full_history=_history(2))  # 4 messages < window
    assert state.pending_messages(cfg) == []
    assert state.should_compress(cfg) is False


def test_messages_older_than_window_are_detected():
    cfg = CompressionConfig(recent_messages_limit=6, compression_batch_size=10)
    state = CompressionState(full_history=_history(10))  # 20 messages
    # window = last 6 -> 14 older messages
    assert state.window_start_index(cfg) == 14
    assert len(state.pending_messages(cfg)) == 14
    assert state.should_compress(cfg) is True


def test_compression_batch_threshold():
    cfg = CompressionConfig(recent_messages_limit=6, compression_batch_size=10)
    # 13 messages -> 7 older than window -> below batch of 10
    state = CompressionState(full_history=_history(7)[:13])
    assert len(state.pending_messages(cfg)) == 7
    assert state.should_compress(cfg) is False
    # add one more -> 8 older ... add until >= 10
    state.full_history = _history(8)  # 16 messages -> 10 older
    assert len(state.pending_messages(cfg)) == 10
    assert state.should_compress(cfg) is True


def test_already_summarized_messages_are_not_pending_again():
    cfg = CompressionConfig(recent_messages_limit=4, compression_batch_size=2)
    state = CompressionState(full_history=_history(10))  # 20 messages
    state.summarized_count = 12  # first 12 already folded in
    pending = state.pending_messages(cfg)
    # window start = 20-4 = 16 -> pending = messages 12..15 (4 messages)
    assert len(pending) == 4
    assert pending[0] == state.full_history[12]
    # messages already summarized are never returned
    assert all(p not in state.full_history[:12] for p in pending)


def test_recent_window_always_verbatim():
    cfg = CompressionConfig(recent_messages_limit=4, compression_batch_size=2)
    state = CompressionState(full_history=_history(10))
    state.summarized_count = 16
    window = state.recent_window(cfg)
    assert window == state.full_history[-4:]


def test_next_summarized_count_advances_to_window_start():
    cfg = CompressionConfig(recent_messages_limit=4, compression_batch_size=2)
    state = CompressionState(full_history=_history(10))  # 20 messages
    assert state.next_summarized_count(cfg) == 16
    state.summarized_count = 16
    assert state.pending_messages(cfg) == []


def test_build_summary_prompt_uses_previous_summary_and_new_block_only():
    block = [_msg("user", "new fact"), _msg("assistant", "ok")]
    payload = build_summary_prompt("OLD SUMMARY TEXT", block)
    assert payload[0]["role"] == "system"
    user_content = payload[1]["content"]
    assert "OLD SUMMARY TEXT" in user_content
    assert "new fact" in user_content
    assert "ok" in user_content


def test_working_context_contains_summary_then_recent():
    cfg = CompressionConfig(recent_messages_limit=2, compression_batch_size=2)
    state = CompressionState(full_history=_history(4))  # 8 messages
    state.summary = "durable facts"
    state.summarized_count = 6
    ctx = state.build_working_context(cfg, "SYSTEM")
    assert ctx[0] == {"role": "system", "content": "SYSTEM"}
    assert "durable facts" in ctx[1]["content"]
    assert ctx[2:] == state.full_history[-2:]
    # summarized messages are NOT present verbatim
    assert all(m not in ctx for m in state.full_history[:6])


# ----------------------------------------------------------------------
# service (with the shared fake provider)
# ----------------------------------------------------------------------
class _FakeProvider:
    """Records calls and returns canned content per call."""

    def __init__(self, summary: str = "SUMMARY-V1") -> None:
        self.calls: list[dict] = []
        self.summary = summary
        self.raise_error: Exception | None = None
        self.empty_summary = False

    async def generate(self, messages, *, model=None, max_tokens=1024, thinking=False):
        self.calls.append({"messages": [dict(m) for m in messages], "model": model})
        if self.raise_error is not None:
            raise self.raise_error
        # The summary call has a system message containing "compressor".
        is_summary = any(
            "compressor" in (m.get("content") or "") for m in messages
        )
        if is_summary:
            content = "" if self.empty_summary else self.summary
        else:
            content = "ANSWER"
        return content, "stop", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def _service(provider, **kwargs) -> Day9CompressionService:
    settings = Settings(deepseek_api_key="test", environment="test", **kwargs)
    return Day9CompressionService(settings, deepseek=provider)


async def test_service_short_history_no_compression():
    provider = _FakeProvider()
    svc = _service(provider)
    svc.seed_demo(recent_messages_limit=6, compression_batch_size=50)
    resp = await svc.chat(message="hi", mode="compressed")
    assert resp.compression_performed is False
    assert resp.diagnostics.compression_cycles == 0
    # only the normal answer call happened (no summary call)
    assert len(provider.calls) == 1


async def test_service_compression_runs_when_batch_reached():
    provider = _FakeProvider(summary="ORION FACTS")
    svc = _service(provider)
    svc.seed_demo(recent_messages_limit=6, compression_batch_size=10)
    # seed_demo installs 10 opening + 14*2 = 38 messages -> pending >= 10
    resp = await svc.chat(message="check", mode="compressed")
    assert resp.compression_performed is True
    assert resp.diagnostics.compression_cycles == 1
    assert resp.diagnostics.summary == "ORION FACTS"
    assert resp.diagnostics.summarized_messages > 0
    # a summary call + an answer call
    assert len(provider.calls) == 2


async def test_service_recent_window_verbatim_after_compression():
    provider = _FakeProvider()
    svc = _service(provider)
    svc.seed_demo(recent_messages_limit=6, compression_batch_size=10)
    # Capture the window as it was BEFORE this turn was appended.
    window_before = svc.state.recent_window(svc._config)
    await svc.chat(message="check", mode="compressed")
    # the answer call must contain those last 6 messages verbatim
    answer_call = provider.calls[-1]["messages"]
    for m in window_before:
        assert m in answer_call
    # and after committing the turn the window is the last 6 of the new history
    assert svc.state.recent_window(svc._config) == svc.state.full_history[-6:]


async def test_service_summary_uses_previous_summary_and_new_block_only():
    provider = _FakeProvider(summary="S1")
    svc = _service(provider)
    svc.seed_demo(recent_messages_limit=6, compression_batch_size=10)
    await svc.chat(message="first", mode="compressed")
    first_cycles = svc.state.compression_cycles
    # add enough messages to exceed the batch again
    for i in range(12):
        svc.state.full_history.append(_msg("user", f"extra-u{i}"))
        svc.state.full_history.append(_msg("assistant", f"extra-a{i}"))
    provider.summary = "S2"
    await svc.chat(message="second", mode="compressed")
    assert svc.state.compression_cycles == first_cycles + 1
    # the second summary call must include the FIRST summary text
    summary_calls = [
        c for c in provider.calls if any("compressor" in (m.get("content") or "") for m in c["messages"])
    ]
    assert len(summary_calls) == 2
    second_prompt = summary_calls[-1]["messages"][-1]["content"]
    assert "S1" in second_prompt


async def test_service_summary_failure_keeps_state():
    provider = _FakeProvider()
    provider.raise_error = DeepSeekError("boom", 502)
    svc = _service(provider)
    svc.seed_demo(recent_messages_limit=6, compression_batch_size=10)
    before = svc.state.summarized_count
    with pytest.raises(DeepSeekError):
        await svc.chat(message="x", mode="compressed")
    # failed summary did not corrupt the state
    assert svc.state.summarized_count == before
    assert svc.state.summary is None


async def test_service_empty_summary_is_reported_and_not_committed():
    provider = _FakeProvider()
    provider.empty_summary = True
    svc = _service(provider)
    svc.seed_demo(recent_messages_limit=6, compression_batch_size=10)
    resp = await svc.chat(message="x", mode="compressed")
    assert resp.compression_performed is False
    assert resp.compression_error is not None
    assert svc.state.summary is None
    assert svc.state.compression_cycles == 0


async def test_service_full_mode_has_zero_savings():
    provider = _FakeProvider()
    svc = _service(provider)
    svc.seed_demo(recent_messages_limit=6, compression_batch_size=10)
    resp = await svc.chat(message="x", mode="full")
    assert resp.usage.tokens_saved == 0
    assert resp.usage.tokens_saved_percent == 0.0
    # full mode never builds a summary
    assert svc.state.compression_cycles == 0


async def test_service_compressed_mode_reports_savings():
    provider = _FakeProvider()
    svc = _service(provider)
    svc.seed_demo(recent_messages_limit=6, compression_batch_size=10)
    resp = await svc.chat(message="x", mode="compressed")
    assert resp.usage.full_context_tokens_estimated > 0
    assert (
        resp.usage.compressed_context_tokens_estimated
        < resp.usage.full_context_tokens_estimated
    )
    assert resp.usage.tokens_saved > 0
    assert resp.usage.tokens_saved_percent > 0


def test_reset_clears_history_summary_and_metadata():
    provider = _FakeProvider()
    svc = _service(provider)
    svc.seed_demo()
    svc.state.summary = "something"
    svc.state.summarized_count = 10
    svc.state.compression_cycles = 2
    svc.reset()
    assert svc.state.full_history == []
    assert svc.state.summary is None
    assert svc.state.summarized_count == 0
    assert svc.state.compression_cycles == 0


def test_savings_safe_against_division_by_zero():
    saved, pct = Day9CompressionService._savings(0, 0)
    assert saved == 0
    assert pct == 0.0


# ----------------------------------------------------------------------
# API level
# ----------------------------------------------------------------------
def test_api_seed_and_diagnostics(client):
    seeded = client.post("/api/day9/seed", json={}).json()
    assert seeded["count"] > 0
    assert seeded["diagnostics"]["full_history_messages"] == seeded["count"]
    diag = client.get("/api/day9/diagnostics").json()["diagnostics"]
    assert diag["compression_cycles"] == 0
    assert diag["summary"] is None


def test_api_compressed_chat_runs_compression(client):
    client.post("/api/day9/seed", json={"compression_batch_size": 10})
    resp = client.post(
        "/api/day9/chat", json={"message": "контрольный вопрос", "mode": "compressed"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "compressed"
    assert body["compression_performed"] is True
    assert body["diagnostics"]["compression_cycles"] == 1
    assert body["usage"]["tokens_saved"] > 0
    assert body["usage"]["tokens_saved_percent"] > 0


def test_api_full_mode_has_no_savings(client):
    client.post("/api/day9/seed", json={"compression_batch_size": 10})
    resp = client.post(
        "/api/day9/chat", json={"message": "контрольный вопрос", "mode": "full"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"]["tokens_saved"] == 0
    assert body["diagnostics"]["compression_cycles"] == 0


def test_api_clear_removes_everything(client):
    client.post("/api/day9/seed", json={})
    client.post("/api/day9/chat", json={"message": "x", "mode": "compressed"})
    assert client.delete("/api/day9/history").status_code == 200
    diag = client.get("/api/day9/diagnostics").json()["diagnostics"]
    assert diag["full_history_messages"] == 0
    assert diag["summary"] is None
    assert diag["compression_cycles"] == 0


def test_api_invalid_mode_rejected(client):
    resp = client.post("/api/day9/chat", json={"message": "x", "mode": "bogus"})
    assert resp.status_code == 422


def test_api_day9_does_not_touch_day7_history(client):
    client.post("/api/day9/seed", json={})
    client.post("/api/day9/chat", json={"message": "x", "mode": "compressed"})
    assert client.get("/api/chat/history").json()["count"] == 0


def test_api_diagnostics_exposes_model(client):
    body = client.get("/api/day9/diagnostics").json()
    assert body["model"]
    assert "diagnostics" in body


# ----------------------------------------------------------------------
# transparency: what the UI shows as "agent memory"
# ----------------------------------------------------------------------
def test_api_compressed_chat_exposes_summary_for_ui(client):
    """The response must carry the summary + zone counts the dialog renders."""
    client.post("/api/day9/seed", json={"compression_batch_size": 10})
    body = client.post(
        "/api/day9/chat", json={"message": "контрольный вопрос", "mode": "compressed"}
    ).json()
    diag = body["diagnostics"]
    assert body["compression_performed"] is True
    assert diag["summary"]  # agent memory is visible to the UI
    assert diag["summarized_messages"] > 0
    assert diag["recent_messages_sent"] <= diag["full_history_messages"]
    assert diag["pending_messages"] >= 0
