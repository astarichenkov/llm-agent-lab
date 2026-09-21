"""Day 8 — token accounting, cost, limits and context-overflow tests.

All provider calls are mocked; nothing here touches the real DeepSeek API.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.config import Settings
from app.services import model_params
from app.services.day8 import Day8TokenService
from app.services.deepseek import (
    DeepSeekInvalidRequestError,
    DeepSeekOutputLimitError,
)
from app.services.token_estimate import (
    estimate_context_breakdown,
    estimate_message_tokens,
    estimate_text_tokens,
)


# ----------------------------------------------------------------------
# estimation
# ----------------------------------------------------------------------
def test_estimate_empty_text_is_zero():
    assert estimate_text_tokens("") == 0
    assert estimate_message_tokens("") == 0


def test_estimate_latin_uses_calibrated_ratio():
    # 1000 ASCII letters * 0.21 = 210 tokens (measured ~0.21/char for letters)
    assert estimate_text_tokens("a" * 1000) == 210


def test_estimate_cjk_uses_calibrated_ratio():
    # 100 CJK chars * 1.0 = 100 tokens (measured ~1.02/char)
    assert estimate_text_tokens("漢" * 100) == 100


def test_estimate_cyrillic_uses_calibrated_ratio():
    # 100 Cyrillic chars * 0.336 = 33.6 -> ceil 34
    assert estimate_text_tokens("п" * 100) == 34


def test_estimate_message_adds_overhead():
    text_only = estimate_text_tokens("hello world")
    assert estimate_message_tokens("hello world") > text_only


def test_breakdown_empty_history():
    parts = estimate_context_breakdown(
        system_prompt="sys",
        history=[],
        current_user_message="hi",
    )
    assert parts["history_tokens"] == 0
    assert parts["current_user_tokens"] > 0
    assert parts["system_prompt_tokens"] > 0
    assert parts["estimated_input_tokens"] == (
        parts["system_prompt_tokens"]
        + parts["history_tokens"]
        + parts["current_user_tokens"]
    )


def test_breakdown_grows_with_history():
    small = estimate_context_breakdown(
        system_prompt="sys", history=[{"role": "user", "content": "a"}],
        current_user_message="q",
    )
    big = estimate_context_breakdown(
        system_prompt="sys",
        history=[{"role": "user", "content": "a" * 500} for _ in range(5)],
        current_user_message="q",
    )
    assert big["history_tokens"] > small["history_tokens"]
    assert big["estimated_input_tokens"] > small["estimated_input_tokens"]


def test_breakdown_current_message_only_differs_from_history():
    parts = estimate_context_breakdown(
        system_prompt="", history=[], current_user_message="unique question"
    )
    assert parts["history_tokens"] == 0
    assert parts["estimated_input_tokens"] == parts["current_user_tokens"]


# ----------------------------------------------------------------------
# model parameters & pricing
# ----------------------------------------------------------------------
def test_limits_from_docs_for_flash():
    limits = model_params.get_limits("deepseek-v4-flash")
    assert limits.context_window == 1_048_576
    assert limits.max_output_tokens == 384_000


def test_legacy_alias_matches_canonical_model():
    assert model_params.get_limits("deepseek-v4-flash") == model_params.get_limits(
        "deepseek-flash"
    )
    assert model_params.get_pricing("deepseek-v4-flash") == model_params.get_pricing(
        "deepseek-flash"
    )


def test_unknown_model_has_no_pricing():
    assert model_params.get_pricing("some/unknown-model") is None
    assert model_params.is_known_model("some/unknown-model") is False


def test_cost_formula_known_values():
    # flash off-peak: miss $0.15 / output $0.6 per 1M tokens.
    cost = model_params.estimate_cost(
        "deepseek-v4-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000
    )
    assert cost["input"] == Decimal("0.15000000")
    assert cost["output"] == Decimal("0.60000000")
    assert cost["total"] == Decimal("0.75000000")
    assert cost["currency"] == "USD"
    assert cost["estimated"] is True


def test_cost_uses_cache_hit_rate_when_split_present():
    # 1M prompt, 500k cache hit -> 500k hit at 0.003 + 500k miss at 0.15
    cost = model_params.estimate_cost(
        "deepseek-v4-flash",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_hit_tokens=500_000,
        cache_miss_tokens=500_000,
    )
    expected = (Decimal(500_000) * Decimal("0.003") + Decimal(500_000) * Decimal("0.15")) / Decimal(1_000_000)
    assert cost["input"] == expected.quantize(Decimal("0.00000001"))


def test_cost_unknown_model_is_not_fabricated():
    cost = model_params.estimate_cost("mystery/model", prompt_tokens=100, completion_tokens=10)
    assert cost["total"] is None
    assert cost["estimated"] is False


def test_cost_scales_linearly():
    one = model_params.estimate_cost("deepseek-v4-flash", prompt_tokens=1000, completion_tokens=200)
    two = model_params.estimate_cost("deepseek-v4-flash", prompt_tokens=2000, completion_tokens=400)
    assert two["total"] == one["total"] * 2


# ----------------------------------------------------------------------
# usage mapping (exact API values)
# ----------------------------------------------------------------------
async def test_chat_maps_exact_usage(fake_service, settings):
    fake_service.answer = "ok"
    svc = Day8TokenService(settings, deepseek=fake_service)
    result = await svc.chat(message="hi", history=None, model=None, system_prompt=None)
    # FakeDeepSeekService returns prompt=1 completion=1 total=2
    assert result.usage.prompt_tokens == 1
    assert result.usage.completion_tokens == 1
    assert result.usage.total_tokens == 2
    # estimates are present and independent
    assert result.usage.current_user_tokens_estimated > 0
    assert result.usage.estimated_input_tokens > 0


async def test_chat_uses_documented_max_output_as_output_budget(fake_service, settings):
    svc = Day8TokenService(settings, deepseek=fake_service)
    await svc.chat(message="hi", history=None, model=None, system_prompt=None)
    call = fake_service.generate_calls[-1]
    assert call["max_tokens"] == 384_000  # Day 8 uses the model maximum output
    assert call["thinking"] is False


async def test_chat_history_breakdown_matches_sent_context(fake_service, settings):
    svc = Day8TokenService(settings, deepseek=fake_service)
    history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    result = await svc.chat(
        message="new question", history=history, model=None, system_prompt=None
    )
    assert result.usage.history_tokens_estimated > 0
    # synthetic history must NOT be persisted into the isolated dialog
    assert svc.history == []


# ----------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------
def test_day8_estimate_endpoint_no_llm_call(client, fake_service):
    response = client.post(
        "/api/day8/estimate",
        json={"message": "hello", "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["estimated_input_tokens"] > 0
    assert body["limits"]["context_window"] == 1_048_576
    # the provider counts the output budget toward the context window
    assert body["effective_input_tokens"] == (
        body["usage"]["estimated_input_tokens"]
        + body["limits"]["max_output_tokens"]
    )
    assert body["exceeds_context"] is False
    assert fake_service.generate_calls == []  # zero provider calls


def test_day8_estimate_detects_overflow_without_api_call(client, fake_service):
    # a huge synthetic history -> estimate must exceed the limit locally
    history = [
        {"role": "user", "content": "x" * 200_000},
        {"role": "assistant", "content": "y" * 200_000},
    ] * 20  # ~8M chars -> ~1.7M estimated tokens > 1M window
    response = client.post(
        "/api/day8/estimate", json={"message": "go", "history": history}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["exceeds_context"] is True
    assert body["overflow_tokens"] > 0
    assert fake_service.generate_calls == []


def test_day8_chat_returns_usage_cost_limits(client):
    response = client.post("/api/day8/chat", json={"message": "hi"})
    assert response.status_code == 200
    body = response.json()
    assert body["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 1
    assert body["usage"]["completion_tokens"] == 1
    assert body["cost"]["estimated"] is True
    assert body["limits"]["context_window"] == 1_048_576


def test_day8_overflow_endpoint_generates_server_side_context(client, fake_service):
    response = client.post("/api/day8/overflow", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["exceeds_context"] is True
    assert body["overflow_tokens"] > 0
    assert body["turns"] > 0
    # the returned request is a ready-to-send chat payload
    req = body["request"]
    assert req["message"]
    assert isinstance(req["history"], list)
    assert len(req["history"]) == body["turns"] * 2
    # generation is LOCAL only -> no provider call
    assert fake_service.generate_calls == []


def test_day8_overflow_is_deterministic(client):
    a = client.post("/api/day8/overflow", json={}).json()
    b = client.post("/api/day8/overflow", json={}).json()
    assert a["usage"]["estimated_input_tokens"] == b["usage"]["estimated_input_tokens"]
    assert a["turns"] == b["turns"]
    assert a["request"]["history"][0] == b["request"]["history"][0]


def test_day8_overflow_request_runs_through_chat_endpoint(client, fake_service):
    """The generated payload must be accepted by /api/day8/chat (round-trip)."""
    gen = client.post("/api/day8/overflow", json={"overshoot_factor": 1.05}).json()
    resp = client.post("/api/day8/chat", json=gen["request"])
    assert resp.status_code == 200
    assert resp.json()["usage"]["history_tokens_estimated"] > 1_000_000


def test_day8_clear_and_history(client):
    client.post("/api/day8/chat", json={"message": "one"})
    hist = client.get("/api/day8/history").json()
    assert hist["count"] == 2
    assert client.delete("/api/day8/history").status_code == 200
    assert client.get("/api/day8/history").json()["count"] == 0


def test_day8_does_not_touch_day7_history(client):
    client.post("/api/day8/chat", json={"message": "day8 only"})
    assert client.get("/api/chat/history").json()["count"] == 0


# ----------------------------------------------------------------------
# error separation
# ----------------------------------------------------------------------
def test_day8_context_overflow_maps_to_provider_status(client, fake_service):
    fake_service.raise_error = DeepSeekInvalidRequestError(
        "This model's maximum context length is exceeded.", 400
    )
    response = client.post("/api/day8/chat", json={"message": "too big"})
    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
    assert "context length" in body["detail"].lower()
    # never leak internals
    assert "Traceback" not in body["detail"]
    assert "sk-" not in body["detail"]


def test_day8_output_limit_still_maps_to_502(client, fake_service):
    fake_service.raise_error = DeepSeekOutputLimitError()
    response = client.post("/api/day8/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "max_tokens" in response.json()["detail"]
