"""Tests for Day 5 (OpenRouter). No real OpenRouter calls are made."""
import asyncio

import pytest

from app.config import Settings
from app.or_models import (
    DEFAULT_MODELS,
    MODEL_ALLOWLIST,
    get_model,
    is_allowed,
    models_for_ui,
)
from app.schemas.day5 import OpenRouterRunRequest
from app.services.openrouter_service import OpenRouterError, OpenRouterService

DEFAULT_PROMPT = "Объясни простыми словами, почему бинарный поиск работает быстрее линейного."


# ---------------- CONFIG ----------------
def test_openrouter_key_setting():
    s = Settings(deepseek_api_key="dk", openrouter_api_key="or-test")
    assert s.openrouter_api_key == "or-test"
    assert s.deepseek_api_key == "dk"  # DeepSeek unaffected
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_missing_openrouter_key_service_raises():
    # force an empty OR key (env may contain a real one)
    svc = OpenRouterService(Settings(deepseek_api_key="dk", openrouter_api_key=""))
    req = OpenRouterRunRequest(message="x", model=DEFAULT_MODELS["weak"])
    with pytest.raises(OpenRouterError) as e:
        asyncio.run(svc.run(req))
    assert e.value.status_code == 503


# ---------------- MODELS ----------------
def test_models_list_present_and_complete():
    assert len(MODEL_ALLOWLIST) >= 3
    for e in MODEL_ALLOWLIST:
        assert e["id"] and e["name"]
        assert "category" in e and e["url"].startswith("https://openrouter.ai/")
        assert set(e) >= {"id", "name", "url", "input_per_million", "output_per_million", "context_length"}


def test_defaults_cover_categories():
    for cat in ("weak", "medium", "strong"):
        assert DEFAULT_MODELS[cat] in {e["id"] for e in MODEL_ALLOWLIST}


def test_invalid_model_not_allowed():
    assert not is_allowed("fake/not-a-model")
    assert get_model("fake/not-a-model") is None


def test_models_for_ui_is_safe_copy():
    for e in models_for_ui():
        assert "api_key" not in str(e).lower()
        assert "authorization" not in str(e).lower()


# ---------------- SERVICE (mocked provider) ----------------
class _Msg:
    pass


class _Ch:
    pass


class _Completion:
    pass


def _completion(content, model="qwen/qwen-2.5-7b-instruct", finish="stop", usage=None):
    m = _Msg(); m.content = content
    c = _Ch(); c.message = m; c.finish_reason = finish
    r = _Completion(); r.choices = [c]; r.model = model; r.usage = usage
    return r


def _usage(prompt=50, completion=80, total=130, cost=None):
    class U:
        pass

    u = U(); u.prompt_tokens = prompt; u.completion_tokens = completion
    u.total_tokens = total; u.cost = cost
    return u


class _FakeCompletions:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


def make_service(monkeypatch, results):
    class _Client:
        pass

    class _Chat:
        pass

    completions = _FakeCompletions(results)
    chat = type("_Chat", (), {"completions": completions})()
    client = type("_Client", (), {"chat": chat})()

    captured = {}
    def factory(**kwargs):
        captured.update(kwargs)
        return client
    monkeypatch.setattr("app.services.openrouter_service.AsyncOpenAI", factory)
    svc = OpenRouterService(Settings(deepseek_api_key="dk", openrouter_api_key="or-test"))
    return svc, completions, captured


def _req(model_id=None, **over):
    base = dict(message=DEFAULT_PROMPT, model=model_id or DEFAULT_MODELS["weak"],
                max_tokens=500, temperature=0.3, stop_sequence=None)
    base.update(over)
    return OpenRouterRunRequest(**base)


def test_request_passed_to_provider_exactly(monkeypatch):
    svc, comp, captured = make_service(monkeypatch, [_completion("Ответ")])
    resp = asyncio.run(svc.run(_req(DEFAULT_MODELS["weak"], stop_sequence="STOP")))
    call = comp.calls[0]
    assert call["model"] == DEFAULT_MODELS["weak"]
    assert call["messages"] == [{"role": "user", "content": DEFAULT_PROMPT}]
    assert call["max_tokens"] == 500
    assert call["temperature"] == 0.3
    assert call["stop"] == ["STOP"]
    assert "thinking" not in call  # no DeepSeek thinking config sent
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert resp.answer == "Ответ"


def test_empty_stop_omitted(monkeypatch):
    svc, comp, _ = make_service(monkeypatch, [_completion("Ответ")])
    asyncio.run(svc.run(_req(stop_sequence=None)))
    assert "stop" not in comp.calls[0]


def test_actual_model_returned_and_usage_propagated(monkeypatch):
    svc, _, _ = make_service(monkeypatch, [_completion("Ответ", model="provider/actual", usage=_usage(cost=0.0002))])
    resp = asyncio.run(svc.run(_req()))
    assert resp.requested_model == DEFAULT_MODELS["weak"]
    assert resp.actual_model == "provider/actual"
    assert resp.usage == {"prompt_tokens": 50, "completion_tokens": 80, "total_tokens": 130}
    assert resp.cost == 0.0002  # provider-reported
    assert resp.cost_estimated is False
    assert resp.elapsed_ms >= 0
    assert resp.finish_reason == "stop"


def test_cost_estimated_when_provider_missing_but_pricing_known(monkeypatch):
    svc, _, _ = make_service(monkeypatch, [_completion("Ответ", usage=_usage(cost=None))])
    resp = asyncio.run(svc.run(_req()))
    # provider did not return cost, but weak model pricing is known -> estimate
    assert resp.cost == pytest.approx((50 * 0.10 + 80 * 0.20) / 1_000_000.0)
    assert resp.cost_estimated is True
    assert resp.usage == {"prompt_tokens": 50, "completion_tokens": 80, "total_tokens": 130}


def test_invalid_model_rejected_before_provider_call(monkeypatch):
    svc, comp, _ = make_service(monkeypatch, [_completion("Ответ")])
    with pytest.raises(OpenRouterError) as e:
        asyncio.run(svc.run(_req(model_id="fake/not-a-model")))
    assert e.value.status_code == 400
    assert len(comp.calls) == 0  # zero provider calls


# ---------------- API (mocked service) ----------------
from fastapi.testclient import TestClient
from app.main import create_app
from app.api.routes import get_openrouter_service
from app.schemas.day5 import OpenRouterRunResponse


class _FakeOR:
    def __init__(self):
        self.calls = []

    async def run(self, request):
        self.calls.append(request)
        return OpenRouterRunResponse(
            answer="Ответ (fake)", requested_model=request.model,
            actual_model=request.model, finish_reason="stop", elapsed_ms=12.5,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            cost=None, cost_estimated=False, model_url="https://openrouter.ai/x",
        )


@pytest.fixture
def or_app(monkeypatch):
    app = create_app(Settings(deepseek_api_key="dk", openrouter_api_key="or-test"))
    fake = _FakeOR()
    app.dependency_overrides[get_openrouter_service] = lambda: fake
    return app, fake


def test_api_openrouter_models_list(or_app):
    c = TestClient(or_app[0])
    d = c.get("/api/openrouter/models").json()
    assert len(d["models"]) == 3
    assert set(d["defaults"]) == {"weak", "medium", "strong"}


def test_api_run_success(or_app):
    c, fake = TestClient(or_app[0]), or_app[1]
    r = c.post("/api/openrouter/run", json={
        "message": "Привет", "model": DEFAULT_MODELS["weak"],
        "max_tokens": 500, "temperature": 0.3, "stop_sequence": None})
    assert r.status_code == 200
    body = r.json()
    assert body["requested_model"] == DEFAULT_MODELS["weak"]
    assert body["answer"]
    assert len(fake.calls) == 1


def test_api_validation_zero_provider_calls(or_app):
    c, fake = TestClient(or_app[0]), or_app[1]
    bads = [
        {"message": "", "model": DEFAULT_MODELS["weak"]},
        {"message": "x", "model": DEFAULT_MODELS["weak"], "max_tokens": 5},
        {"message": "x", "model": DEFAULT_MODELS["weak"], "temperature": 3},
        {"message": "x", "model": "", },
    ]
    for p in bads:
        c.post("/api/openrouter/run", json=p)
    assert len(fake.calls) == 0


def test_strong_default_is_current_sonnet():
    from app.or_models import MODEL_ALLOWLIST
    strong = next(e for e in MODEL_ALLOWLIST if e["category"] == "strong")
    assert strong["id"] == "anthropic/claude-sonnet-4.6"
    assert "claude-3.5-sonnet" not in {e["id"] for e in MODEL_ALLOWLIST}
    assert strong["input_per_million"] == 3.0
    assert strong["output_per_million"] == 15.0
