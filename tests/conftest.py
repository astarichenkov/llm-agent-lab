"""Shared fixtures for the test suite.

No test in this suite ever touches the real DeepSeek API: every normal
DeepSeek call is replaced either by dependency override (API tests) or by
mocking the OpenAI client (service unit tests). The only exception is the
explicitly-marked integration smoke test (``pytest -m integration``).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import get_agent_manager, get_deepseek_service
from app.agents.llm import DeepSeekLLMClient
from app.agents.manager import AgentManager
from app.agents.repository import SQLiteContextRepository
from app.config import Settings, get_settings
from app.main import create_app
from app.schemas.chat import ChatResponse
from app.schemas.compare import (
    FIXED_RESPONSE_FORMAT,
    CompareRequest,
    CompareResponse,
    CompareResult,
    ControlledSettings,
)
from app.schemas.reasoning import ReasoningRequest, ReasoningResponse
from app.schemas.temperature import TemperatureRequest, TemperatureResponse

class FakeDeepSeekService:
    """In-memory stand-in for ``DeepSeekService``.

    Records the last message it received and can be told to raise a
    specific exception to exercise error handling in the API layer.
    ``compare`` returns a canned comparison unless ``compare_result`` is
    set (e.g. to simulate a partial failure).
    """

    def __init__(self, answer: str = "Mocked answer from DeepSeek.") -> None:
        self.answer = answer
        self.last_message: str | None = None
        self.raise_error: Exception | None = None
        self.compare_calls: list[CompareRequest] = []
        self.compare_result: CompareResponse | None = None
        self.reasoning_calls: list[ReasoningRequest] = []
        self.reasoning_result: ReasoningResponse | None = None
        self.temperature_calls: list[TemperatureRequest] = []
        # Day 7: generic completions used by the Agent layer.
        self.generate_calls: list[dict] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: str | None = None,
        thinking: bool | None = None,
    ) -> tuple[str, str | None, dict | None]:
        """Mirror DeepSeekService.generate (used by the persistent Agent)."""
        self.generate_calls.append(
            {
                "messages": [dict(m) for m in messages],
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                "thinking": thinking,
            }
        )
        if messages:
            self.last_message = messages[-1].get("content")
        if self.raise_error is not None:
            raise self.raise_error
        return (
            self.answer,
            "stop",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def chat(self, message: str) -> ChatResponse:
        self.last_message = message
        if self.raise_error is not None:
            raise self.raise_error
        return ChatResponse(answer=self.answer)

    async def compare(self, request: CompareRequest) -> CompareResponse:
        self.compare_calls.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        if self.compare_result is not None:
            return self.compare_result
        stop_list = [request.stop_sequence] if request.stop_sequence else None
        settings = ControlledSettings(
            response_format=dict(FIXED_RESPONSE_FORMAT),
            max_tokens=request.max_tokens,
            stop=stop_list,
            json_structure=request.json_structure,
        )
        return CompareResponse(
            unrestricted=CompareResult(
                answer="Unrestricted answer.", finish_reason="stop"
            ),
            settings=settings,
            controlled=CompareResult(
                answer='{"products": []}',
                finish_reason="length",
                settings=settings,
            ),
        )

    async def reasoning(self, request: ReasoningRequest) -> ReasoningResponse:
        self.reasoning_calls.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        if self.reasoning_result is not None:
            return self.reasoning_result
        if request.method == "generate_prompt":
            return ReasoningResponse(
                method="generate_prompt",
                kind="generated_prompt",
                prompt_sent="stage1 prompt",
                generated_prompt="Реши задачу: " + request.task[:20],
                finish_reason="stop",
                usage={"completion_tokens": 40},
            )
        return ReasoningResponse(
            method=request.method,
            kind="solution",
            prompt_sent="prompt",
            solution="Ответ (mock)",
            finish_reason="stop",
            status="indeterminate",
        )

    async def complete_with_temperature(self, request: TemperatureRequest) -> TemperatureResponse:
        self.temperature_calls.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        return TemperatureResponse(
            answer=self.answer,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40},
            applied_parameters={
                "model": "deepseek-v4-flash",
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stop": [request.stop_sequence] if request.stop_sequence else None,
            },
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(deepseek_api_key="test-key", environment="test")


@pytest.fixture
def fake_service() -> FakeDeepSeekService:
    return FakeDeepSeekService()


@pytest.fixture
def agent_manager(settings: Settings, fake_service: FakeDeepSeekService, tmp_path):
    """AgentManager backed by a throwaway SQLite DB and the fake LLM."""
    repository = SQLiteContextRepository(tmp_path / "agents.db")
    llm = DeepSeekLLMClient(fake_service)
    return AgentManager(
        settings,
        repository,
        {"deepseek": llm, "openrouter": llm},
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings=settings)


@pytest.fixture
def client(
    app: FastAPI,
    settings: Settings,
    fake_service: FakeDeepSeekService,
    agent_manager: AgentManager,
):
    """TestClient with the DeepSeek service and AgentManager swapped for fakes."""
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_deepseek_service] = lambda: fake_service
    app.dependency_overrides[get_agent_manager] = lambda: agent_manager
    # Starlette 1.x re-raises handled server errors by design; the app ships
    # a global error handler, so capture its response instead.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
