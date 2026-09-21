"""Shared fixtures for the test suite.

No test in this suite ever touches the real DeepSeek API: every normal
DeepSeek call is replaced either by dependency override (API tests) or by
mocking the OpenAI client (service unit tests). The only exception is the
explicitly-marked integration smoke test (``pytest -m integration``).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import (
    get_agent_manager,
    get_day8_service,
    get_day9_service,
    get_day10_service,
    get_day11_service,
    get_day12_service,
    get_day13_service,
    get_day14_service,
    get_day15_service,
    get_deepseek_service,
)
from app.agents.llm import DeepSeekLLMClient
from app.agents.manager import AgentManager
from app.agents.repository import SQLiteContextRepository
from app.config import Settings, get_settings
from app.main import create_app
from app.schemas.chat import ChatResponse
from app.services.day8 import Day8TokenService
from app.services.day9 import Day9CompressionService
from app.services.day10 import Day10ContextService
from app.services.day11 import Day11MemoryService
from app.services.day11.store import LongTermMemoryStore
from app.services.day12 import Day12ProfileService, ProfileStore
from app.services.day13 import Day13TaskService, TaskStore
from app.services.day14 import Day14InvariantService, InvariantStore
from app.services.day15 import Day15LifecycleService, LifecycleStore
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
def settings(tmp_path) -> Settings:
    return Settings(
        deepseek_api_key="test-key",
        environment="test",
        day11_long_term_path=str(tmp_path / "day11_long_term.json"),
        day12_profile_path=str(tmp_path / "day12_profile.json"),
        day13_task_path=str(tmp_path / "day13_task_state.json"),
        day14_invariants_path=str(tmp_path / "day14_invariants.json"),
        day15_task_path=str(tmp_path / "day15_lifecycle_state.json"),
    )


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
def day8_service(settings: Settings, fake_service: FakeDeepSeekService) -> Day8TokenService:
    """Day 8 token service backed by the deterministic fake provider."""
    return Day8TokenService(settings, deepseek=fake_service)


@pytest.fixture
def day9_service(settings: Settings, fake_service: FakeDeepSeekService) -> Day9CompressionService:
    """Day 9 compression service backed by the deterministic fake provider."""
    return Day9CompressionService(settings, deepseek=fake_service)


@pytest.fixture
def day10_service(settings: Settings, fake_service: FakeDeepSeekService) -> Day10ContextService:
    """Day 10 context-strategy service backed by the deterministic fake provider."""
    return Day10ContextService(settings, deepseek=fake_service)


@pytest.fixture
def day11_service(settings: Settings, fake_service: FakeDeepSeekService) -> Day11MemoryService:
    """Day 11 memory service backed by the fake provider and a temp JSON file."""
    return Day11MemoryService(
        settings,
        deepseek=fake_service,
        long_term_store=LongTermMemoryStore(settings.day11_long_term_path),
    )


@pytest.fixture
def day12_service(
    settings: Settings,
    fake_service: FakeDeepSeekService,
    day11_service: Day11MemoryService,
) -> Day12ProfileService:
    """Day 12 profile service sharing the Day 11 memory service."""
    return Day12ProfileService(
        settings,
        memory_service=day11_service,
        profile_store=ProfileStore(settings.day12_profile_path),
    )


@pytest.fixture
def day13_service(
    settings: Settings,
    fake_service: FakeDeepSeekService,
) -> Day13TaskService:
    """Day 13 task service backed by the fake provider and a temp JSON file."""
    return Day13TaskService(
        settings,
        deepseek=fake_service,
        store=TaskStore(settings.day13_task_path),
    )


@pytest.fixture
def day14_service(
    settings: Settings,
    fake_service: FakeDeepSeekService,
) -> Day14InvariantService:
    """Day 14 invariants service backed by the fake provider and a temp file."""
    return Day14InvariantService(
        settings,
        deepseek=fake_service,
        store=InvariantStore(settings.day14_invariants_path),
    )


@pytest.fixture
def day15_service(
    settings: Settings,
    fake_service: FakeDeepSeekService,
) -> Day15LifecycleService:
    """Day 15 lifecycle service backed by the fake provider and a temp file."""
    return Day15LifecycleService(
        settings,
        deepseek=fake_service,
        store=LifecycleStore(settings.day15_task_path),
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
    day8_service: Day8TokenService,
    day9_service: Day9CompressionService,
    day10_service: Day10ContextService,
    day11_service: Day11MemoryService,
    day12_service: Day12ProfileService,
    day13_service: Day13TaskService,
    day14_service: Day14InvariantService,
    day15_service: Day15LifecycleService,
):
    """TestClient with every provider-backed service swapped for a fake."""
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_deepseek_service] = lambda: fake_service
    app.dependency_overrides[get_agent_manager] = lambda: agent_manager
    app.dependency_overrides[get_day8_service] = lambda: day8_service
    app.dependency_overrides[get_day9_service] = lambda: day9_service
    app.dependency_overrides[get_day10_service] = lambda: day10_service
    app.dependency_overrides[get_day11_service] = lambda: day11_service
    app.dependency_overrides[get_day12_service] = lambda: day12_service
    app.dependency_overrides[get_day13_service] = lambda: day13_service
    app.dependency_overrides[get_day14_service] = lambda: day14_service
    app.dependency_overrides[get_day15_service] = lambda: day15_service
    # Starlette 1.x re-raises handled server errors by design; the app ships
    # a global error handler, so capture its response instead.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
