"""Shared fixtures for the test suite.

No test in this suite ever touches the real DeepSeek API: every normal
DeepSeek call is replaced either by dependency override (API tests) or by
mocking the OpenAI client (service unit tests). The only exception is the
explicitly-marked integration smoke test (``pytest -m integration``).
"""
import json

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
    get_day17_service,
    get_day18_service,
    get_day19_service,
    get_day20_service,
    get_day22_evaluation_service,
    get_day22_service,
    get_day23_evaluation_service,
    get_day23_service,
    get_day24_evaluation_service,
    get_day24_service,
    get_day25_evaluation_service,
    get_day25_service,
    get_deepseek_service,
    get_mcp_client,
    get_monitoring_service,
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
from app.services.day17 import Day17LogsService
from app.services.day18 import Day18MonitoringAgentService, MonitoringService
from app.services.day19 import Day19PipelineService, Day19Toolset
from app.services.day20 import Day20Service, ReportsToolset
from app.services.rag.answer_service import RagAnswerService
from app.services.rag.day23_evaluation import Day23EvaluationService
from app.services.rag.day24_evaluation import Day24EvaluationService
from app.services.rag.evaluation import EvaluationService
from app.services.rag.grounding.service import GroundedRagService
from app.services.rag.sources.links import SourceLinkResolver
from app.services.day25 import (
    ChatRepository,
    Day25ChatService,
    Day25EvaluationService,
    RuleBasedTaskStateExtractor,
    TaskStateUpdater,
)
from app.services.rag.generation.base import GenerationResult
from app.services.rag.rewriting.base import RewriteResult
from app.services.rag.models import SearchHit
from app.services.mcp import MCPClient, MCPToolCallResult
from app.services.mcp.reports.client import ReportsMCPClient
from app.schemas.day16 import (
    MCPServerInfo,
    MCPStatusResponse,
    MCPToolInfo,
    MCPTraceStep,
)
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
        # Day 17: scripted tool-calling completions.
        self.generate_with_tools_calls: list[dict] = []
        self.tool_sequence: list[dict] = []
        # Day 19: optional queue of plain ``generate`` answers. When set, each
        # call pops one item (e.g. a structured analysis JSON, then a final
        # textual answer). Existing behaviour is unchanged when left empty.
        self.generate_sequence: list[str] = []

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
        if self.generate_sequence:
            return (
                self.generate_sequence.pop(0),
                "stop",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
        return (
            self.answer,
            "stop",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def generate_with_tools(
        self,
        messages: list[dict],
        *,
        tools: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        thinking: bool | None = None,
    ) -> tuple[str, str | None, dict | None, list[dict]]:
        """Mirror DeepSeekService.generate_with_tools using a scripted queue.

        Each ``tool_sequence`` entry is a dict with optional ``content``,
        ``finish_reason`` and ``tool_calls``. When the queue is empty a plain
        final answer (``self.answer``) with no tool calls is returned.
        """
        self.generate_with_tools_calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking": thinking,
            }
        )
        if self.raise_error is not None:
            raise self.raise_error
        if self.tool_sequence:
            item = self.tool_sequence.pop(0)
            return (
                item.get("content", "") or "",
                item.get("finish_reason", "tool_calls"),
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                item.get("tool_calls", []),
            )
        return (
            self.answer,
            "stop",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            [],
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
        day18_monitoring_db_path=str(tmp_path / "day18_monitoring.db"),
        day19_artifact_root=str(tmp_path / "day19_runs"),
        day20_artifact_root=str(tmp_path / "day20_investigations"),
        day25_chat_db_path=str(tmp_path / "day25_chat.sqlite3"),
        day25_scenarios_path=str(tmp_path / "day25_scenarios"),
        day25_eval_results_path=str(tmp_path / "day25_eval_results.json"),
        day25_state_extractor="rules",
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


class FakeMCPClient:
    """Deterministic MCP client double for the Day 17 API tests.

    The real subprocess path is covered separately by test_day17.py; here it
    only has to report a tool list and return a canned tool result.
    """

    def __init__(
        self,
        *,
        connected: bool = True,
        error: str | None = None,
        call_result: dict | None = None,
    ) -> None:
        self.connected = connected
        self.error = error
        self.call_result = call_result or {
            "service": "orders-service",
            "since_minutes": 15,
            "level": "ERROR",
            "query": 'service:"orders-service" AND level:"ERROR"',
            "count": 2,
            "limit": 100,
            "truncated": False,
            "logs": [],
        }
        self.call_tool_calls: list[tuple[str, dict]] = []

    async def discover_tools(self) -> MCPStatusResponse:
        trace = [
            MCPTraceStep(step="initialize", status="ok", message="initialized"),
            MCPTraceStep(step="list_tools", status="ok", message="Received 1 tools"),
        ]
        tools = [
            MCPToolInfo(
                name="search_logs",
                description="Search recent stage logs.",
                input_schema={
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            )
        ]
        return MCPStatusResponse(
            connected=self.connected,
            server=MCPServerInfo(name="VictoriaLogs MCP", transport="stdio"),
            tools_count=len(tools) if self.connected else 0,
            tools=tools if self.connected else [],
            trace=trace,
            error=self.error,
        )

    async def call_tool(self, tool_name: str, arguments: dict):
        self.call_tool_calls.append((tool_name, arguments))
        return MCPToolCallResult(
            tool=tool_name,
            is_error=False,
            structured=self.call_result,
            trace=[MCPTraceStep(step="call_tool", status="ok", message="called")],
        )


@pytest.fixture
def mcp_client() -> MCPClient:
    """Day 16 MCP client using the real local Demo MCP server (stdio)."""
    return MCPClient()


@pytest.fixture
def day17_service(
    settings: Settings, fake_service: FakeDeepSeekService
) -> Day17LogsService:
    """Day 17 service backed by the fake LLM and a fake MCP client."""
    return Day17LogsService(
        settings,
        deepseek=fake_service,
        mcp_client=FakeMCPClient(),
    )


@pytest.fixture
def monitoring_service(settings: Settings) -> MonitoringService:
    """Day 18 monitoring service over a temporary SQLite DB (no scheduler)."""
    return MonitoringService(settings)


class FakeDay19VictoriaLogs:
    """Deterministic stand-in for the Day 17 client used by Day 19 tests."""

    def __init__(self, rows=None, error: Exception | None = None) -> None:
        self.rows = rows if rows is not None else [
            {
                "_time": "2026-09-28T10:00:00Z",
                "_msg": "connection timeout to db",
                "level": "ERROR",
            },
            {
                "_time": "2026-09-28T10:01:00Z",
                "_msg": "502 bad gateway",
                "level": "ERROR",
            },
        ]
        self.error = error
        self.calls: list[dict] = []

    async def search(self, *, query, start, end, limit):
        self.calls.append({"query": query, "start": start, "end": end, "limit": limit})
        if self.error is not None:
            raise self.error
        return list(self.rows), 0


@pytest.fixture
def day19_service(
    settings: Settings, fake_service: FakeDeepSeekService
) -> Day19PipelineService:
    """Day 19 pipeline backed by the fake LLM and a fake VictoriaLogs client."""
    toolset = Day19Toolset(
        settings,
        deepseek=fake_service,
        client_factory=lambda: FakeDay19VictoriaLogs(),
    )
    return Day19PipelineService(settings, toolset=toolset, deepseek=fake_service)


@pytest.fixture
def day18_service(
    settings: Settings,
    fake_service: FakeDeepSeekService,
    monitoring_service: MonitoringService,
) -> Day18MonitoringAgentService:
    """Day 18 agent backed by the fake LLM and a temporary monitoring service."""
    return Day18MonitoringAgentService(
        settings, monitoring_service, deepseek=fake_service
    )


def _day20_tool(name: str, description: str, schema: dict | None = None) -> MCPToolInfo:
    return MCPToolInfo(
        name=name,
        description=description,
        input_schema=schema or {"type": "object", "properties": {}},
    )


DAY20_VICTORIALOGS_TOOLS = [
    _day20_tool("search_logs", "Search recent stage logs in VictoriaLogs.")
]
DAY20_GITEA_TOOLS = [
    _day20_tool("recent_commits", "List recent commits (read-only)."),
    _day20_tool("get_commit", "Get one commit (read-only)."),
    _day20_tool("get_commit_diff", "Get one commit diff (read-only)."),
]

DAY20_DEFAULT_RESULTS = {
    "search_logs": {
        "count": 3,
        "limit": 100,
        "truncated": False,
        "logs": [
            {"timestamp": "2026-09-28T23:17:00Z", "message": "upstream timeout", "fields": {}}
        ],
    },
    "recent_commits": {
        "count": 1,
        "commits": [
            {
                "sha": "def4567890abcdef",
                "short_sha": "def4567",
                "message": "Change upstream timeout",
                "author": "dev",
                "created_at": "2026-09-28T23:05:00Z",
            }
        ],
    },
    "get_commit": {
        "sha": "def4567890abcdef",
        "short_sha": "def4567",
        "message": "Change upstream timeout",
        "author": "dev",
        "created_at": "2026-09-28T23:05:00Z",
        "parents": ["abc1234"],
    },
    "get_commit_diff": {
        "sha": "def4567890abcdef",
        "short_sha": "def4567",
        "files": ["config/http.yaml"],
        "diff": "-timeout: 5s\n+timeout: 1s",
        "truncated": False,
    },
}


class FakeDay20MCPClient:
    """Deterministic MCP client double with a configurable tool list."""

    def __init__(
        self,
        *,
        name: str,
        tools: list[MCPToolInfo],
        results: dict | None = None,
        connected: bool = True,
        error: str | None = None,
    ) -> None:
        self.name = name
        self.tools = list(tools)
        self.results = results if results is not None else dict(DAY20_DEFAULT_RESULTS)
        self.connected = connected
        self.error = error
        self.call_tool_calls: list[tuple[str, dict]] = []

    async def discover_tools(self) -> MCPStatusResponse:
        return MCPStatusResponse(
            connected=self.connected,
            server=MCPServerInfo(name=self.name, transport="stdio"),
            tools_count=len(self.tools) if self.connected else 0,
            tools=list(self.tools) if self.connected else [],
            trace=[
                MCPTraceStep(step="initialize", status="ok", message="initialized"),
                MCPTraceStep(step="list_tools", status="ok", message="listed"),
            ],
            error=self.error,
        )

    async def call_tool(self, tool_name: str, arguments: dict):
        self.call_tool_calls.append((tool_name, arguments))
        payload = self.results.get(tool_name, {})
        if callable(payload):
            payload = payload(tool_name, arguments)
        is_error = bool(isinstance(payload, dict) and payload.get("error"))
        return MCPToolCallResult(
            tool=tool_name,
            is_error=is_error,
            structured=payload if isinstance(payload, dict) else None,
            error=("tool error" if is_error else None),
        )


def make_day20_client_factory(
    *,
    victorialogs: FakeDay20MCPClient | None = None,
    gitea: FakeDay20MCPClient | None = None,
):
    """Return a Day 20 client factory using fake VL/Gitea and a real Reports."""

    def factory(toolset: ReportsToolset) -> dict:
        return {
            "victorialogs": victorialogs
            or FakeDay20MCPClient(
                name="VictoriaLogs MCP", tools=DAY20_VICTORIALOGS_TOOLS
            ),
            "gitea": gitea
            or FakeDay20MCPClient(name="Gitea MCP", tools=DAY20_GITEA_TOOLS),
            "reports": ReportsMCPClient(toolset),
        }

    return factory


@pytest.fixture
def day20_service(
    settings: Settings, fake_service: FakeDeepSeekService
) -> Day20Service:
    """Day 20 service backed by fake MCP clients and the fake LLM."""
    return Day20Service(
        settings,
        deepseek=fake_service,
        client_factory=make_day20_client_factory(),
    )


# ----------------------------------------------------------------------
# Day 22 — first RAG request test doubles (no SQLite, no embeddings, no LLM)
# ----------------------------------------------------------------------
def _day22_hit(score, chunk_id, text, metadata):
    return SearchHit(
        score=score,
        chunk_id=chunk_id,
        text=text,
        metadata=metadata,
        doc_id="doc-" + chunk_id,
    )


class FakeDay22RagService:
    """Deterministic stand-in for ``RagService`` used by Day 22 tests.

    Records every ``search`` call so tests can assert No-RAG never retrieves
    and that Top-K is applied. No SQLite and no embedding provider are used.
    """

    def __init__(self, hits=None, *, exists=True):
        self.hits = list(hits) if hits is not None else [
            _day22_hit(
                0.91,
                "manual-chunk-1",
                "Давление в шинах 205/55R16: 2,1 бар.",
                {
                    "source_type": "manual",
                    "source": "20_XPANDER_RU1.pdf",
                    "page": 238,
                },
            ),
            _day22_hit(
                0.82,
                "telegram-chunk-1",
                "Владелец связал вспучивание пластика с пеной для мойки.",
                {
                    "source_type": "telegram",
                    "source": "result.json",
                    "chat_name": "Xpander Club",
                    "message_ids": [83159, 83165, 83205],
                    "authors": ["Владелец A", "Владелец B"],
                    "date_from": "2026-09-22T12:08:00",
                    "date_to": "2026-09-22T14:20:00",
                },
            ),
        ]
        self.exists = exists
        self.index_path = "data/test/day22/rag_index.sqlite3"
        self.search_calls = []
        # Day 24: queries listed here return deliberately weak hits so the
        # grounding gate can be exercised without a real index.
        self.low_score_queries: set[str] = set()
        # Day 25: when True EVERY query returns weak hits (used to exercise
        # the contextual-query refusal path without matching exact text).
        self.force_low_score = False
        # Day 25 controlled context expansion: chunk_id -> neighbour chunks.
        self.neighbors_map: dict = {}
        self.low_score_hits = [
            _day22_hit(
                0.20,
                "low-chunk-1",
                "Нерелевантный фрагмент вне темы вопроса.",
                {"source_type": "manual", "source": "other.pdf", "page": 1},
            )
        ]

    def neighbors(self, chunk_id, *, radius=1, limit=8):
        return list(self.neighbors_map.get(chunk_id, []))[:limit]

    def index_exists(self):
        return self.exists

    def search(self, query, *, top_k=5, source_type=None):
        self.search_calls.append((query, top_k, source_type))
        if self.force_low_score or query in self.low_score_queries:
            hits = list(self.low_score_hits)
        else:
            hits = self.hits
        if source_type:
            hits = [h for h in hits if h.metadata.get("source_type") == source_type]
        return list(hits)[:top_k]

    def stats(self):
        return {
            "chunks": len(self.hits),
            "index_path": "data/test/day22/rag_index.sqlite3",
            "exists": self.exists,
            "embedding_model": "bge-m3",
            "chunking": "structural",
            "chunks_by_source_type": {"manual": 1, "telegram": 1},
        }


class FakeDay22Generation:
    """Deterministic generation provider recording every prompt it receives."""

    name = "fake"
    model = "fake-generation-model"

    def __init__(self, answer="Fake generated answer."):
        self.answer = answer
        self.calls = []
        self.error = None

    async def generate(
        self, messages, *, temperature=None, max_tokens=None, response_format=None
    ):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        if self.error is not None:
            raise self.error
        return GenerationResult(
            content=self.answer,
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )


class FakeGroundedGeneration:
    """Fake generation provider returning a structured grounded payload."""

    name = "fake-grounded"
    model = "fake-grounded-model"

    def __init__(self, payload=None, *, answer="Fake grounded answer.", error=None):
        self.payload = payload
        self.answer = answer
        self.error = error
        self.calls = []
        self.generate_call_count = 0

    async def generate(
        self, messages, *, temperature=None, max_tokens=None, response_format=None
    ):
        self.generate_call_count += 1
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        if self.error is not None:
            raise self.error
        if self.payload is not None:
            content = (
                self.payload
                if isinstance(self.payload, str)
                else json.dumps(self.payload, ensure_ascii=False)
            )
        else:
            content = json.dumps(
                {
                    "status": "answered",
                    "answer": self.answer,
                    "evidence": [
                        {
                            "chunk_id": "manual-chunk-1",
                            "quote": "Давление в шинах 205/55R16: 2,1 бар.",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        return GenerationResult(
            content=content,
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )


class FakeQueryRewriter:
    """Deterministic query rewriter recording every call (no LLM)."""

    name = "fake"
    model = "fake-generation-model"

    def __init__(self, rewritten="rewritten search query", *, applied=True, error=None):
        self.rewritten = rewritten
        self.applied = applied
        self.error = error
        self.calls = []

    async def rewrite(self, question: str) -> RewriteResult:
        self.calls.append(question)
        # ``rewritten=None`` means passthrough: the retrieval query is the
        # original question. Day 24 fixtures use it so a fake retriever can
        # key on the real question text.
        rewritten = question if self.rewritten is None else self.rewritten
        if self.applied:
            return RewriteResult(
                original_query=question,
                rewritten_query=rewritten,
                applied=True,
                provider=self.name,
                model=self.model,
            )
        return RewriteResult(
            original_query=question,
            rewritten_query=question,
            applied=False,
            error=self.error or "rewrite failed",
            provider=self.name,
            model=self.model,
        )


@pytest.fixture
def day22_rag_service() -> FakeDay22RagService:
    return FakeDay22RagService()


@pytest.fixture
def day22_generation() -> FakeDay22Generation:
    return FakeDay22Generation()


@pytest.fixture
def day22_service(
    settings: Settings,
    day22_rag_service: FakeDay22RagService,
    day22_generation: FakeDay22Generation,
) -> RagAnswerService:
    """Day 22 answer service backed by a fake retriever and fake LLM."""
    return RagAnswerService(
        settings, rag_service=day22_rag_service, generation=day22_generation
    )


_DAY22_EVAL_QUESTIONS = [
    {
        "id": "q01",
        "question": "Давление в шинах?",
        "category": "manual_fact",
        "expected_facts": ["2,1 бар"],
        "expected_sources": [
            {"source_type": "manual", "source": "20_XPANDER_RU1.pdf", "page": 238}
        ],
    },
    {
        "id": "q02",
        "question": "Что обсуждали про пластик?",
        "category": "telegram_experience",
        "expected_facts": ["Пена для мойки"],
        "expected_sources": [
            {"source_type": "telegram", "message_ids": [83159]}
        ],
    },
]

_DAY23_EVAL_QUESTIONS = _DAY22_EVAL_QUESTIONS

_DAY24_NEGATIVE_QUESTIONS = [
    {
        "id": "n01",
        "question": "Как выполняется адаптация вариатора Xpander сканером?",
        "note": "нет процедуры в базе",
        "expected_status": "insufficient_context",
    },
    {
        "id": "n02",
        "question": "Сколько литров масла в двигателе Hyundai Creta?",
        "note": "другая марка",
        "expected_status": "insufficient_context",
    },
]
_DAY24_NEGATIVE_TEXTS = {q["question"] for q in _DAY24_NEGATIVE_QUESTIONS}


@pytest.fixture
def day22_evaluation_service(
    settings: Settings, day22_service: RagAnswerService, tmp_path
) -> EvaluationService:
    """Day 22 evaluation service over a temporary dataset and grade store."""
    dataset = tmp_path / "day22_evaluation_questions.json"
    dataset.write_text(
        json.dumps(_DAY22_EVAL_QUESTIONS, ensure_ascii=False), encoding="utf-8"
    )
    return EvaluationService(
        settings,
        answer_service=day22_service,
        dataset_path=str(dataset),
        results_path=str(tmp_path / "day22_evaluation_results.json"),
    )


@pytest.fixture
def day23_rewriter() -> FakeQueryRewriter:
    return FakeQueryRewriter()


@pytest.fixture
def day23_service(
    settings: Settings,
    day22_rag_service: FakeDay22RagService,
    day22_generation: FakeDay22Generation,
    day23_rewriter: FakeQueryRewriter,
) -> RagAnswerService:
    """Day 23 improved service backed by the Day 22 fakes + a fake rewriter."""
    return RagAnswerService(
        settings,
        rag_service=day22_rag_service,
        generation=day22_generation,
        rewriter=day23_rewriter,
    )


@pytest.fixture
def day23_evaluation_service(
    settings: Settings, day23_service: RagAnswerService, tmp_path
) -> Day23EvaluationService:
    """Day 23 evaluation service over the shared temp dataset and grades."""
    dataset = tmp_path / "day23_evaluation_questions.json"
    dataset.write_text(
        json.dumps(_DAY23_EVAL_QUESTIONS, ensure_ascii=False), encoding="utf-8"
    )
    return Day23EvaluationService(
        settings,
        answer_service=day23_service,
        dataset_path=str(dataset),
        results_path=str(tmp_path / "day23_evaluation_results.json"),
    )


@pytest.fixture
def day24_generation() -> FakeGroundedGeneration:
    return FakeGroundedGeneration()


@pytest.fixture
def day24_answer_service(
    settings: Settings,
    day22_rag_service: FakeDay22RagService,
    day24_generation: FakeGroundedGeneration,
) -> RagAnswerService:
    """Day 24 answer service: passthrough rewrite + weak hits for negatives."""
    day22_rag_service.low_score_queries = set(_DAY24_NEGATIVE_TEXTS)
    return RagAnswerService(
        settings,
        rag_service=day22_rag_service,
        generation=day24_generation,
        rewriter=FakeQueryRewriter(rewritten=None),
    )


@pytest.fixture
def day24_sources(tmp_path):
    """Synthetic manual dir + Telegram export for clickable-source tests."""
    manual_dir = tmp_path / "manual_sources"
    manual_dir.mkdir()
    (manual_dir / "20_XPANDER_RU1.pdf").write_bytes(b"%PDF-1.4 dummy")
    (manual_dir / "doc.pdf").write_bytes(b"%PDF-1.4 dummy")
    export = tmp_path / "telegram_export.json"
    export.write_text(
        json.dumps(
            {
                "id": 1780128600,
                "name": "Xpander Club",
                "type": "public_supergroup",
                "messages": [
                    {
                        "id": 83159,
                        "type": "message",
                        "date": "2026-09-22T12:08:00",
                        "from": "Owner A",
                        "text": "Владелец связал вспучивание пластика с пеной для мойки.",
                    },
                    {
                        "id": 83165,
                        "type": "message",
                        "date": "2026-09-22T12:20:00",
                        "from": "Owner B",
                        "text": "Другое сообщение в обсуждении.",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manual_dir, export


@pytest.fixture
def day24_service(
    settings: Settings, day24_answer_service: RagAnswerService, day24_sources
) -> GroundedRagService:
    manual_dir, export = day24_sources
    resolver = SourceLinkResolver(
        settings,
        manual_root=manual_dir,
        telegram_path=export,
        telegram_link_base="",
    )
    return GroundedRagService(
        settings,
        answer_service=day24_answer_service,
        source_link_resolver=resolver,
    )


@pytest.fixture
def day24_evaluation_service(
    settings: Settings, day24_service: GroundedRagService, tmp_path
) -> Day24EvaluationService:
    dataset = tmp_path / "day24_evaluation_questions.json"
    dataset.write_text(
        json.dumps(_DAY22_EVAL_QUESTIONS, ensure_ascii=False), encoding="utf-8"
    )
    negative = tmp_path / "day24_negative_questions.json"
    negative.write_text(
        json.dumps(_DAY24_NEGATIVE_QUESTIONS, ensure_ascii=False), encoding="utf-8"
    )
    return Day24EvaluationService(
        settings,
        answer_service=day24_service.answer_service,
        grounded_service=day24_service,
        dataset_path=str(dataset),
        results_path=str(tmp_path / "day24_evaluation_results.json"),
        negative_path=str(negative),
    )


_DAY25_SCENARIOS = [
    {
        "id": "scenario_01_diagnostics",
        "title": "Battery diagnostics",
        "initial_goal": "Разобраться, почему машина плохо заводится зимой.",
        "turns": [
            {
                "user": "Хочу разобраться, почему машина плохо заводится зимой.",
                "kind": "goal_change",
                "expect_rag": True,
                "expected_state_contains": ["плохо заводится"],
            },
            {
                "user": "Пробег 82 тысячи.",
                "kind": "fact_update",
                "expect_rag": False,
                "expected_state_contains": ["82000"],
            },
            {
                "user": "А что проверить первым?",
                "kind": "question",
                "expect_rag": True,
                "expect_contextual_terms": ["82000"],
            },
            {
                "user": "Исправляю: пробег не 82, а 92 тысячи.",
                "kind": "correction",
                "expect_rag": False,
                "correction": True,
                "expect_active": {"mileage_km": "92000"},
                "supersedes": ["82000"],
                "expected_state_contains": ["92000"],
            },
        ],
    },
    {
        "id": "scenario_02_maintenance",
        "title": "CVT maintenance",
        "initial_goal": "Разобраться с обслуживанием вариатора.",
        "turns": [
            {
                "user": "Хочу разобраться с обслуживанием вариатора.",
                "kind": "goal_change",
                "expect_rag": True,
                "expected_state_contains": ["вариатора"],
            },
            {
                "user": "Коробка CVT.",
                "kind": "fact_update",
                "expect_rag": False,
                "expected_state_contains": ["CVT"],
            },
        ],
    },
]


@pytest.fixture
def day25_scenarios_dir(tmp_path):
    import json as _json

    directory = tmp_path / "day25_scenarios"
    directory.mkdir(parents=True, exist_ok=True)
    for scenario in _DAY25_SCENARIOS:
        (directory / (scenario["id"] + ".json")).write_text(
            _json.dumps(scenario, ensure_ascii=False), encoding="utf-8"
        )
    return directory


@pytest.fixture
def day25_repository(settings: Settings) -> ChatRepository:
    return ChatRepository(settings.day25_chat_db_path)


@pytest.fixture
def day25_service(
    settings: Settings,
    day24_service: GroundedRagService,
    day25_repository: ChatRepository,
) -> Day25ChatService:
    return Day25ChatService(
        settings,
        repository=day25_repository,
        grounded_service=day24_service,
        updater=TaskStateUpdater(RuleBasedTaskStateExtractor()),
    )


@pytest.fixture
def day25_evaluation_service(
    settings: Settings,
    day25_service: Day25ChatService,
    day25_scenarios_dir,
) -> Day25EvaluationService:
    return Day25EvaluationService(
        settings,
        chat_service=day25_service,
        scenarios_path=str(day25_scenarios_dir),
        results_path=settings.day25_eval_results_path,
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
    mcp_client: MCPClient,
    day17_service: Day17LogsService,
    monitoring_service: MonitoringService,
    day18_service: Day18MonitoringAgentService,
    day19_service: Day19PipelineService,
    day20_service: Day20Service,
    day22_service: RagAnswerService,
    day22_evaluation_service: EvaluationService,
    day23_service: RagAnswerService,
    day23_evaluation_service: Day23EvaluationService,
    day24_service: GroundedRagService,
    day24_evaluation_service: Day24EvaluationService,
    day25_service: Day25ChatService,
    day25_evaluation_service: Day25EvaluationService,
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
    app.dependency_overrides[get_mcp_client] = lambda: mcp_client
    app.dependency_overrides[get_day17_service] = lambda: day17_service
    app.dependency_overrides[get_monitoring_service] = lambda: monitoring_service
    app.dependency_overrides[get_day18_service] = lambda: day18_service
    app.dependency_overrides[get_day19_service] = lambda: day19_service
    app.dependency_overrides[get_day20_service] = lambda: day20_service
    app.dependency_overrides[get_day22_service] = lambda: day22_service
    app.dependency_overrides[get_day22_evaluation_service] = lambda: day22_evaluation_service
    app.dependency_overrides[get_day23_service] = lambda: day23_service
    app.dependency_overrides[get_day23_evaluation_service] = lambda: day23_evaluation_service
    app.dependency_overrides[get_day24_service] = lambda: day24_service
    app.dependency_overrides[get_day24_evaluation_service] = lambda: day24_evaluation_service
    app.dependency_overrides[get_day25_service] = lambda: day25_service
    app.dependency_overrides[get_day25_evaluation_service] = lambda: day25_evaluation_service
    # Starlette 1.x re-raises handled server errors by design; the app ships
    # a global error handler, so capture its response instead.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
