"""Day 20 — multi-server MCP orchestration tests.

No test contacts a real VictoriaLogs, Gitea or DeepSeek:

* Gitea HTTP is exercised through ``httpx.MockTransport``;
* the MCP servers are replaced by deterministic fake clients (VictoriaLogs /
  Gitea) plus the REAL in-memory Reports MCP server;
* the LLM is the shared ``FakeDeepSeekService`` scripted via ``tool_sequence``.

The agent loop behaviour (multi-step, result -> next LLM call, limits,
partial failures) is tested, but it is NOT a hardcoded production flow: the
test scripts the model's choices and asserts the orchestrator executes and
routes them.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas.day16 import (
    MCPServerInfo,
    MCPStatusResponse,
    MCPToolInfo,
    MCPTraceStep,
)
from app.schemas.day20 import (
    Day20InvestigateRequest,
    InvestigationReportInput,
    TRACE_STATUS_TIMEOUT,
)
from app.services.day20 import (
    Day20Orchestrator,
    Day20Service,
    InvestigationArtifactError,
    ReportsToolset,
    generate_investigation_id,
    validate_investigation_id,
)
from app.services.day20.orchestrator import safe_arguments
from app.services.gitea.client import (
    GiteaAuthError,
    GiteaClient,
    GiteaConfigError,
    GiteaError,
    GiteaNotFoundError,
    GiteaTimeoutError,
    extract_diff_files,
    parse_commit,
    validate_ref,
    validate_sha,
)
from app.services.mcp import MCPToolCallResult
from app.services.mcp.gitea.server import (
    GET_COMMIT_DIFF_DESCRIPTION,
    GET_COMMIT_DESCRIPTION,
    RECENT_COMMITS_DESCRIPTION,
    SERVER_NAME,
    build_server,
)
from app.services.mcp.registry import (
    MCPServerRegistry,
    ToolCollisionError,
    UnknownToolError,
)
from app.services.mcp.reports.client import ReportsMCPClient


# ----------------------------------------------------------------------
# Fakes / helpers
# ----------------------------------------------------------------------
def _tool(name: str, description: str = "") -> MCPToolInfo:
    return MCPToolInfo(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
    )


VICTORIALOGS_TOOLS = [_tool("search_logs", "Search logs")]
GITEA_TOOLS = [
    _tool("recent_commits", "Recent commits"),
    _tool("get_commit", "Get commit"),
    _tool("get_commit_diff", "Get commit diff"),
]

DEFAULT_RESULTS = {
    "search_logs": {
        "count": 2,
        "limit": 100,
        "truncated": False,
        "logs": [
            {"timestamp": "2026-09-28T23:17:00Z", "message": "upstream timeout", "fields": {}}
        ],
    },
    "recent_commits": {
        "count": 1,
        "commits": [{"sha": "def4567890abcdef", "short_sha": "def4567", "message": "m"}],
    },
    "get_commit": {"sha": "def4567890abcdef", "short_sha": "def4567", "message": "m"},
    "get_commit_diff": {
        "sha": "def4567890abcdef",
        "short_sha": "def4567",
        "files": ["config/http.yaml"],
        "diff": "-timeout: 5s\n+timeout: 1s",
        "truncated": False,
    },
}


class _FakeClient:
    def __init__(
        self,
        name: str,
        tools: list[MCPToolInfo],
        *,
        results: dict | None = None,
        connected: bool = True,
        error: str | None = None,
        delay: float = 0.0,
    ) -> None:
        self.name = name
        self.tools = tools
        self.results = results if results is not None else dict(DEFAULT_RESULTS)
        self.connected = connected
        self.error = error
        self.delay = delay
        self.calls: list[tuple[str, dict]] = []

    async def discover_tools(self) -> MCPStatusResponse:
        return MCPStatusResponse(
            connected=self.connected,
            server=MCPServerInfo(name=self.name, transport="stdio"),
            tools_count=len(self.tools) if self.connected else 0,
            tools=list(self.tools) if self.connected else [],
            trace=[MCPTraceStep(step="initialize", status="ok", message="init")],
            error=self.error,
        )

    async def call_tool(self, tool_name: str, arguments: dict) -> MCPToolCallResult:
        self.calls.append((tool_name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        payload = self.results.get(tool_name, {})
        if callable(payload):
            payload = payload(tool_name, arguments)
        is_error = bool(isinstance(payload, dict) and payload.get("error"))
        return MCPToolCallResult(
            tool=tool_name,
            is_error=is_error,
            structured=payload if isinstance(payload, dict) else None,
            error="tool error" if is_error else None,
        )


def _tool_call(call_id: str, name: str, arguments: dict | None = None) -> dict:
    return {"id": call_id, "name": name, "arguments": arguments or {}}


def _script(*steps: dict) -> list[dict]:
    return list(steps)


def _call_step(call_id: str, name: str, arguments: dict | None = None) -> dict:
    return {"content": "", "tool_calls": [_tool_call(call_id, name, arguments)]}


def _final_step(text: str = "Final investigation answer.") -> dict:
    return {"content": text, "tool_calls": []}


def _registry(settings, *, vl=None, gitea=None, toolset=None):
    toolset = toolset or ReportsToolset(settings)
    registry = MCPServerRegistry()
    registry.register(
        "victorialogs",
        vl or _FakeClient("VictoriaLogs MCP", VICTORIALOGS_TOOLS),
        label="VictoriaLogs MCP",
    )
    registry.register(
        "gitea",
        gitea or _FakeClient("Gitea MCP", GITEA_TOOLS),
        label="Gitea MCP",
    )
    registry.register(
        "reports",
        ReportsMCPClient(toolset),
        label="Reports MCP",
        transport="in-memory",
    )
    return registry, toolset


def _request(**overrides) -> Day20InvestigateRequest:
    base = {"message": "Investigate ERROR application", "service": "application"}
    base.update(overrides)
    return Day20InvestigateRequest(**base)


# ----------------------------------------------------------------------
# 1. Gitea HTTP client
# ----------------------------------------------------------------------
def _gitea(handler, **kwargs) -> GiteaClient:
    return GiteaClient(
        base_url="http://gitea.test:8108",
        token="secret-token-value",
        owner="acme",
        repository="app",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def test_gitea_recent_commits_parsing_and_time_forwarding() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["authorization"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json=[
                {
                    "sha": "abc1234567890abcdef",
                    "created": "2026-09-28T10:05:00Z",
                    "commit": {
                        "message": "Fix timeout\n\nlong body",
                        "author": {"name": "Alice", "date": "2026-09-28T10:05:00Z"},
                    },
                }
            ],
        )

    since = datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc)
    until = datetime(2026, 9, 28, 11, 0, tzinfo=timezone.utc)
    result = await _gitea(handler).recent_commits(
        since=since, until=until, limit=5
    )

    assert seen["path"] == "/api/v1/repos/acme/app/commits"
    assert seen["params"]["since"] == "2026-09-28T10:00:00Z"
    assert seen["params"]["until"] == "2026-09-28T11:00:00Z"
    assert seen["params"]["limit"] == "5"
    assert seen["authorization"] == "token secret-token-value"
    assert result["repository"] == "acme/app"
    assert result["count"] == 1
    assert result["commits"][0]["short_sha"] == "abc1234"
    assert result["commits"][0]["message"] == "Fix timeout"
    assert result["commits"][0]["author"] == "Alice"


async def test_gitea_recent_commits_limit_is_clamped() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    await _gitea(handler, max_commits=7).recent_commits(limit=999)
    assert seen["params"]["limit"] == "7"


async def test_gitea_get_commit_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/repos/acme/app/git/commits/def456"
        return httpx.Response(
            200,
            json={
                "sha": "def4567890abcdef",
                "commit": {
                    "message": "Change upstream timeout",
                    "author": {"name": "Bob", "date": "2026-09-28T23:05:00Z"},
                },
                "parents": [{"sha": "parent111"}],
            },
        )

    result = await _gitea(handler).get_commit("def456")
    assert result["sha"] == "def4567890abcdef"
    assert result["parents"] == ["parent111"]
    assert result["author"] == "Bob"


async def test_gitea_get_commit_diff_truncates() -> None:
    long_diff = "diff --git a/a.py b/a.py\n" + ("+x" * 5000)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/repos/acme/app/git/commits/def456.diff"
        return httpx.Response(200, text=long_diff)

    result = await _gitea(handler, max_diff_chars=1000).get_commit_diff("def456")
    assert result["truncated"] is True
    assert len(result["diff"]) == 1000
    assert result["files"] == ["a.py"]


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, GiteaAuthError),
        (403, GiteaAuthError),
        (404, GiteaNotFoundError),
        (500, GiteaError),
    ],
)
async def test_gitea_http_errors_are_classified(status, expected) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"message": "nope"})

    with pytest.raises(expected):
        await _gitea(handler).recent_commits(limit=1)


async def test_gitea_token_never_appears_in_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    try:
        await _gitea(handler).get_commit("def456")
    except GiteaError as exc:
        assert "secret-token-value" not in str(exc)
    else:  # pragma: no cover - the call must fail
        raise AssertionError("expected GiteaError")


async def test_gitea_timeout_is_classified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(GiteaTimeoutError):
        await _gitea(handler).recent_commits(limit=1)


def test_gitea_requires_configuration() -> None:
    client = GiteaClient(base_url="", owner="", repository="")
    with pytest.raises(GiteaConfigError):
        asyncio.run(client.recent_commits(limit=1))


@pytest.mark.parametrize("bad", ["", "zzzz", "abc 123", "../etc", "deadbeef!"])
def test_gitea_validate_sha_rejects(bad: str) -> None:
    with pytest.raises(GiteaConfigError):
        validate_sha(bad)


def test_gitea_validate_ref_accepts_sha_and_branch() -> None:
    assert validate_ref("main") == "main"
    assert validate_ref("def4567890abcdef") == "def4567890abcdef"
    with pytest.raises(GiteaConfigError):
        validate_ref("bad ref!")


def test_extract_diff_files_preserves_order() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n"
        "diff --git a/b/c.py b/b/c.py\n"
        "diff --git a/a.py b/a.py\n"
    )
    assert extract_diff_files(diff) == ["a.py", "b/c.py"]


# ----------------------------------------------------------------------
# 2. Gitea MCP server exposes read-only tools
# ----------------------------------------------------------------------
async def test_gitea_mcp_server_tools() -> None:
    server = build_server()
    tools = await server.list_tools()
    names = {tool.name for tool in tools}
    assert names == {"recent_commits", "get_commit", "get_commit_diff"}
    for tool in tools:
        assert tool.description
    # The tool set contains NO write operation.
    for forbidden in ("push", "create_commit", "delete", "merge", "edit_file"):
        assert forbidden not in names


def test_gitea_server_metadata_constants() -> None:
    assert SERVER_NAME == "Gitea MCP"
    assert "read-only" in RECENT_COMMITS_DESCRIPTION.lower()
    assert "read-only" in GET_COMMIT_DESCRIPTION.lower()
    assert "read-only" in GET_COMMIT_DIFF_DESCRIPTION.lower()


# ----------------------------------------------------------------------
# 3. Registry / routing
# ----------------------------------------------------------------------
async def test_registry_builds_tool_to_server_map(settings) -> None:
    registry, _ = _registry(settings)
    discovery = await registry.discover_all()
    assert discovery.connected_any is True
    assert registry.tool_map["search_logs"] == "victorialogs"
    assert registry.tool_map["recent_commits"] == "gitea"
    assert registry.tool_map["get_commit"] == "gitea"
    assert registry.tool_map["get_commit_diff"] == "gitea"
    assert registry.tool_map["save_report"] == "reports"


async def test_registry_routes_calls_to_the_right_server(settings) -> None:
    registry, _ = _registry(settings)
    await registry.discover_all()
    vl = registry.get_client("victorialogs")
    gitea = registry.get_client("gitea")

    await registry.call_tool("search_logs", {"service": "application"})
    await registry.call_tool("recent_commits", {"limit": 3})

    assert vl.calls[0][0] == "search_logs"
    assert gitea.calls[0][0] == "recent_commits"


async def test_registry_unknown_tool_is_rejected(settings) -> None:
    registry, _ = _registry(settings)
    await registry.discover_all()
    with pytest.raises(UnknownToolError):
        registry.resolve("does_not_exist")


async def test_registry_detects_duplicate_tool_names(settings) -> None:
    registry = MCPServerRegistry()
    registry.register("a", _FakeClient("A", [_tool("shared_tool")]))
    registry.register("b", _FakeClient("B", [_tool("shared_tool")]))
    discovery = await registry.discover_all()
    assert discovery.collisions
    assert "shared_tool" not in discovery.tool_map
    with pytest.raises(ToolCollisionError):
        raise ToolCollisionError(discovery.collisions)


async def test_registry_one_failed_server_does_not_hide_others(settings) -> None:
    registry = MCPServerRegistry()
    registry.register(
        "gitea",
        _FakeClient("Gitea MCP", GITEA_TOOLS, connected=False, error="down"),
    )
    registry.register(
        "victorialogs", _FakeClient("VictoriaLogs MCP", VICTORIALOGS_TOOLS)
    )
    discovery = await registry.discover_all()
    assert discovery.tool_map.get("search_logs") == "victorialogs"
    assert "recent_commits" not in discovery.tool_map
    assert discovery.connected_any is True


# ----------------------------------------------------------------------
# 4. Reports toolset
# ----------------------------------------------------------------------
async def test_reports_toolset_saves_investigation(settings) -> None:
    toolset = ReportsToolset(settings)
    investigation_id = toolset.begin_investigation(request="investigate")
    result = await toolset.save_report(
        InvestigationReportInput(
            title="Incident", relevant_diff="timeout changed", correlation="change precedes error"
        )
    )
    assert result["investigation_id"] == investigation_id
    run_dir = Path(settings.day20_artifact_root) / investigation_id
    assert (run_dir / "investigation.md").is_file()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["investigation_id"] == investigation_id
    assert "Correlation" in (run_dir / "investigation.md").read_text(encoding="utf-8")
    assert metadata["servers_used"] == []


def test_investigation_id_validation_rejects_traversal() -> None:
    for bad in ("../../escape", "a/b", "..", "inv id"):
        with pytest.raises(InvestigationArtifactError):
            validate_investigation_id(bad)
    assert generate_investigation_id().startswith("inv_")


def test_reports_store_rejects_unknown_filename(settings) -> None:
    toolset = ReportsToolset(settings)
    investigation_id = toolset.begin_investigation(request="x")
    with pytest.raises(InvestigationArtifactError):
        toolset.store.read(investigation_id, "secrets.txt")


# ----------------------------------------------------------------------
# 5. Agent loop / orchestration
# ----------------------------------------------------------------------
async def test_orchestration_multi_step_flow(settings, fake_service) -> None:
    registry, toolset = _registry(settings)
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "application"}),
        _call_step("c2", "recent_commits", {"limit": 5}),
        _call_step("c3", "get_commit_diff", {"sha": "def4567890abcdef"}),
        _call_step("c4", "save_report", {"title": "Incident", "request": "r"}),
        _final_step("Final: correlation is not causation."),
    )
    orchestrator = Day20Orchestrator(settings, registry, deepseek=fake_service)
    response = await orchestrator.investigate(_request())

    assert response.status == "completed"
    assert "correlation is not causation" in response.answer
    assert [call.tool for call in response.tool_calls] == [
        "search_logs",
        "recent_commits",
        "get_commit_diff",
        "save_report",
    ]
    assert response.servers_used == ["victorialogs", "gitea", "reports"]
    assert response.artifacts is not None
    assert response.report_saved is True
    # The following LLM call receives the previous tool result in context.
    second_messages = fake_service.generate_with_tools_calls[1]["messages"]
    tool_messages = [m for m in second_messages if m.get("role") == "tool"]
    assert any("count" in m["content"] for m in tool_messages)
    # The routing really happened against the owning servers.
    assert registry.get_client("victorialogs").calls
    assert registry.get_client("gitea").calls


async def test_orchestration_skips_tools_the_model_does_not_choose(
    settings, fake_service
) -> None:
    registry, _ = _registry(settings)
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "application"}),
        _final_step("Logs only answer."),
    )
    orchestrator = Day20Orchestrator(settings, registry, deepseek=fake_service)
    response = await orchestrator.investigate(_request())

    assert response.status == "completed"
    assert [call.tool for call in response.tool_calls] == ["search_logs"]
    assert response.servers_used == ["victorialogs"]
    assert registry.get_client("gitea").calls == []


async def test_orchestration_iteration_limit_returns_partial(
    settings, fake_service
) -> None:
    registry, _ = _registry(settings)
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "application"}),
        _call_step("c2", "recent_commits", {}),
        _call_step("c3", "get_commit_diff", {"sha": "def4567890abcdef"}),
    )
    orchestrator = Day20Orchestrator(
        settings, registry, deepseek=fake_service, max_tool_iterations=2
    )
    response = await orchestrator.investigate(_request())

    assert response.status == "partial"
    assert "Maximum" in (response.error or "")
    assert len(response.tool_calls) == 2


async def test_orchestration_total_tool_call_cap(settings, fake_service) -> None:
    registry, _ = _registry(settings)
    fake_service.tool_sequence = _script(
        {
            "content": "",
            "tool_calls": [
                _tool_call("c1", "search_logs", {"service": "application"}),
                _tool_call("c2", "recent_commits", {}),
                _tool_call("c3", "get_commit_diff", {"sha": "def4567890abcdef"}),
            ],
        },
        _final_step("partial"),
    )
    orchestrator = Day20Orchestrator(
        settings, registry, deepseek=fake_service, max_total_tool_calls=2
    )
    response = await orchestrator.investigate(_request())

    assert len(response.tool_calls) == 2
    assert response.status == "partial"


async def test_orchestration_tool_timeout_is_handled(settings, fake_service) -> None:
    slow = _FakeClient("VictoriaLogs MCP", VICTORIALOGS_TOOLS, delay=0.2)
    registry, _ = _registry(settings, vl=slow)
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "application"}),
        _final_step("Recovered after timeout."),
    )
    orchestrator = Day20Orchestrator(
        settings, registry, deepseek=fake_service, tool_timeout_seconds=0.01
    )
    response = await orchestrator.investigate(_request())

    assert response.answer == "Recovered after timeout."
    statuses = [step.status for step in response.trace if step.kind == "tool_call"]
    assert TRACE_STATUS_TIMEOUT in statuses


async def test_orchestration_one_server_down_does_not_crash(
    settings, fake_service
) -> None:
    gitea = _FakeClient("Gitea MCP", GITEA_TOOLS, connected=False, error="unavailable")
    registry, _ = _registry(settings, gitea=gitea)
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "application"}),
        _final_step("Logs analysed; Gitea unavailable."),
    )
    orchestrator = Day20Orchestrator(settings, registry, deepseek=fake_service)
    response = await orchestrator.investigate(_request())

    assert response.status == "completed"
    assert response.servers_used == ["victorialogs"]
    assert "Gitea unavailable" in response.answer


async def test_orchestration_unknown_tool_returns_error_to_llm(
    settings, fake_service
) -> None:
    registry, _ = _registry(settings)
    fake_service.tool_sequence = _script(
        _call_step("c1", "made_up_tool", {}),
        _final_step("Recovered."),
    )
    orchestrator = Day20Orchestrator(settings, registry, deepseek=fake_service)
    response = await orchestrator.investigate(_request())

    assert response.answer == "Recovered."
    assert response.tool_calls == []
    error_steps = [s for s in response.trace if s.error]
    assert any("unknown MCP tool" in (s.error or "") for s in error_steps)


async def test_orchestration_reports_context_metadata(settings, fake_service) -> None:
    registry, toolset = _registry(settings)
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "application"}),
        _call_step("c2", "save_report", {"title": "Incident"}),
        _final_step("done"),
    )
    orchestrator = Day20Orchestrator(settings, registry, deepseek=fake_service)
    response = await orchestrator.investigate(_request())

    metadata = json.loads(
        (
            Path(settings.day20_artifact_root)
            / response.investigation_id
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["tools_used"] == ["search_logs", "save_report"]
    assert metadata["tool_calls"] == 2
    assert metadata["status"] == "completed"
    assert "VICTORIA_LOGS" not in json.dumps(metadata)


# ----------------------------------------------------------------------
# 6. Trace argument safety
# ----------------------------------------------------------------------
def test_safe_arguments_bounds_and_masks() -> None:
    safe = safe_arguments(
        {
            "logs": [{"message": "x" * 5000}],
            "authorization": "Bearer supersecret",
            "token": "abc123",
            "long": "y" * 1000,
        }
    )
    assert "supersecret" not in json.dumps(safe)
    assert "abc123" not in json.dumps(safe)
    assert safe["authorization"] == "[REDACTED]"
    assert safe["token"] == "[REDACTED]"
    assert "chars>" in safe["long"]


async def test_trace_never_contains_token(settings, fake_service) -> None:
    registry, _ = _registry(settings)
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "application", "token": "leakme"}),
        _final_step("done"),
    )
    orchestrator = Day20Orchestrator(settings, registry, deepseek=fake_service)
    response = await orchestrator.investigate(_request())
    blob = json.dumps(response.model_dump())
    assert "leakme" not in blob


# ----------------------------------------------------------------------
# 6b. Opt-in identifier masking ("Маскировать данные")
# ----------------------------------------------------------------------
MASKED_ROWS = [
    {
        "timestamp": "2026-09-28T10:00:00Z",
        "message": (
            "upstream https://api.internal.example.com/v1 failed "
            "via db.internal:5432 from 10.0.0.7"
        ),
        "fields": {
            "service": "payment-service",
            "container_name": "payments-7f9c2a",
        },
    }
]


async def test_mask_data_masks_llm_context_and_saved_report(
    settings, fake_service
) -> None:
    vl = _FakeClient(
        "VictoriaLogs MCP",
        VICTORIALOGS_TOOLS,
        results={
            "search_logs": {
                "service": "payment-service",
                "count": 1,
                "limit": 100,
                "truncated": False,
                "logs": MASKED_ROWS,
            }
        },
    )
    registry, _ = _registry(settings, vl=vl)
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "payment-service"}),
        _call_step(
            "c2",
            "save_report",
            {
                "title": "Incident",
                "incident": "payment-service timeout at https://api.internal.example.com",
            },
        ),
        _final_step("Done."),
    )
    orchestrator = Day20Orchestrator(settings, registry, deepseek=fake_service)
    response = await orchestrator.investigate(
        _request(mask_data=True, service="payment-service")
    )
    assert response.status == "completed"

    secrets = (
        "payment-service",
        "payments-7f9c2a",
        "api.internal.example.com",
        "db.internal",
        "10.0.0.7",
    )
    # The masked rows (never the originals) reach the next LLM call.
    second_messages = fake_service.generate_with_tools_calls[1]["messages"]
    tool_messages = [m for m in second_messages if m.get("role") == "tool"]
    tool_blob = json.dumps(tool_messages)
    for secret in secrets:
        assert secret not in tool_blob, secret
    assert "[MASKED]" in tool_blob and "[URL]" in tool_blob

    # Trace arguments are masked too.
    assert response.tool_calls[0].tool == "search_logs"
    trace_blob = json.dumps([s.model_dump() for s in response.trace])
    for secret in secrets:
        assert secret not in trace_blob, secret

    # Saved artifacts contain no original identifiers.
    run_dir = Path(settings.day20_artifact_root) / response.investigation_id
    blob = (run_dir / "investigation.md").read_text(encoding="utf-8")
    for secret in secrets:
        assert secret not in blob, f"{secret} leaked into investigation.md"
    assert "[MASKED]" in blob


async def test_mask_data_disabled_keeps_original_identifiers(
    settings, fake_service
) -> None:
    vl = _FakeClient(
        "VictoriaLogs MCP",
        VICTORIALOGS_TOOLS,
        results={
            "search_logs": {
                "service": "payment-service",
                "count": 1,
                "limit": 100,
                "truncated": False,
                "logs": MASKED_ROWS,
            }
        },
    )
    registry, _ = _registry(settings, vl=vl)
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "payment-service"}),
        _final_step("Done."),
    )
    orchestrator = Day20Orchestrator(settings, registry, deepseek=fake_service)
    response = await orchestrator.investigate(_request(mask_data=False))
    second_messages = fake_service.generate_with_tools_calls[1]["messages"]
    tool_blob = json.dumps([m for m in second_messages if m.get("role") == "tool"])
    assert "payment-service" in tool_blob


# ----------------------------------------------------------------------
# 7. HTTP API
# ----------------------------------------------------------------------
def test_day20_mcp_status_endpoint(client) -> None:
    response = client.get("/api/week4/day20/mcp/status")
    assert response.status_code == 200
    data = response.json()
    assert data["connected"] is True
    assert len(data["servers"]) == 3
    names = {server["name"] for server in data["servers"]}
    assert names == {"victorialogs", "gitea", "reports"}
    routing = {row["tool"]: row["server"] for row in data["routing"]}
    assert routing["search_logs"] == "victorialogs"
    assert routing["recent_commits"] == "gitea"
    assert routing["get_commit_diff"] == "gitea"
    assert routing["save_report"] == "reports"


def test_day20_investigate_endpoint(client, fake_service) -> None:
    fake_service.tool_sequence = _script(
        _call_step("c1", "search_logs", {"service": "application"}),
        _call_step("c2", "recent_commits", {"limit": 5}),
        _call_step("c3", "get_commit_diff", {"sha": "def4567890abcdef"}),
        _call_step("c4", "save_report", {"title": "Incident"}),
        _final_step("Investigation complete."),
    )
    response = client.post(
        "/api/week4/day20/investigate",
        json={"message": "Investigate ERROR application", "service": "application"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["answer"] == "Investigation complete."
    assert data["report_saved"] is True
    assert data["investigation_id"].startswith("inv_")
    assert [call["tool"] for call in data["tool_calls"]] == [
        "search_logs",
        "recent_commits",
        "get_commit_diff",
        "save_report",
    ]

    investigation_id = data["artifacts"]["investigation_id"]
    report = client.get(
        f"/api/week4/day20/investigations/{investigation_id}/artifacts/investigation.md"
    )
    assert report.status_code == 200
    assert "# Incident" in report.text
    metadata = client.get(
        f"/api/week4/day20/investigations/{investigation_id}/artifacts/metadata.json"
    )
    assert metadata.status_code == 200
    assert json.loads(metadata.text)["investigation_id"] == investigation_id

    # Unknown artifact / unknown investigation are rejected.
    assert (
        client.get(
            f"/api/week4/day20/investigations/{investigation_id}/artifacts/secret.txt"
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/week4/day20/investigations/inv_missing/artifacts/metadata.json"
        ).status_code
        == 404
    )


def test_day20_investigate_validation(client) -> None:
    assert (
        client.post("/api/week4/day20/investigate", json={"message": ""}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/week4/day20/investigate",
            json={"message": "x", "since_minutes": 0},
        ).status_code
        == 422
    )


# ----------------------------------------------------------------------
# 8. UI
# ----------------------------------------------------------------------
def test_week4_day20_enabled(client) -> None:
    import re

    html = client.get("/").text
    assert "Day 20 — MCP Orchestration" in html
    assert 'id="panel-day20"' in html
    assert 'src="/static/js/day20.js"' in html
    day20 = re.search(r'<button[^>]*id="tab-day20"[^>]*>', html)
    assert day20 and "disabled" not in day20.group(0)


def test_day20_form_and_ui_elements(client) -> None:
    html = client.get("/").text
    for element in (
        'id="d20-servers"',
        'id="d20-routing"',
        'id="d20-message"',
        'id="d20-mask"',
        'id="d20-run"',
        'id="d20-trace"',
        'id="d20-answer"',
        'id="d20-artifacts"',
        'id="d20-artifact-preview"',
    ):
        assert element in html, element
    assert "Маскировать данные" in html


def test_day20_js_dom_references_exist(client) -> None:
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(
        re.findall(r'\$\("([^"]+)"\)', (base / "day20.js").read_text(encoding="utf-8"))
    )
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day20.js references missing ids: {missing}"


def test_day20_frontend_has_no_secrets() -> None:
    from pathlib import Path

    js = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "static"
        / "js"
        / "day20.js"
    ).read_text(encoding="utf-8")
    assert "/api/week4/day20/investigate" in js
    assert "/api/week4/day20/mcp/status" in js
    assert "GITEA_TOKEN" not in js
    assert "Authorization" not in js
