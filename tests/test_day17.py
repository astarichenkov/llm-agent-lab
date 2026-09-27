"""Day 17 — VictoriaLogs MCP tool + agent tool-calling tests.

No test contacts the real stage VictoriaLogs. The HTTP layer is exercised
through ``httpx.MockTransport`` and, for the real MCP subprocess path, through
a tiny **local** mock VictoriaLogs HTTP server (``http.server``) so the whole
chain ``MCP client -> stdio subprocess -> HTTP`` is real while the data source
is synthetic.
"""
from __future__ import annotations

import http.server
import json
import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from pydantic import ValidationError

from app.api.routes import get_day17_service
from app.config import Settings
from app.schemas.day16 import (
    MCPServerInfo,
    MCPStatusResponse,
    MCPToolInfo,
    MCPTraceStep,
)
from app.services.day17 import Day17LogsService, MAX_TOOL_ITERATIONS
from app.services.mcp import (
    MCPClient,
    MCPToolCallResult,
    victorialogs_server_config,
)
from app.services.mcp.victorialogs.client import (
    VictoriaLogsClient,
    VictoriaLogsConfigError,
    VictoriaLogsHTTPError,
    VictoriaLogsNetworkError,
    VictoriaLogsTimeoutError,
    normalize_logs,
)
from app.services.mcp.victorialogs.query_builder import (
    QueryBuildError,
    build_logsql,
    quote_value,
    validate_field_name,
)
from app.services.mcp.victorialogs.sanitizer import sanitize_text, sanitize_value
from app.services.mcp.victorialogs.schemas import SearchLogsInput
from app.services.mcp.victorialogs.server import SERVER_NAME, build_server


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _mock_client(handler) -> VictoriaLogsClient:
    return VictoriaLogsClient(
        base_url="http://victorialogs.test:9428",
        transport=httpx.MockTransport(handler),
    )


def _jsonl(*objects: dict) -> bytes:
    return ("\n".join(json.dumps(o) for o in objects) + "\n").encode("utf-8")


class _LocalVictoriaLogs:
    """Minimal local HTTP stand-in for VictoriaLogs ``/select/logsql/query``."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (http.server API)
                outer.requests.append(
                    {
                        "path": self.path,
                        "query": self.headers.get("X-Test-Query"),
                    }
                )
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/stream+json")
                self.send_header("Content-Length", str(len(outer.body)))
                self.end_headers()
                if outer.status < 400:
                    self.wfile.write(outer.body)

            def log_message(self, *args):  # silence test output
                return

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self):
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _FakeMCPClient:
    """In-process MCP client double whose tool call result is controllable."""

    def __init__(self, payload: dict | None = None, *, connected: bool = True):
        self.payload = payload or {
            "service": "orders-service",
            "since_minutes": 30,
            "level": "ERROR",
            "text_contains": None,
            "query": 'service:"orders-service" AND level:"ERROR"',
            "count": 2,
            "limit": 10,
            "truncated": False,
            "logs": [
                {
                    "timestamp": "2026-09-26T10:00:00Z",
                    "message": "boom",
                    "fields": {"level": "ERROR"},
                }
            ],
        }
        self.connected = connected
        self.discover_calls = 0
        self.call_tool_calls: list[tuple[str, dict]] = []

    async def discover_tools(self) -> MCPStatusResponse:
        self.discover_calls += 1
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
            server=MCPServerInfo(name=SERVER_NAME, transport="stdio"),
            tools_count=len(tools) if self.connected else 0,
            tools=tools if self.connected else [],
            trace=[
                MCPTraceStep(
                    step="initialize", status="ok", message="initialized"
                ),
                MCPTraceStep(
                    step="list_tools", status="ok", message="Received 1 tools"
                ),
            ],
            error=None if self.connected else "MCP server is not available.",
        )

    async def call_tool(self, tool_name: str, arguments: dict) -> MCPToolCallResult:
        self.call_tool_calls.append((tool_name, arguments))
        return MCPToolCallResult(
            tool=tool_name, is_error=False, structured=self.payload
        )


def _settings(**overrides) -> Settings:
    base = dict(
        deepseek_api_key="test-key",
        victoria_logs_base_url="",
        victoria_logs_service_field="service",
    )
    base.update(overrides)
    return Settings(**base)


# ----------------------------------------------------------------------
# 1. MCP server registration & schema
# ----------------------------------------------------------------------
async def test_search_logs_is_registered_with_description_and_schema() -> None:
    server = build_server()
    tools = await server.list_tools()
    names = {tool.name for tool in tools}
    assert names == {"search_logs"}

    tool = tools[0]
    assert tool.description
    assert "VictoriaLogs" in tool.description

    props = tool.inputSchema["properties"]
    assert set(props) >= {
        "service",
        "since_minutes",
        "level",
        "text_contains",
        "limit",
    }
    assert tool.inputSchema["required"] == ["service"]
    assert props["since_minutes"]["minimum"] == 1
    assert props["since_minutes"]["maximum"] == 360
    assert props["limit"]["maximum"] == 500


# ----------------------------------------------------------------------
# 2. Input validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"service": "svc", "since_minutes": 0},
        {"service": "svc", "since_minutes": 361},
        {"service": "svc", "limit": 501},
        {"service": "svc", "limit": 0},
        {"service": ""},
    ],
)
def test_search_logs_input_rejects_out_of_range(kwargs) -> None:
    with pytest.raises(ValidationError):
        SearchLogsInput(**kwargs)


def test_search_logs_input_defaults() -> None:
    params = SearchLogsInput(service="orders-service")
    assert params.since_minutes == 15
    assert params.limit == 100
    assert params.level is None
    assert params.text_contains is None


def test_query_builder_rejects_unsafe_field_name() -> None:
    with pytest.raises(QueryBuildError):
        validate_field_name('service" OR *')
    with pytest.raises(QueryBuildError):
        validate_field_name("")


def test_query_builder_escapes_values() -> None:
    query = build_logsql(
        service='orders" OR *',
        service_field="service",
        level="ERROR",
        text_contains='timeout" OR "x',
    )
    # The injected quote is escaped, so it cannot terminate the phrase.
    assert query.startswith('service:"orders\\" OR *"')
    assert 'level:"ERROR"' in query
    assert '\\" OR \\"x' in query
    # No unescaped closing quote before the end of the service phrase.
    assert quote_value('a"b\\c') == '"a\\"b\\\\c"'


def test_query_builder_service_only() -> None:
    assert build_logsql(service="orders-service") == 'service:"orders-service"'


# ----------------------------------------------------------------------
# 3. VictoriaLogs HTTP client
# ----------------------------------------------------------------------
async def test_client_builds_endpoint_and_passes_parameters() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, content=_jsonl({"_time": "t", "_msg": "m"}))

    client = _mock_client(handler)
    start = datetime(2026, 9, 26, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 9, 26, 9, 30, tzinfo=timezone.utc)
    rows, malformed = await client.search(
        query='service:"x"', start=start, end=end, limit=25
    )

    assert captured["path"] == "/select/logsql/query"
    assert captured["params"]["query"] == 'service:"x"'
    assert captured["params"]["start"] == "2026-09-26T09:00:00Z"
    assert captured["params"]["end"] == "2026-09-26T09:30:00Z"
    assert captured["params"]["limit"] == "25"
    assert captured["params"]["timeout"].endswith("s")
    assert rows == [{"_time": "t", "_msg": "m"}]
    assert malformed == 0


async def test_client_uses_base_url_with_or_without_trailing_slash() -> None:
    for base in ("http://vl.test:9428", "http://vl.test:9428/"):
        client = VictoriaLogsClient(
            base_url=base,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"")
            ),
        )
        assert client.config.query_url() == "http://vl.test:9428/select/logsql/query"


async def test_client_parses_jsonl_and_skips_blank_and_malformed_lines() -> None:
    body = (
        '{"_time":"t1","_msg":"one"}\n'
        "\n"
        "not-json\n"
        '{"_time":"t2","_msg":"two"}\n'
        "[1,2,3]\n"
    ).encode()
    client = _mock_client(lambda request: httpx.Response(200, content=body))
    rows, malformed = await client.search(
        query="*",
        start=datetime.now(timezone.utc),
        end=datetime.now(timezone.utc),
        limit=100,
    )
    assert [r["_msg"] for r in rows] == ["one", "two"]
    assert malformed == 2  # "not-json" and the JSON array


async def test_client_empty_response() -> None:
    client = _mock_client(lambda request: httpx.Response(200, content=b"\n\n"))
    rows, malformed = await client.search(
        query="*",
        start=datetime.now(timezone.utc),
        end=datetime.now(timezone.utc),
        limit=100,
    )
    assert rows == []
    assert malformed == 0


async def test_client_http_error_is_controlled() -> None:
    client = _mock_client(
        lambda request: httpx.Response(500, content=b"boom detail")
    )
    with pytest.raises(VictoriaLogsHTTPError) as exc:
        await client.search(
            query="*",
            start=datetime.now(timezone.utc),
            end=datetime.now(timezone.utc),
            limit=10,
        )
    assert exc.value.status_code == 500
    assert "boom" in exc.value.message


async def test_client_timeout_is_controlled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    client = _mock_client(handler)
    with pytest.raises(VictoriaLogsTimeoutError):
        await client.search(
            query="*",
            start=datetime.now(timezone.utc),
            end=datetime.now(timezone.utc),
            limit=10,
        )


async def test_client_network_error_is_controlled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns failure")

    client = _mock_client(handler)
    with pytest.raises(VictoriaLogsNetworkError):
        await client.search(
            query="*",
            start=datetime.now(timezone.utc),
            end=datetime.now(timezone.utc),
            limit=10,
        )


async def test_client_missing_base_url_is_lazy_config_error() -> None:
    # No URL -> constructing is fine, using it fails with a controlled error.
    client = VictoriaLogsClient(base_url="", env={})
    with pytest.raises(VictoriaLogsConfigError) as exc:
        await client.search(
            query="*",
            start=datetime.now(timezone.utc),
            end=datetime.now(timezone.utc),
            limit=10,
        )
    assert "VICTORIA_LOGS_BASE_URL" in exc.value.message


# ----------------------------------------------------------------------
# 4. Normalization & sanitization
# ----------------------------------------------------------------------
def test_normalize_logs_maps_time_and_msg() -> None:
    rows = [
        {
            "_time": "2026-09-26T10:00:00Z",
            "_msg": "hello",
            "level": "INFO",
            "service": "orders-service",
        }
    ]
    logs = normalize_logs(rows, sanitize=sanitize_value)
    assert logs[0]["timestamp"] == "2026-09-26T10:00:00Z"
    assert logs[0]["message"] == "hello"
    assert logs[0]["fields"]["level"] == "INFO"
    assert "_time" not in logs[0]["fields"]
    assert "_msg" not in logs[0]["fields"]


@pytest.mark.parametrize(
    "raw",
    [
        "Authorization: Bearer super-secret-token",
        "Bearer super-secret-token",
        "password=hunter2",
        "token=abc123",
        "api_key=xyz",
        "apikey=xyz",
        "Cookie: session=topsecret",
        "level=ERROR password='p@ss token=xyz'",
    ],
)
def test_sanitizer_masks_secrets(raw: str) -> None:
    cleaned = sanitize_text(raw)
    assert "[REDACTED]" in cleaned
    for secret in ("super-secret-token", "hunter2", "abc123", "xyz", "topsecret", "p@ss"):
        assert secret not in cleaned


def test_sanitizer_recurses_into_fields() -> None:
    payload = {
        "message": "password=secret",
        "nested": {"token": "tok", "list": ["api_key=key", "plain"]},
    }
    cleaned = sanitize_value(payload)
    assert cleaned["message"] == "password=[REDACTED]"
    assert cleaned["nested"]["token"] == "[REDACTED]"
    assert cleaned["nested"]["list"][0] == "api_key=[REDACTED]"
    assert sanitize_value({"authorization": "Bearer abcdef123456"}) == {
        "authorization": "[REDACTED]"
    }
    assert cleaned["nested"]["list"][1] == "plain"


# ----------------------------------------------------------------------
# 5. Real local MCP subprocess + local mock VictoriaLogs HTTP
# ----------------------------------------------------------------------
async def test_real_mcp_subprocess_calls_search_logs_end_to_end() -> None:
    body = _jsonl(
        {"_time": "2026-09-26T10:00:00Z", "_msg": "boom", "level": "ERROR"},
        {"_time": "2026-09-26T10:01:00Z", "_msg": "bang", "level": "ERROR"},
    )
    with _LocalVictoriaLogs(body) as mock:
        env = {
            "VICTORIA_LOGS_BASE_URL": mock.base_url,
            "VICTORIA_LOGS_SERVICE_FIELD": "service",
            "VICTORIA_LOGS_VERIFY_SSL": "0",
            "VICTORIA_LOGS_TIMEOUT_SECONDS": "5",
        }
        client = MCPClient(
            victorialogs_server_config(env=env), timeout_seconds=30
        )
        status = await client.discover_tools()
        assert status.connected is True
        assert status.server.name == SERVER_NAME
        assert {t.name for t in status.tools} == {"search_logs"}

        result = await client.call_tool(
            "search_logs",
            {
                "service": "orders-service",
                "since_minutes": 15,
                "level": "ERROR",
                "limit": 10,
            },
        )
        assert result.is_error is False
        payload = result.as_payload()
        assert payload["count"] == 2
        assert payload["query"] == 'service:"orders-service" AND level:"ERROR"'
        assert payload["logs"][0]["message"] == "boom"
        # The local mock really received a bounded query.
        assert mock.requests, "mock VictoriaLogs was not called"
        assert "query=" in mock.requests[0]["path"]
        assert "limit=10" in mock.requests[0]["path"]


async def test_real_mcp_subprocess_reports_missing_configuration() -> None:
    # No VICTORIA_LOGS_BASE_URL in the forwarded env: the tool must return a
    # controlled configuration error, not crash the subprocess.
    env = {
        "VICTORIA_LOGS_BASE_URL": "",
        "VICTORIA_LOGS_VERIFY_SSL": "0",
    }
    client = MCPClient(victorialogs_server_config(env=env), timeout_seconds=30)
    result = await client.call_tool(
        "search_logs", {"service": "orders-service", "since_minutes": 15}
    )
    payload = result.as_payload()
    assert "not configured" in payload["error"]
    assert payload["error_kind"] == "config"


# ----------------------------------------------------------------------
# 6. Agent flow (fake LLM selects the MCP tool)
# ----------------------------------------------------------------------
async def test_agent_flow_tool_result_reaches_second_llm_call() -> None:
    class FakeLLM:
        def __init__(self):
            self.calls: list[list[dict]] = []

        async def generate_with_tools(self, messages, **kwargs):
            self.calls.append(json.loads(json.dumps(messages)))
            if len(self.calls) == 1:
                return (
                    "",
                    "tool_calls",
                    {"total_tokens": 5},
                    [
                        {
                            "id": "call_1",
                            "name": "search_logs",
                            "arguments": {
                                "service": "orders-service",
                                "since_minutes": 30,
                                "level": "ERROR",
                                "limit": 10,
                            },
                        }
                    ],
                )
            return ("Найдено 2 ошибки в orders-service.", "stop", {}, [])

    llm = FakeLLM()
    mcp = _FakeMCPClient()
    service = Day17LogsService(
        _settings(victoria_logs_base_url="http://vl.test"), deepseek=llm, mcp_client=mcp
    )
    response = await service.chat("Покажи ERROR для orders-service за 30 минут")

    assert response.error is None
    assert "2" in response.answer
    assert len(mcp.call_tool_calls) == 1
    assert mcp.call_tool_calls[0][0] == "search_logs"
    assert mcp.call_tool_calls[0][1]["service"] == "orders-service"

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.server == "victorialogs"
    assert call.tool == "search_logs"
    assert call.result_summary["count"] == 2

    # The tool result really reached the SECOND LLM call as a `tool` message.
    second_messages = llm.calls[1]
    tool_messages = [m for m in second_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert '"count": 2' in tool_messages[0]["content"]
    assert tool_messages[0]["tool_call_id"] == "call_1"

    steps = [step.step for step in response.trace]
    assert "discover_tools" in steps
    assert "llm_select" in steps
    assert "call_tool" in steps
    assert "victorialogs_response" in steps
    assert "llm_final" in steps


async def test_agent_flow_stops_after_max_iterations() -> None:
    class LoopingLLM:
        async def generate_with_tools(self, messages, **kwargs):
            return (
                "",
                "tool_calls",
                {},
                [
                    {
                        "id": "call_x",
                        "name": "search_logs",
                        "arguments": {"service": "orders-service"},
                    }
                ],
            )

    mcp = _FakeMCPClient()
    service = Day17LogsService(
        _settings(), deepseek=LoopingLLM(), mcp_client=mcp
    )
    response = await service.chat("loop please")

    assert response.error is not None
    assert len(mcp.call_tool_calls) == MAX_TOOL_ITERATIONS
    assert any(step.step == "iteration_limit" for step in response.trace)


async def test_agent_flow_controls_deepseek_error() -> None:
    from app.services.deepseek import DeepSeekNetworkError

    class BrokenLLM:
        async def generate_with_tools(self, messages, **kwargs):
            raise DeepSeekNetworkError()

    service = Day17LogsService(
        _settings(), deepseek=BrokenLLM(), mcp_client=_FakeMCPClient()
    )
    response = await service.chat("logs?")
    assert response.error
    assert any(step.status == "error" for step in response.trace)


async def test_agent_flow_handles_mcp_unavailable() -> None:
    service = Day17LogsService(
        _settings(),
        deepseek=object(),
        mcp_client=_FakeMCPClient(connected=False),
    )
    response = await service.chat("logs?")
    assert response.error
    assert response.tool_calls == []
    assert any(step.status == "error" for step in response.trace)


# ----------------------------------------------------------------------
# 7. API endpoint
# ----------------------------------------------------------------------
def test_day17_endpoint_with_fake_service(client) -> None:
    response = client.post(
        "/api/week4/day17/chat",
        json={"message": "Покажи ERROR для orders-service"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "trace" in payload


def test_day17_endpoint_rejects_empty_message(client) -> None:
    response = client.post("/api/week4/day17/chat", json={"message": ""})
    assert response.status_code == 422


def test_day17_endpoint_surfaces_tool_call(client, app, settings) -> None:
    from conftest import FakeDeepSeekService

    llm = FakeDeepSeekService(answer="Финальный ответ по логам.")
    llm.tool_sequence = [
        {
            "content": "",
            "finish_reason": "tool_calls",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search_logs",
                    "arguments": {"service": "orders-service", "level": "ERROR"},
                }
            ],
        },
        {"content": "Финальный ответ по логам.", "finish_reason": "stop", "tool_calls": []},
    ]
    service = Day17LogsService(
        settings, deepseek=llm, mcp_client=_FakeMCPClient()
    )
    app.dependency_overrides[get_day17_service] = lambda: service
    try:
        response = client.post(
            "/api/week4/day17/chat",
            json={"message": "Покажи ERROR для orders-service"},
        )
    finally:
        app.dependency_overrides.pop(get_day17_service, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Финальный ответ по логам."
    assert payload["tool_calls"][0]["tool"] == "search_logs"
    assert payload["tool_calls"][0]["server"] == "victorialogs"
    assert any(step["step"] == "call_tool" for step in payload["trace"])


# ----------------------------------------------------------------------
# 8. UI
# ----------------------------------------------------------------------
def test_week4_day17_enabled_and_future_days_disabled(client) -> None:
    import re

    html = client.get("/").text
    assert "Day 17 — VictoriaLogs MCP Tool" in html
    assert 'id="panel-day17"' in html
    assert 'src="/static/js/day17.js"' in html

    day17 = re.search(r'<button[^>]*id="tab-day17"[^>]*>', html)
    assert day17 and "disabled" not in day17.group(0)
    # Days 16-19 are implemented and enabled; only Day 20 stays disabled.
    for day in ("day16", "day18", "day19"):
        btn = re.search(r'<button[^>]*id="tab-' + day + r'"[^>]*>', html)
        assert btn and "disabled" not in btn.group(0), f"{day} must be enabled"
    day20 = re.search(r'<button[^>]*id="tab-day20"[^>]*>', html)
    assert day20 and "disabled" in day20.group(0)


def test_day17_js_dom_references_exist(client) -> None:
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(
        re.findall(
            r'\$\("([^"]+)"\)',
            (base / "day17.js").read_text(encoding="utf-8"),
        )
    )
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day17.js references missing ids: {missing}"


def test_day17_frontend_does_not_hardcode_logs() -> None:
    from pathlib import Path

    js = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "static"
        / "js"
        / "day17.js"
    ).read_text(encoding="utf-8")
    assert "/api/week4/day17/chat" in js
    # No service name / stage URL / hardcoded log content in the frontend.
    assert "orders-service" not in js
    assert "VICTORIA_LOGS" not in js


# ----------------------------------------------------------------------
# 9. Explicit start/end window (UI interval picker)
# ----------------------------------------------------------------------
def test_search_logs_input_accepts_explicit_window() -> None:
    params = SearchLogsInput(
        service="orders-service",
        start=datetime(2026, 9, 26, 8, 0, tzinfo=timezone.utc),
        end=datetime(2026, 9, 26, 9, 0, tzinfo=timezone.utc),
    )
    assert params.start is not None and params.end is not None


@pytest.mark.parametrize(
    "start,end",
    [
        (datetime(2026, 9, 26, 9, 0), datetime(2026, 9, 26, 8, 0)),
        (datetime(2026, 9, 26, 0, 0), datetime(2026, 9, 27, 1, 0)),
    ],
)
def test_search_logs_input_rejects_bad_window(start, end) -> None:
    with pytest.raises(ValidationError):
        SearchLogsInput(service="svc", start=start, end=end)


def test_search_logs_input_rejects_partial_window() -> None:
    with pytest.raises(ValidationError):
        SearchLogsInput(service="svc", start=datetime(2026, 9, 26, 8, 0))


async def test_real_mcp_subprocess_uses_explicit_window() -> None:
    body = _jsonl({"_time": "2026-09-26T08:30:00Z", "_msg": "ok", "level": "INFO"})
    with _LocalVictoriaLogs(body) as mock:
        env = {
            "VICTORIA_LOGS_BASE_URL": mock.base_url,
            "VICTORIA_LOGS_SERVICE_FIELD": "service",
            "VICTORIA_LOGS_VERIFY_SSL": "0",
        }
        client = MCPClient(victorialogs_server_config(env=env), timeout_seconds=30)
        result = await client.call_tool(
            "search_logs",
            {
                "service": "orders-service",
                "start": "2026-09-26T08:00:00Z",
                "end": "2026-09-26T09:00:00Z",
                "limit": 5,
            },
        )
        assert result.is_error is False
        payload = result.as_payload()
        assert payload["start"] == "2026-09-26T08:00:00Z"
        assert payload["end"] == "2026-09-26T09:00:00Z"
        assert mock.requests, "mock VictoriaLogs was not called"
        path = mock.requests[0]["path"]
        assert "start=2026-09-26T08%3A00%3A00Z" in path or "start=2026-09-26T08:00:00Z" in path
        assert "end=2026-09-26T09" in path


async def test_agent_flow_enforces_selected_window() -> None:
    class FakeLLM:
        def __init__(self):
            self.calls: list[list[dict]] = []

        async def generate_with_tools(self, messages, **kwargs):
            self.calls.append(json.loads(json.dumps(messages)))
            if len(self.calls) == 1:
                return (
                    "",
                    "tool_calls",
                    {},
                    [
                        {
                            "id": "call_1",
                            "name": "search_logs",
                            "arguments": {"service": "orders-service"},
                        }
                    ],
                )
            return ("готово", "stop", {}, [])

    llm = FakeLLM()
    mcp = _FakeMCPClient()
    service = Day17LogsService(
        _settings(), deepseek=llm, mcp_client=mcp
    )
    start = datetime(2026, 9, 26, 8, 0, tzinfo=timezone.utc)
    end = datetime(2026, 9, 26, 9, 0, tzinfo=timezone.utc)
    response = await service.chat("Покажи ERROR", start=start, end=end)

    assert response.error is None
    _, arguments = mcp.call_tool_calls[0]
    assert arguments["start"] == "2026-09-26T08:00:00Z"
    assert arguments["end"] == "2026-09-26T09:00:00Z"
    # The window is also stated in the LLM context.
    assert "2026-09-26T08:00:00Z" in llm.calls[0][0]["content"]
    assert any(step.step == "time_window" for step in response.trace)


def test_day17_api_accepts_window(client, app, settings) -> None:
    from conftest import FakeDeepSeekService

    llm = FakeDeepSeekService(answer="ok")
    service = Day17LogsService(settings, deepseek=llm, mcp_client=_FakeMCPClient())
    app.dependency_overrides[get_day17_service] = lambda: service
    try:
        response = client.post(
            "/api/week4/day17/chat",
            json={
                "message": "logs",
                "start": "2026-09-26T08:00:00Z",
                "end": "2026-09-26T09:00:00Z",
            },
        )
        assert response.status_code == 200
        bad = client.post(
            "/api/week4/day17/chat",
            json={
                "message": "logs",
                "start": "2026-09-26T09:00:00Z",
                "end": "2026-09-26T08:00:00Z",
            },
        )
        assert bad.status_code == 422
    finally:
        app.dependency_overrides.pop(get_day17_service, None)
