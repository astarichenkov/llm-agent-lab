"""Day 16 — MCP connection & tool discovery tests.

Day 16 is about the MCP PROTOCOL, not business logic: the local Demo MCP
server is spawned over ``stdio``, a session is initialized and ``tools/list``
is really called. No LLM and no network are involved. These tests exercise the
real subprocess so a fake/mock HTTP endpoint cannot satisfy them.

Test 2/3 spawn the real Demo MCP server subprocess; Test 4 points the client at
a broken module and checks the failure is controlled.
"""
from __future__ import annotations

import sys

import pytest

from app.api.routes import get_mcp_client
from app.services.mcp.client import PROJECT_ROOT, MCPClient, MCPServerConfig
from app.services.mcp.demo_server import SERVER_NAME, build_server


# ----------------------------------------------------------------------
# Test 1 — the Demo MCP server registers the expected tools
# ----------------------------------------------------------------------
async def test_demo_server_registers_expected_tools() -> None:
    server = build_server()
    tools = await server.list_tools()
    names = {tool.name for tool in tools}

    assert names == {"echo", "get_server_info"}

    echo = next(tool for tool in tools if tool.name == "echo")
    assert "text" in echo.inputSchema["properties"]
    assert echo.inputSchema["required"] == ["text"]

    info = next(tool for tool in tools if tool.name == "get_server_info")
    assert info.inputSchema["properties"] == {}


# ----------------------------------------------------------------------
# Test 2 — the MCP client really connects, initializes and lists tools
# ----------------------------------------------------------------------
async def test_mcp_client_connect_initialize_and_list_tools() -> None:
    client = MCPClient(timeout_seconds=20)
    result = await client.discover_tools()

    assert result.connected is True
    assert result.error is None
    assert result.server.name == SERVER_NAME
    assert result.server.transport == "stdio"
    assert result.tools_count >= 2
    assert result.tools_count == len(result.tools)

    names = {tool.name for tool in result.tools}
    assert "echo" in names
    assert "get_server_info" in names

    # The trace proves the real stages ran, including tools/list.
    steps = [step.step for step in result.trace]
    assert "connect" in steps
    assert "initialize" in steps
    assert "list_tools" in steps
    assert "closed" in steps
    assert all(step.status == "ok" for step in result.trace)


# ----------------------------------------------------------------------
# Test 3 — the Day 16 API endpoint returns the discovered tools
# ----------------------------------------------------------------------
def test_day16_mcp_status_endpoint(client) -> None:
    response = client.get("/api/week4/day16/mcp/status")
    assert response.status_code == 200
    payload = response.json()

    assert payload["connected"] is True
    assert payload["server"]["transport"] == "stdio"
    assert payload["server"]["name"] == SERVER_NAME
    assert payload["tools_count"] >= 2

    names = {tool["name"] for tool in payload["tools"]}
    assert "echo" in names
    assert "get_server_info" in names

    echo = next(t for t in payload["tools"] if t["name"] == "echo")
    assert "text" in echo["input_schema"]["properties"]
    assert echo["input_schema"]["required"] == ["text"]

    assert any(step["step"] == "list_tools" for step in payload["trace"])


# ----------------------------------------------------------------------
# Test 4 — a broken MCP server yields a controlled error (no stack trace)
# ----------------------------------------------------------------------
def test_day16_endpoint_handles_broken_mcp_server(client, app) -> None:
    broken = MCPClient(
        MCPServerConfig(
            command=sys.executable,
            args=["-m", "app.services.mcp.does_not_exist"],
            cwd=str(PROJECT_ROOT),
        ),
        timeout_seconds=5,
    )
    app.dependency_overrides[get_mcp_client] = lambda: broken
    try:
        response = client.get("/api/week4/day16/mcp/status")
    finally:
        del app.dependency_overrides[get_mcp_client]

    # Controlled response, not a 500 stack trace.
    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is False
    assert payload["tools"] == []
    assert payload["tools_count"] == 0
    assert payload["error"]
    # The trace records the failure stage.
    assert any(
        step["status"] == "error" for step in payload["trace"]
    )


# ----------------------------------------------------------------------
# Test 5 — frontend does not hardcode the MCP tool list
# ----------------------------------------------------------------------
def test_frontend_does_not_hardcode_tools() -> None:
    from pathlib import Path

    js = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "static"
        / "js"
        / "day16.js"
    ).read_text(encoding="utf-8")
    assert "get_server_info" not in js
    assert "/api/week4/day16/mcp/status" in js


@pytest.mark.parametrize("bad_timeout", [0.001])
async def test_mcp_client_timeout_is_controlled(bad_timeout: float) -> None:
    """An unresponsive server must not hang or raise out of the client."""
    client = MCPClient(
        MCPServerConfig(
            command=sys.executable,
            args=["-c", "import time; time.sleep(30)"],
            cwd=str(PROJECT_ROOT),
        ),
        timeout_seconds=bad_timeout,
    )
    result = await client.discover_tools()
    assert result.connected is False
    assert result.error


# ----------------------------------------------------------------------
# Test 6 — UI: only Week 4 is visible; Day 16 is active, Days 17-20 disabled
# ----------------------------------------------------------------------
def test_week4_navigation_and_day16_panel(client) -> None:
    html = client.get("/").text

    assert "Week 4 — MCP" in html
    for day, label in (
        ("day16", "Day 16 — MCP Connection"),
        ("day17", "Day 17"),
        ("day18", "Day 18"),
        ("day19", "Day 19"),
        ("day20", "Day 20"),
    ):
        assert f'id="tab-{day}"' in html
        assert label in html

    # Day 16 is active and its panel exists; future days are disabled.
    import re

    day16_btn = re.search(r'<button[^>]*id="tab-day16"[^>]*>', html)
    assert day16_btn and "disabled" not in day16_btn.group(0)
    assert 'id="panel-day16"' in html
    assert 'src="/static/js/day16.js"' in html
    for day in ("day17", "day18", "day19", "day20"):
        btn = re.search(r'<button[^>]*id="tab-' + day + r'"[^>]*>', html)
        assert btn and "disabled" in btn.group(0), f"{day} must be disabled"


# ----------------------------------------------------------------------
# Test 7 — every day16.js DOM reference exists in the rendered homepage
# ----------------------------------------------------------------------
def test_day16_js_dom_references_exist(client) -> None:
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(
        re.findall(
            r'\$\("([^"]+)"\)',
            (base / "day16.js").read_text(encoding="utf-8"),
        )
    )
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day16.js references missing ids: {missing}"
