"""Day 16 — MCP client (stdio transport).

This module owns ALL MCP transport concerns so the HTTP layer and the UI never
touch the protocol directly. A single ``MCPClient.discover_tools()`` call:

1. spawns the local Demo MCP server as a subprocess (``stdio``);
2. opens an MCP session and runs the ``initialize`` handshake;
3. calls ``tools/list``;
4. maps the result to the API schema;
5. closes the session and the subprocess (context managers).

Only a local subprocess is used: no ports are bound, no remote server is
contacted and no credentials are required.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
    stdio_client,
)

from app.schemas.day16 import (
    MCPServerInfo,
    MCPStatusResponse,
    MCPToolInfo,
    MCPTraceStep,
)

# Project root (…/app/services/mcp/client.py -> parents[3]). Used as the
# subprocess working directory so ``python -m app.services.mcp.demo_server``
# resolves the ``app`` package both locally and inside the Docker image.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class MCPServerConfig:
    """How to launch the local MCP server (stdio transport)."""

    command: str
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    # Human-readable name shown before the initialize handshake completes.
    name: str = "Week 4 Demo MCP"
    # Extra environment variables for the subprocess. The MCP SDK only
    # inherits a safe allow-list of variables by default, so anything the
    # server needs (e.g. VICTORIA_LOGS_BASE_URL) MUST be passed explicitly.
    env: dict[str, str] | None = None


def default_server_config() -> MCPServerConfig:
    """Launch ``app.services.mcp.demo_server`` with the current interpreter."""
    return MCPServerConfig(
        command=sys.executable,
        args=["-m", "app.services.mcp.demo_server"],
        cwd=str(PROJECT_ROOT),
    )


def victorialogs_server_config(
    env: dict[str, str] | None = None,
) -> MCPServerConfig:
    """Launch the Day 17 VictoriaLogs MCP server with an explicit env.

    The MCP stdio transport only inherits a small allow-list of environment
    variables, so ``VICTORIA_LOGS_*`` must be forwarded explicitly. The
    allow-list is merged first so the child interpreter still has PATH etc.
    """
    merged = get_default_environment()
    if env:
        merged.update(env)
    return MCPServerConfig(
        command=sys.executable,
        args=["-m", "app.services.mcp.victorialogs.server"],
        cwd=str(PROJECT_ROOT),
        name="VictoriaLogs MCP",
        env=merged,
    )


def gitea_server_config(
    env: dict[str, str] | None = None,
) -> MCPServerConfig:
    """Launch the Day 20 Gitea MCP server with an explicit env.

    ``GITEA_*`` variables must be forwarded explicitly for the same reason as
    VictoriaLogs above. The token is only passed to the local subprocess and
    is never logged by the client.
    """
    merged = get_default_environment()
    if env:
        merged.update(env)
    return MCPServerConfig(
        command=sys.executable,
        args=["-m", "app.services.mcp.gitea.server"],
        cwd=str(PROJECT_ROOT),
        name="Gitea MCP",
        env=merged,
    )


@dataclass
class MCPToolCallResult:
    """Normalized outcome of one MCP ``tools/call`` request."""

    tool: str
    is_error: bool
    structured: dict[str, Any] | None = None
    text: str = ""
    trace: list[MCPTraceStep] = field(default_factory=list)
    error: str | None = None

    def as_payload(self) -> dict[str, Any]:
        """Return the tool result in the shape handed to the LLM.

        Prefer the structured result (FastMCP returns one for typed tools);
        otherwise parse the text content as JSON, and finally fall back to a
        plain ``{"text": ...}`` wrapper.
        """
        if self.structured is not None:
            return self.structured
        if self.text:
            try:
                parsed = json.loads(self.text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            return {"text": self.text}
        if self.error:
            return {"error": self.error}
        return {}


def _flatten_error(exc: BaseException) -> str:
    """Return a short message, unwrapping anyio ExceptionGroups."""
    if isinstance(exc, BaseExceptionGroup):  # py3.11+
        parts = [_flatten_error(child) for child in exc.exceptions]
        parts = [p for p in parts if p]
        return "; ".join(parts) if parts else exc.__class__.__name__
    return str(exc) or exc.__class__.__name__


class MCPClient:
    """Spawns a local MCP server, initializes a session and lists its tools.

    A fresh subprocess/session is created on every ``discover_tools()`` call:
    this keeps the demo stateless and makes the "Reconnect / Refresh tools"
    button genuinely re-run the whole MCP handshake.
    """

    def __init__(
        self,
        config: MCPServerConfig | None = None,
        *,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.config = config or default_server_config()
        self.timeout_seconds = timeout_seconds

    async def discover_tools(self) -> MCPStatusResponse:
        """Connect, initialize and return the MCP server's ``tools/list``.

        Never raises for connection/protocol failures: the failure is returned
        as ``connected=False`` with a controlled message and a trace, so the
        HTTP endpoint can render it without leaking a stack trace.
        """
        trace: list[MCPTraceStep] = []

        def add(step: str, status: str, message: str) -> None:
            trace.append(MCPTraceStep(step=step, status=status, message=message))

        server = MCPServerInfo(name=self.config.name, transport="stdio")
        command_line = " ".join([self.config.command, *self.config.args])
        add("start_server", "ok", f"Starting MCP server: {command_line}")

        params = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            cwd=self.config.cwd,
            env=self.config.env,
        )
        read_timeout = timedelta(seconds=self.timeout_seconds)

        try:
            async with stdio_client(params) as (read_stream, write_stream):
                add("connect", "ok", "Connected via stdio")
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=read_timeout
                ) as session:
                    add("initialize", "ok", "Initializing MCP session")
                    init_result = await session.initialize()
                    server = MCPServerInfo(
                        name=init_result.serverInfo.name,
                        transport="stdio",
                    )
                    add(
                        "initialize",
                        "ok",
                        f"MCP session initialized: {server.name} "
                        f"{init_result.serverInfo.version}",
                    )

                    add("list_tools", "ok", "Requesting tools/list")
                    list_result = await session.list_tools()
                    tools = [
                        MCPToolInfo(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=tool.inputSchema or {},
                        )
                        for tool in list_result.tools
                    ]
                    add("list_tools", "ok", f"Received {len(tools)} tools")

            add("closed", "ok", "Connection closed")
            return MCPStatusResponse(
                connected=True,
                server=server,
                tools_count=len(tools),
                tools=tools,
                trace=trace,
            )
        except BaseExceptionGroup as exc:  # anyio task-group failures
            message = _flatten_error(exc)
            add("error", "error", message)
            return MCPStatusResponse(
                connected=False,
                server=server,
                tools_count=0,
                tools=[],
                trace=trace,
                error=message,
            )
        except Exception as exc:  # TimeoutError, OSError, protocol errors…
            message = _flatten_error(exc)
            add("error", "error", message)
            return MCPStatusResponse(
                connected=False,
                server=server,
                tools_count=0,
                tools=[],
                trace=trace,
                error=message,
            )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        """Really call an MCP tool over a fresh stdio session.

        Spawns the server subprocess, runs the ``initialize`` handshake and
        sends ``tools/call``. Connection/protocol failures are returned as a
        controlled ``is_error=True`` result (no stack trace escapes).
        """
        trace: list[MCPTraceStep] = []

        def add(step: str, status: str, message: str) -> None:
            trace.append(MCPTraceStep(step=step, status=status, message=message))

        command_line = " ".join([self.config.command, *self.config.args])
        add("start_server", "ok", f"Starting MCP server: {command_line}")
        add("call_tool", "ok", f"Calling MCP tool: {tool_name}")

        params = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            cwd=self.config.cwd,
            env=self.config.env,
        )
        read_timeout = timedelta(seconds=self.timeout_seconds)

        try:
            async with stdio_client(params) as (read_stream, write_stream):
                add("connect", "ok", "Connected via stdio")
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=read_timeout
                ) as session:
                    await session.initialize()
                    add("initialize", "ok", "MCP session initialized")
                    result = await session.call_tool(tool_name, arguments)
            add("result", "ok", f"MCP tool returned: {tool_name}")
            add("closed", "ok", "Connection closed")

            structured = getattr(result, "structuredContent", None)
            text_parts: list[str] = []
            for item in getattr(result, "content", None) or []:
                text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
            text = "\n".join(text_parts)
            return MCPToolCallResult(
                tool=tool_name,
                is_error=bool(getattr(result, "isError", False)),
                structured=structured if isinstance(structured, dict) else None,
                text=text,
                trace=trace,
                error="Tool reported an error" if getattr(result, "isError", False) else None,
            )
        except BaseExceptionGroup as exc:  # anyio task-group failures
            message = _flatten_error(exc)
            add("error", "error", message)
            return MCPToolCallResult(
                tool=tool_name, is_error=True, trace=trace, error=message
            )
        except Exception as exc:
            message = _flatten_error(exc)
            add("error", "error", message)
            return MCPToolCallResult(
                tool=tool_name, is_error=True, trace=trace, error=message
            )
