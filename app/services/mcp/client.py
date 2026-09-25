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

import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

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


def default_server_config() -> MCPServerConfig:
    """Launch ``app.services.mcp.demo_server`` with the current interpreter."""
    return MCPServerConfig(
        command=sys.executable,
        args=["-m", "app.services.mcp.demo_server"],
        cwd=str(PROJECT_ROOT),
    )


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
