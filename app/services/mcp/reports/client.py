"""Day 20 — MCP client for the Reports server (in-memory transport).

Mirrors the Day 16/17/18 ``MCPClient`` surface (``discover_tools`` /
``call_tool`` returning ``MCPStatusResponse`` / ``MCPToolCallResult``) but
connects to the Reports MCP server through the MCP SDK's in-memory transport.
The investigation state lives in the application process, so in-memory is the
right fit.
"""
from __future__ import annotations

from typing import Any

from mcp.shared.memory import create_connected_server_and_client_session

from app.schemas.day16 import (
    MCPServerInfo,
    MCPStatusResponse,
    MCPToolInfo,
    MCPTraceStep,
)
from app.services.mcp.client import MCPToolCallResult, _flatten_error
from app.services.mcp.reports.server import SERVER_NAME, build_reports_server


class ReportsMCPClient:
    """In-process MCP client for the Day 20 Reports tools."""

    def __init__(self, toolset: Any, *, timeout_seconds: float = 30.0) -> None:
        self._toolset = toolset
        self.timeout_seconds = timeout_seconds
        # Set to a dict when a save_report call succeeds (consumed by the
        # Day 20 orchestrator to report the saved location).
        self.last_artifacts: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Orchestrator-managed state
    # ------------------------------------------------------------------
    @property
    def toolset(self) -> Any:
        return self._toolset

    def begin_investigation(self, **kwargs: Any) -> str:
        return self._toolset.begin_investigation(**kwargs)

    def update_context(self, **updates: Any) -> None:
        self._toolset.update_context(**updates)

    def set_masker(self, masker: Any | None) -> None:
        """Forward the opt-in identifier masker to the Reports toolset."""
        self._toolset.set_masker(masker)

    def _build_server(self):
        return build_reports_server(self._toolset)

    async def discover_tools(self) -> MCPStatusResponse:
        """Run a real MCP handshake and ``tools/list`` over in-memory streams."""
        trace: list[MCPTraceStep] = []

        def add(step: str, status: str, message: str) -> None:
            trace.append(MCPTraceStep(step=step, status=status, message=message))

        server = MCPServerInfo(name=SERVER_NAME, transport="in-memory")
        add("connect", "ok", "Connected to Reports MCP (in-memory transport)")
        try:
            async with create_connected_server_and_client_session(
                self._build_server()
            ) as session:
                add("initialize", "ok", "MCP session initialized")
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
            add("closed", "ok", "MCP session closed")
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
                connected=False, server=server, tools_count=0, tools=[],
                trace=trace, error=message,
            )
        except Exception as exc:
            message = _flatten_error(exc)
            add("error", "error", message)
            return MCPStatusResponse(
                connected=False, server=server, tools_count=0, tools=[],
                trace=trace, error=message,
            )

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolCallResult:
        """Perform one real MCP ``tools/call`` over in-memory streams."""
        trace: list[MCPTraceStep] = []

        def add(step: str, status: str, message: str) -> None:
            trace.append(MCPTraceStep(step=step, status=status, message=message))

        add("call_tool", "ok", f"Calling MCP tool: {tool_name}")
        try:
            async with create_connected_server_and_client_session(
                self._build_server()
            ) as session:
                add("initialize", "ok", "MCP session initialized")
                result = await session.call_tool(tool_name, arguments)
            add("result", "ok", f"MCP tool returned: {tool_name}")
            add("closed", "ok", "MCP session closed")

            structured = getattr(result, "structuredContent", None)
            text_parts: list[str] = []
            for item in getattr(result, "content", None) or []:
                text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
            text = "\n".join(text_parts)
            call_result = MCPToolCallResult(
                tool=tool_name,
                is_error=bool(getattr(result, "isError", False)),
                structured=structured if isinstance(structured, dict) else None,
                text=text,
                trace=trace,
                error=(
                    "Tool reported an error"
                    if getattr(result, "isError", False)
                    else None
                ),
            )
            if tool_name == "save_report" and not call_result.is_error:
                payload = call_result.as_payload()
                if isinstance(payload, dict) and payload.get("investigation_id"):
                    self.last_artifacts = payload
            return call_result
        except BaseExceptionGroup as exc:
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


__all__ = ["ReportsMCPClient"]
