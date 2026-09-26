"""Day 17 — the tool-calling agent loop around the VictoriaLogs MCP server.

There is exactly ONE agent loop here and it reuses the existing
``DeepSeekService`` (via the generic tool-calling completion). The flow is::

    discover MCP tools (tools/list)
      -> expose them to the LLM as OpenAI function tools
      -> LLM requests search_logs
      -> MCPClient.call_tool()  (stdio -> VictoriaLogs MCP server -> HTTP API)
      -> MCP result appended to the LLM messages
      -> second LLM turn -> final natural-language answer

The number of tool iterations per user request is capped so a model stuck in
a tool loop is stopped in a controlled way.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.schemas.day16 import MCPStatusResponse, MCPToolInfo
from app.schemas.day17 import (
    Day17ChatResponse,
    Day17ToolCallInfo,
    Day17TraceStep,
)
from app.services.deepseek import DeepSeekError, DeepSeekService
from app.services.mcp import MCPClient, victorialogs_server_config

logger = logging.getLogger("app.services.day17")

# Day 17 really needs ONE tool call; three is a safe upper bound.
MAX_TOOL_ITERATIONS = 3

SERVER_LABEL = "victorialogs"

SYSTEM_PROMPT = (
    "You are an SRE assistant for a staging environment. You have access to "
    "MCP tools. When the user asks about service logs, errors or incidents, "
    "you MUST call the available search_logs tool with structured filters "
    "instead of guessing. Base your answer only on the tool result. If the "
    "tool returns an error, explain it briefly. Answer in the same language "
    "as the user."
)


def _openai_tool(tool: MCPToolInfo) -> dict[str, Any]:
    """Convert one MCP ``tools/list`` entry into an OpenAI function tool."""
    schema = tool.input_schema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


class Day17LogsService:
    """Runs the Day 17 MCP tool-calling flow for one user message."""

    def __init__(
        self,
        settings: Settings,
        *,
        deepseek: Any | None = None,
        mcp_client: MCPClient | None = None,
    ) -> None:
        self._settings = settings
        self._deepseek = deepseek or DeepSeekService(settings)
        self._mcp_client = mcp_client or MCPClient(
            victorialogs_server_config(env=self._server_env(settings))
        )

    @staticmethod
    def _server_env(settings: Settings) -> dict[str, str]:
        """Environment forwarded to the VictoriaLogs MCP subprocess."""
        return {
            "VICTORIA_LOGS_BASE_URL": settings.victoria_logs_base_url,
            "VICTORIA_LOGS_SERVICE_FIELD": settings.victoria_logs_service_field,
            "VICTORIA_LOGS_TIMEOUT_SECONDS": str(
                settings.victoria_logs_timeout_seconds
            ),
            "VICTORIA_LOGS_VERIFY_SSL": (
                "1" if settings.victoria_logs_verify_ssl else "0"
            ),
            "VICTORIA_LOGS_CA_BUNDLE": settings.victoria_logs_ca_bundle,
        }

    async def chat(
        self,
        message: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Day17ChatResponse:
        """Run the full MCP tool-calling turn and return a traceable result.

        When ``start``/``end`` are supplied (the UI interval picker), the exact
        window is placed in the LLM context AND enforced on the actual
        ``search_logs`` call, so the model cannot drift to another range.
        """
        trace: list[Day17TraceStep] = []

        def add(step: str, status: str, message: str) -> None:
            trace.append(Day17TraceStep(step=step, status=status, message=message))

        text = (message or "").strip()
        if not text:
            return Day17ChatResponse(error="Message must not be empty.", trace=trace)

        window_args, window_note = self._time_window(start, end)
        add("user_request", "ok", "Agent received user request")
        if window_args:
            add(
                "time_window",
                "ok",
                f"Time window selected: {window_args['start']} → {window_args['end']}",
            )

        # 1) Real MCP tool discovery over stdio (tools/list).
        status = await self._discover_tools(add)
        if status is None or not status.connected:
            error = "MCP server is not available."
            if status is not None and status.error:
                error = status.error
            add("discover_tools", "error", error)
            return Day17ChatResponse(trace=trace, error=error)

        openai_tools = [_openai_tool(tool) for tool in status.tools]
        allowed = {tool.name for tool in status.tools}

        system_prompt = SYSTEM_PROMPT + ("\n" + window_note if window_note else "")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        tool_calls_out: list[Day17ToolCallInfo] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                content, _finish, _usage, llm_calls = await self._deepseek.generate_with_tools(
                    messages,
                    tools=openai_tools,
                    model=self._settings.deepseek_model,
                    temperature=0.2,
                    max_tokens=1024,
                    thinking=False,
                )
            except DeepSeekError as exc:
                add("llm_error", "error", exc.message)
                return Day17ChatResponse(
                    tool_calls=tool_calls_out, trace=trace, error=exc.message
                )

            if not llm_calls:
                add("llm_final", "ok", "LLM generated final answer")
                return Day17ChatResponse(
                    answer=content or "",
                    tool_calls=tool_calls_out,
                    trace=trace,
                )

            # Enforce the UI-selected window on the actual tool call.
            if window_args:
                for call in llm_calls:
                    if call.get("name") == "search_logs":
                        call["arguments"] = {
                            **(call.get("arguments") or {}),
                            **window_args,
                        }

            selected = ", ".join(call["name"] for call in llm_calls)
            add("llm_select", "ok", f"LLM selected MCP tool(s): {selected}")
            messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in llm_calls
                    ],
                }
            )

            for call in llm_calls:
                await self._run_tool_call(
                    call, allowed, messages, tool_calls_out, add
                )

        add(
            "iteration_limit",
            "error",
            f"Stopped after {MAX_TOOL_ITERATIONS} MCP tool iterations",
        )
        return Day17ChatResponse(
            tool_calls=tool_calls_out,
            trace=trace,
            error="Maximum number of MCP tool calls exceeded.",
        )

    @staticmethod
    def _time_window(
        start: datetime | None, end: datetime | None
    ) -> tuple[dict[str, str], str]:
        """Normalize the chosen window for the prompt and the tool call."""
        if start is None or end is None:
            return {}, ""

        start_utc = (
            start.replace(tzinfo=timezone.utc)
            if start.tzinfo is None
            else start.astimezone(timezone.utc)
        )
        end_utc = (
            end.replace(tzinfo=timezone.utc)
            if end.tzinfo is None
            else end.astimezone(timezone.utc)
        )
        start_iso = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        note = (
            "The user selected this exact time window: start="
            f"{start_iso} end={end_iso}. Pass these exact start and end "
            "values to search_logs (do not change them)."
        )
        return {"start": start_iso, "end": end_iso}, note

    async def _discover_tools(self, add) -> MCPStatusResponse | None:
        """Connect to the MCP server and record real trace steps."""
        try:
            status = await self._mcp_client.discover_tools()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Day 17 MCP discovery failed")
            add("discover_tools", "error", str(exc) or "MCP discovery failed")
            return None
        for step in status.trace:
            if step.step in ("initialize", "list_tools"):
                add(step.step, step.status, step.message)
        if status.connected:
            add(
                "discover_tools",
                "ok",
                f"Discovered {status.tools_count} MCP tools",
            )
        return status

    async def _run_tool_call(
        self,
        call: dict[str, Any],
        allowed: set[str],
        messages: list[dict[str, Any]],
        tool_calls_out: list[Day17ToolCallInfo],
        add,
    ) -> None:
        """Perform one real MCP ``tools/call`` and append its result."""
        name = call.get("name") or ""
        arguments = call.get("arguments") or {}

        if name not in allowed:
            add("call_tool", "error", f"Unknown MCP tool requested: {name}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"error": f"unknown tool: {name}"}),
                }
            )
            return

        add("call_tool", "ok", f"Calling MCP tool: {name}")
        add("victorialogs_request", "ok", "VictoriaLogs request started")

        try:
            result = await self._mcp_client.call_tool(name, arguments)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Day 17 MCP call_tool failed")
            add("call_tool", "error", str(exc) or "MCP call failed")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"error": "MCP call failed"}),
                }
            )
            return

        payload = result.as_payload()
        payload_error = payload.get("error") if isinstance(payload, dict) else None

        if result.is_error or payload_error:
            message = str(payload_error or result.error or "MCP tool error")
            add("call_tool", "error", message)
        else:
            count = payload.get("count") if isinstance(payload, dict) else None
            add(
                "victorialogs_response",
                "ok",
                f"VictoriaLogs returned {count} logs",
            )

        summary_keys = ("count", "limit", "truncated", "query", "error")
        summary = {
            key: payload[key]
            for key in summary_keys
            if isinstance(payload, dict) and key in payload
        }
        tool_calls_out.append(
            Day17ToolCallInfo(
                server=SERVER_LABEL,
                tool=name,
                arguments=arguments,
                result_summary=summary,
            )
        )

        add("mcp_result", "ok", "MCP result returned to LLM")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(payload, ensure_ascii=False),
            }
        )
