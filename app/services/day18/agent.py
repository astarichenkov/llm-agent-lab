"""Day 18 — the monitoring agent (MCP tool-calling loop).

One agent loop, reusing the existing ``DeepSeekService.generate_with_tools``.
The LLM is given ONLY the Day 18 monitoring MCP tools (no Day 17 search_logs,
no unrelated tools) and decides which one to call::

    discover monitoring MCP tools (tools/list)
      -> expose them to the LLM as OpenAI function tools
      -> LLM requests start_monitoring / get_monitoring_summary / ...
      -> MonitoringMCPClient.call_tool() (in-memory MCP -> MonitoringService)
      -> MCP result appended to the LLM messages
      -> second LLM turn -> final natural-language answer

There is NO LLM call inside the scheduled runs — this loop only runs when the
user sends a chat message.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings
from app.schemas.day16 import MCPStatusResponse, MCPToolInfo
from app.schemas.day18 import (
    Day18ChatResponse,
    Day18ToolCallInfo,
    Day18TraceStep,
)
from app.services.deepseek import DeepSeekError, DeepSeekService
from app.services.day18.service import MonitoringService
from app.services.mcp.monitoring.client import MonitoringMCPClient

logger = logging.getLogger("app.services.day18.agent")

# A single monitoring action is expected; a small cap prevents tool loops.
MAX_TOOL_ITERATIONS = 3

SERVER_LABEL = "monitoring"

SYSTEM_PROMPT = (
    "You are an SRE monitoring assistant. You have MCP tools to start, "
    "inspect and stop recurring log-monitoring jobs for services in "
    "VictoriaLogs. When the user asks to monitor a service periodically, "
    "call start_monitoring with a supported interval. When asked for a "
    "summary/history, call get_monitoring_summary; for the current state call "
    "get_monitoring_status; to stop, call stop_monitoring. Base your answer "
    "only on the tool results and mention the job id. Answer in the same "
    "language as the user."
)

# Keys copied into the trace/result summary (never the raw log rows).
_SUMMARY_KEYS = (
    "job_id",
    "status",
    "service",
    "level",
    "interval_seconds",
    "lookback_minutes",
    "next_run_at",
    "last_run_at",
    "runs_count",
    "runs",
    "successful_runs",
    "failed_runs",
    "total_logs",
    "average_logs_per_run",
    "min_logs_per_run",
    "max_logs_per_run",
    "last_run_logs",
    "trend",
    "count",
    "error",
    "error_kind",
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


class Day18MonitoringAgentService:
    """Runs the Day 18 monitoring MCP tool-calling flow for one message."""

    def __init__(
        self,
        settings: Settings,
        monitoring: MonitoringService,
        *,
        deepseek: Any | None = None,
        mcp_client: MonitoringMCPClient | None = None,
    ) -> None:
        self._settings = settings
        self._monitoring = monitoring
        self._deepseek = deepseek or DeepSeekService(settings)
        self._mcp_client = mcp_client or MonitoringMCPClient(monitoring)

    async def chat(self, message: str) -> Day18ChatResponse:
        trace: list[Day18TraceStep] = []

        def add(step: str, status: str, message: str) -> None:
            trace.append(Day18TraceStep(step=step, status=status, message=message))

        text = (message or "").strip()
        if not text:
            return Day18ChatResponse(error="Message must not be empty.", trace=trace)

        add("user_request", "ok", "Agent received user request")

        status = await self._discover_tools(add)
        if status is None or not status.connected:
            error = "Monitoring MCP server is not available."
            if status is not None and status.error:
                error = status.error
            add("discover_tools", "error", error)
            return Day18ChatResponse(trace=trace, error=error)

        openai_tools = [_openai_tool(tool) for tool in status.tools]
        allowed = {tool.name for tool in status.tools}

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        tool_calls_out: list[Day18ToolCallInfo] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                content, _finish, _usage, llm_calls = (
                    await self._deepseek.generate_with_tools(
                        messages,
                        tools=openai_tools,
                        model=self._settings.deepseek_model,
                        temperature=0.2,
                        max_tokens=1024,
                        thinking=False,
                    )
                )
            except DeepSeekError as exc:
                add("llm_error", "error", exc.message)
                return Day18ChatResponse(
                    tool_calls=tool_calls_out, trace=trace, error=exc.message
                )

            if not llm_calls:
                add("llm_final", "ok", "LLM generated final answer")
                return Day18ChatResponse(
                    answer=content or "",
                    tool_calls=tool_calls_out,
                    trace=trace,
                )

            selected = ", ".join(call["name"] for call in llm_calls)
            add("llm_select", "ok", f"LLM selected monitoring tool(s): {selected}")
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
        return Day18ChatResponse(
            tool_calls=tool_calls_out,
            trace=trace,
            error="Maximum number of MCP tool calls exceeded.",
        )

    async def _discover_tools(self, add) -> MCPStatusResponse | None:
        try:
            status = await self._mcp_client.discover_tools()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Day 18 MCP discovery failed")
            add("discover_tools", "error", str(exc) or "MCP discovery failed")
            return None
        for step in status.trace:
            if step.step in ("initialize", "list_tools"):
                add(step.step, step.status, step.message)
        if status.connected:
            add(
                "discover_tools",
                "ok",
                f"Discovered {status.tools_count} monitoring MCP tools",
            )
        return status

    async def _run_tool_call(
        self,
        call: dict[str, Any],
        allowed: set[str],
        messages: list[dict[str, Any]],
        tool_calls_out: list[Day18ToolCallInfo],
        add,
    ) -> None:
        name = call.get("name") or ""
        arguments = call.get("arguments") or {}

        if name not in allowed:
            add("call_tool", "error", f"Unknown monitoring tool requested: {name}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"error": f"unknown tool: {name}"}),
                }
            )
            return

        add("call_tool", "ok", f"Calling monitoring MCP tool: {name}")
        try:
            result = await self._mcp_client.call_tool(name, arguments)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Day 18 MCP call_tool failed")
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
            if isinstance(payload, dict) and payload.get("job_id"):
                add(
                    "monitoring_job",
                    "ok",
                    f"Monitoring job {payload['job_id']} -> "
                    f"{payload.get('status', 'unknown')}",
                )
            add("mcp_result", "ok", "Monitoring MCP result returned to LLM")

        summary = {
            key: payload[key]
            for key in _SUMMARY_KEYS
            if isinstance(payload, dict) and key in payload
        }
        tool_calls_out.append(
            Day18ToolCallInfo(
                server=SERVER_LABEL,
                tool=name,
                arguments=arguments,
                result_summary=summary,
            )
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(payload, ensure_ascii=False),
            }
        )

    # ------------------------------------------------------------------
    # Small helper used by the HTTP layer
    # ------------------------------------------------------------------
    @property
    def monitoring(self) -> MonitoringService:
        return self._monitoring
