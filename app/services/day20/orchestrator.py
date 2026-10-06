"""Day 20 — multi-server MCP orchestration (the agent loop).

The chain is NOT a hardcoded Python sequence. The orchestrator exposes the
unified tool set discovered from EVERY registered MCP server and lets the LLM
decide, one step at a time, which tool (from which server) to call next based
on the previous results::

    user request
      -> LLM (all discovered tools)
      -> if no tool call: final answer
      -> resolve tool -> MCP server
      -> real MCP tools/call
      -> bounded tool result appended to the LLM context
      -> repeat (bounded by DAY20_MAX_TOOL_ITERATIONS / DAY20_MAX_TOTAL_TOOL_CALLS)

Design / safety:

* one failed MCP server never crashes the whole run — discovery isolates each
  server and a tool call is wrapped in a timeout;
* the loop is bounded in iterations AND total tool calls;
* every tool call is recorded in a server-side trace with bounded, secret-
  masked arguments;
* the analysis policy explicitly separates observed facts / correlation /
  hypothesis / verification required and states "correlation is not causation"
  and "commit timestamp != deployment timestamp".
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from app.config import Settings
from app.schemas.day16 import MCPToolInfo
from app.schemas.day20 import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    TRACE_KIND_DISCOVERY,
    TRACE_KIND_ERROR,
    TRACE_KIND_FINAL,
    TRACE_KIND_LLM,
    TRACE_KIND_TOOL_CALL,
    TRACE_STATUS_ERROR,
    TRACE_STATUS_OK,
    TRACE_STATUS_TIMEOUT,
    Day20Artifacts,
    Day20InvestigateRequest,
    Day20InvestigateResponse,
    Day20MCPStatusResponse,
    Day20ServerInfo,
    Day20ToolCallInfo,
    Day20ToolRoute,
    Day20TraceStep,
)
from app.services.day20.artifacts import generate_investigation_id
from app.services.day20.reports import ReportsToolset
from app.services.day19.masking import DataMasker
from app.services.deepseek import DeepSeekError, DeepSeekService
from app.services.mcp.client import (
    MCPClient,
    gitea_server_config,
    victorialogs_server_config,
)
from app.services.mcp.registry import (
    MCPServerRegistry,
    ToolCollisionError,
    UnknownToolError,
)
from app.services.mcp.reports.client import ReportsMCPClient
from app.services.mcp.victorialogs.sanitizer import sanitize_value

logger = logging.getLogger("app.services.day20.orchestrator")

# Fallback limits when a Settings object does not carry the Day 20 fields.
DEFAULT_MAX_TOOL_ITERATIONS = 16
DEFAULT_MAX_TOTAL_TOOL_CALLS = 24
DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0

# Bounds for the secret-masked arguments stored in the trace.
MAX_ARG_STRING_CHARS = 400
MAX_ARG_ITEMS = 5
MAX_ARG_DEPTH = 6
MAX_SUMMARY_CHARS = 400

SYSTEM_PROMPT = (
    "You are an SRE incident-investigation assistant. Several MCP servers "
    "expose tools to you: logs (search_logs), source control "
    "(recent_commits, get_commit, get_commit_diff) and reports (save_report). "
    "You decide which tool to call next based on the user request and the "
    "results of previous tool calls. Only call a tool when you actually need "
    "its result; skip tools that are not relevant. After each result, decide "
    "the next best action. When you have enough evidence, call save_report "
    "ONCE and then give the final answer.\n\n"
    "INVESTIGATION POLICY (mandatory):\n"
    "- Correlation is not causation.\n"
    "- Separate your findings into: Observed facts, Correlation, Hypothesis, "
    "and Verification required.\n"
    "- Never claim a commit CAUSED an incident merely because it precedes it "
    "in time.\n"
    "- A commit timestamp is not a deployment timestamp; state deployment / "
    "rollback / reproduction checks as verification steps.\n"
    "- Base every statement on real tool results. If a tool fails or a server "
    "is unavailable, say so plainly and continue with what you have.\n"
    "Answer in the same language as the user."
)

_TOOL_REASONS = {
    "search_logs": "Need incident start time and error patterns",
    "recent_commits": "Need changes near the incident start",
    "get_commit": "Need details of a candidate change",
    "get_commit_diff": "Need the concrete change to correlate with the error",
    "save_report": "Persist the investigation report",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _bound_value(value: Any, depth: int = 0) -> Any:
    """Recursively bound a JSON-like value so traces stay small."""
    if depth > MAX_ARG_DEPTH:
        return "<...>"
    if isinstance(value, str):
        if len(value) <= MAX_ARG_STRING_CHARS:
            return value
        return value[:MAX_ARG_STRING_CHARS] + f"...<{len(value)} chars>"
    if isinstance(value, dict):
        items = list(value.items())[:25]
        return {str(key): _bound_value(item, depth + 1) for key, item in items}
    if isinstance(value, list):
        head = [_bound_value(item, depth + 1) for item in value[:MAX_ARG_ITEMS]]
        if len(value) > MAX_ARG_ITEMS:
            head.append(f"...<{len(value)} items>")
        return head
    return value


def safe_arguments(arguments: Any) -> dict[str, Any]:
    """Return a bounded, secret-masked copy of tool arguments for the trace."""
    if not isinstance(arguments, dict):
        return {"value": str(arguments)[:MAX_ARG_STRING_CHARS]}
    bounded = _bound_value(arguments)
    masked = sanitize_value(bounded)
    return masked if isinstance(masked, dict) else {}


def summarize_result(tool: str, payload: Any) -> str:
    """Return a short, secret-free summary of one tool result."""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if error:
        return f"error: {str(error)[:MAX_SUMMARY_CHARS]}"
    if tool == "search_logs":
        count = payload.get("count")
        first = ""
        logs = payload.get("logs")
        if isinstance(logs, list) and logs:
            first_log = logs[0] if isinstance(logs[0], dict) else {}
            timestamp = first_log.get("timestamp")
            if timestamp:
                first = f"; first at {timestamp}"
        return f"{count} logs{first}"
    if tool == "recent_commits":
        count = payload.get("count")
        commits = payload.get("commits")
        if count is None and isinstance(commits, list):
            count = len(commits)
        return f"{count} commits"
    if tool == "get_commit":
        return f"commit {payload.get('short_sha') or payload.get('sha') or ''}".strip()
    if tool == "get_commit_diff":
        files = payload.get("files")
        count = len(files) if isinstance(files, list) else 0
        suffix = " (truncated)" if payload.get("truncated") else ""
        return f"{count} files changed{suffix}"
    if tool == "save_report":
        if payload.get("investigation_id"):
            return f"saved {payload.get('investigation') or 'investigation.md'}"
        return "report saved"
    return ""


def _tool_error(result: Any, payload: Any) -> str | None:
    payload_error = payload.get("error") if isinstance(payload, dict) else None
    if getattr(result, "is_error", False) or payload_error:
        return str(payload_error or getattr(result, "error", None) or "tool error")
    return None


class Day20Orchestrator:
    """Runs the bounded multi-server agent loop for one investigation."""

    def __init__(
        self,
        settings: Settings,
        registry: MCPServerRegistry,
        *,
        deepseek: Any | None = None,
        clock: Callable[[], datetime] | None = None,
        max_tool_iterations: int | None = None,
        max_total_tool_calls: int | None = None,
        tool_timeout_seconds: float | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._deepseek = deepseek or DeepSeekService(settings)
        self._now = clock or _utc_now
        self.max_tool_iterations = (
            max_tool_iterations
            if max_tool_iterations is not None
            else getattr(settings, "day20_max_tool_iterations", DEFAULT_MAX_TOOL_ITERATIONS)
        )
        self.max_total_tool_calls = (
            max_total_tool_calls
            if max_total_tool_calls is not None
            else getattr(
                settings,
                "day20_max_total_tool_calls",
                DEFAULT_MAX_TOTAL_TOOL_CALLS,
            )
        )
        self.tool_timeout_seconds = (
            tool_timeout_seconds
            if tool_timeout_seconds is not None
            else getattr(
                settings,
                "day20_tool_timeout_seconds",
                DEFAULT_TOOL_TIMEOUT_SECONDS,
            )
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def investigate(
        self, request: Day20InvestigateRequest
    ) -> Day20InvestigateResponse:
        started = self._now()
        trace: list[Day20TraceStep] = []
        tool_calls_out: list[Day20ToolCallInfo] = []
        servers_used: list[str] = []
        tools_used: list[str] = []
        sequence = 0

        def add_trace(
            kind: str,
            status: str,
            *,
            server: str | None = None,
            server_label: str | None = None,
            tool: str | None = None,
            arguments_safe: dict[str, Any] | None = None,
            started_at: datetime | None = None,
            finished_at: datetime | None = None,
            duration_ms: int | None = None,
            result_summary: str = "",
            reason: str = "",
            error: str | None = None,
        ) -> None:
            nonlocal sequence
            sequence += 1
            trace.append(
                Day20TraceStep(
                    sequence=sequence,
                    kind=kind,
                    server=server,
                    server_label=server_label,
                    tool=tool,
                    arguments_safe=arguments_safe or {},
                    status=status,
                    started_at=_iso(started_at) if started_at else None,
                    finished_at=_iso(finished_at) if finished_at else None,
                    duration_ms=duration_ms,
                    result_summary=result_summary,
                    reason=reason,
                    error=error,
                )
            )

        # 0) Unified discovery across every registered MCP server.
        try:
            discovery = await self._registry.discover_all()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Day 20 MCP discovery failed")
            add_trace(TRACE_KIND_DISCOVERY, TRACE_STATUS_ERROR, error=str(exc))
            return self._failed(request, trace, f"MCP discovery failed: {exc}")

        for server in discovery.servers:
            add_trace(
                TRACE_KIND_DISCOVERY,
                TRACE_STATUS_OK if server.connected else TRACE_STATUS_ERROR,
                server=server.name,
                server_label=server.label,
                result_summary=(
                    f"{server.tools_count} tools"
                    if server.connected
                    else (server.error or "not connected")
                ),
                error=None if server.connected else (server.error or "not connected"),
            )

        if discovery.collisions:
            detail = "; ".join(
                f"{tool}: {a} vs {b}" for tool, a, b in discovery.collisions
            )
            add_trace(
                TRACE_KIND_DISCOVERY,
                TRACE_STATUS_ERROR,
                error=f"duplicate MCP tool name(s): {detail}",
            )
            return self._failed(
                request, trace, f"duplicate MCP tool name(s): {detail}"
            )

        if not discovery.tools:
            add_trace(
                TRACE_KIND_DISCOVERY,
                TRACE_STATUS_ERROR,
                error="no MCP tools available",
            )
            return self._failed(request, trace, "No MCP tools are available.")

        openai_tools = [_openai_tool(tool) for tool in discovery.tools]

        # 1) Start the investigation context so save_report has a safe id.
        reports_client = self._registry.get_client("reports")
        # Opt-in identifier masking ("Маскировать данные"): when enabled, the
        # masker is applied to every tool result sent to the LLM, the trace
        # arguments, the final answer and the saved report.
        masker = DataMasker([request.service]) if request.mask_data else None
        investigation_id = generate_investigation_id(started)
        if reports_client is not None and hasattr(reports_client, "begin_investigation"):
            try:
                investigation_id = reports_client.begin_investigation(
                    request=(
                        masker.mask_text(request.message)
                        if masker is not None
                        else request.message
                    ),
                    investigation_id=investigation_id,
                    context={
                        "model": self._settings.deepseek_model,
                        "started_at": _iso(started),
                        "servers_used": [],
                        "tools_used": [],
                        "tool_calls": 0,
                        "tool_calls_detail": [],
                    },
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("Day 20: failed to initialise report context")
        if masker is not None and reports_client is not None and hasattr(
            reports_client, "set_masker"
        ):
            reports_client.set_masker(masker)

        def sync_context(pending: int = 0, status: str = "in_progress") -> None:
            if reports_client is None or not hasattr(reports_client, "update_context"):
                return
            try:
                reports_client.update_context(
                    servers_used=list(servers_used),
                    tools_used=list(tools_used),
                    tool_calls=len(tool_calls_out) + pending,
                    tool_calls_detail=[
                        call.model_dump() for call in tool_calls_out
                    ],
                    iterations=iterations,
                    status=status,
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("Day 20: failed to update report context")

        # 2) The agent loop. The LLM decides the next tool every iteration.
        user_content = self._user_message(request)
        if masker is not None:
            user_content = masker.mask_text(user_content)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        iterations = 0
        final_content = ""
        loop_error: str | None = None
        limit_reached = False

        while iterations < self.max_tool_iterations:
            iterations += 1
            try:
                content, _finish, _usage, llm_calls = (
                    await self._deepseek.generate_with_tools(
                        messages,
                        tools=openai_tools,
                        model=self._settings.deepseek_model,
                        temperature=0.2,
                        max_tokens=1500,
                        thinking=False,
                    )
                )
            except DeepSeekError as exc:
                loop_error = exc.message
                add_trace(TRACE_KIND_ERROR, TRACE_STATUS_ERROR, error=exc.message)
                break

            if not llm_calls:
                final_content = content or ""
                add_trace(
                    TRACE_KIND_FINAL,
                    TRACE_STATUS_OK,
                    result_summary="final answer produced",
                )
                break

            selected = ", ".join(
                f"{call.get('name')}" for call in llm_calls if call.get("name")
            )
            add_trace(
                TRACE_KIND_LLM,
                TRACE_STATUS_OK,
                result_summary=f"selected tool(s): {selected}",
            )
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
                                "arguments": json.dumps(call.get("arguments") or {}),
                            },
                        }
                        for call in llm_calls
                    ],
                }
            )

            for call in llm_calls:
                if len(tool_calls_out) >= self.max_total_tool_calls:
                    limit_reached = True
                    add_trace(
                        TRACE_KIND_ERROR,
                        TRACE_STATUS_ERROR,
                        error=(
                            "maximum total MCP tool calls reached "
                            f"({self.max_total_tool_calls})"
                        ),
                    )
                    break
                await self._execute_call(
                    call,
                    messages=messages,
                    trace_add=add_trace,
                    tool_calls_out=tool_calls_out,
                    servers_used=servers_used,
                    tools_used=tools_used,
                    sync_context=sync_context,
                    masker=masker,
                )
            if limit_reached:
                break
        else:
            limit_reached = True

        # 3) Final synthesis. If the loop was cut short (or ended without a
        # textual answer), ask once more WITHOUT tools for a partial summary.
        limit_error = None
        if limit_reached:
            add_trace(
                TRACE_KIND_ERROR,
                TRACE_STATUS_ERROR,
                error=(
                    "maximum MCP tool iterations reached "
                    f"({self.max_tool_iterations})"
                ),
            )
            limit_error = (
                "Maximum number of MCP tool iterations/calls reached; "
                "the answer may be partial."
            )

        if not final_content.strip():
            final_content = await self._final_synthesis(
                messages=messages,
                limit_reached=limit_reached,
                add_trace=add_trace,
            )

        status = STATUS_COMPLETED
        if loop_error and not final_content.strip():
            status = STATUS_FAILED
        elif limit_reached or loop_error:
            status = STATUS_PARTIAL

        if masker is not None and final_content:
            final_content = masker.mask_text(final_content)

        # 4) Read back the saved report descriptor, if save_report ran.
        artifacts = self._extract_artifacts()
        sync_context_status = "completed" if status == STATUS_COMPLETED else status
        if reports_client is not None and hasattr(reports_client, "update_context"):
            try:
                reports_client.update_context(
                    status=sync_context_status,
                    servers_used=list(servers_used),
                    tools_used=list(tools_used),
                    tool_calls=len(tool_calls_out),
                    tool_calls_detail=[call.model_dump() for call in tool_calls_out],
                    iterations=iterations,
                    finished_at=_iso(self._now()),
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("Day 20: failed to finalize report context")

        return Day20InvestigateResponse(
            investigation_id=investigation_id,
            status=status,
            answer=final_content.strip(),
            servers_used=servers_used,
            tools_used=tools_used,
            tool_calls=tool_calls_out,
            trace=trace,
            artifacts=artifacts,
            report_saved=artifacts is not None,
            iterations=iterations,
            error=loop_error or limit_error,
        )

    # ------------------------------------------------------------------
    # One tool call
    # ------------------------------------------------------------------
    async def _execute_call(
        self,
        call: dict[str, Any],
        *,
        messages: list[dict[str, Any]],
        trace_add,
        tool_calls_out: list[Day20ToolCallInfo],
        servers_used: list[str],
        tools_used: list[str],
        sync_context,
        masker: Any | None = None,
    ) -> None:
        name = call.get("name") or ""
        arguments = call.get("arguments") or {}
        started_at = self._now()
        safe_args = safe_arguments(arguments)
        if masker is not None:
            safe_args = masker.mask_value(safe_args)
        reason = _TOOL_REASONS.get(name, "")

        # Resolve server. Unknown tools are returned to the LLM as an error
        # tool result so it can recover instead of the run crashing.
        try:
            server = self._registry.server_for_tool(name)
        except UnknownToolError:
            trace_add(
                TRACE_KIND_TOOL_CALL,
                TRACE_STATUS_ERROR,
                tool=name,
                arguments_safe=safe_args,
                started_at=started_at,
                finished_at=self._now(),
                duration_ms=0,
                result_summary="unknown tool",
                reason=reason,
                error=f"unknown MCP tool: {name}",
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps({"error": f"unknown tool: {name}"}),
                }
            )
            return

        server_name = server.name
        if server_name not in servers_used:
            servers_used.append(server_name)
        if name not in tools_used:
            tools_used.append(name)

        if server_name == "reports":
            safe_args = dict(safe_args)

        # Update the report context BEFORE save_report runs so its metadata
        # already includes the current call count. A successfully persisted
        # report is itself the completion signal for the investigation.
        sync_context(
            1, status="completed" if name == "save_report" else "in_progress"
        )

        result = None
        timed_out = False
        payload: Any = {}
        try:
            result = await asyncio.wait_for(
                self._registry.call_tool(name, arguments),
                timeout=self.tool_timeout_seconds,
            )
            payload = result.as_payload()
        except asyncio.TimeoutError:
            timed_out = True
        except Exception as exc:  # noqa: BLE001 - isolate each tool call
            logger.warning("Day 20 tool call failed: %s", exc.__class__.__name__)
            payload = {"error": "MCP tool call failed"}

        finished_at = self._now()
        duration_ms = max(
            0, int((finished_at - started_at).total_seconds() * 1000)
        )

        if timed_out:
            error = (
                f"tool '{name}' timed out after {self.tool_timeout_seconds:g}s"
            )
            payload = {"error": error}
            status = TRACE_STATUS_TIMEOUT
        else:
            error = _tool_error(result, payload)
            status = TRACE_STATUS_ERROR if error else TRACE_STATUS_OK

        if masker is not None:
            # Collect service/container names from the rows BEFORE masking so
            # their literal occurrences in messages are masked too.
            if isinstance(payload, dict) and isinstance(payload.get("logs"), list):
                masker.add_log_literals(payload["logs"])
            payload = masker.mask_value(payload)

        summary = summarize_result(name, payload)
        if timed_out:
            summary = f"timeout after {self.tool_timeout_seconds:g}s"
        if error and not summary:
            summary = f"error: {error[:MAX_SUMMARY_CHARS]}"

        trace_add(
            TRACE_KIND_TOOL_CALL,
            status,
            server=server_name,
            server_label=server.label,
            tool=name,
            arguments_safe=safe_args,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            result_summary=summary,
            reason=reason,
            error=error,
        )
        tool_calls_out.append(
            Day20ToolCallInfo(
                sequence=len(tool_calls_out) + 1,
                server=server_name,
                server_label=server.label,
                tool=name,
                status=status,
                duration_ms=duration_ms,
                result_summary=summary,
            )
        )
        sync_context()

        if error:
            # Bound the error payload handed back to the model.
            tool_payload: dict[str, Any] = {"error": error}
        elif isinstance(payload, dict):
            tool_payload = payload
        else:
            tool_payload = {"result": payload}

        try:
            content = json.dumps(tool_payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            content = json.dumps({"error": "tool result was not serializable"})
        # Hard cap the tool result size that enters the LLM context.
        if len(content) > 20000:
            content = content[:20000] + "...<truncated>"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": content,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _user_message(request: Day20InvestigateRequest) -> str:
        hints: list[str] = []
        if request.service:
            hints.append(f"service hint: {request.service}")
        hints.append(f"log lookback hint: {request.since_minutes} minutes")
        if request.save_report:
            hints.append("please save the final report with save_report")
        else:
            hints.append("do not save a report")
        return request.message.strip() + "\n\n(" + "; ".join(hints) + ")"

    async def _final_synthesis(
        self, *, messages: list[dict[str, Any]], limit_reached: bool, add_trace
    ) -> str:
        instruction = (
            "Now produce the final investigation answer. Use the structure: "
            "Observed facts; Relevant code changes; Correlation; Hypothesis; "
            "Verification required. Do not claim causation. If a tool or "
            "server was unavailable, say so."
        )
        if limit_reached:
            instruction += (
                " The tool-call limit was reached, so this may be a partial "
                "answer."
            )
        final_messages = list(messages) + [{"role": "user", "content": instruction}]
        try:
            content, _finish, _usage = await self._deepseek.generate(
                final_messages,
                model=self._settings.deepseek_model,
                temperature=0.2,
                max_tokens=1200,
                thinking=False,
            )
        except DeepSeekError as exc:
            add_trace(
                TRACE_KIND_ERROR,
                TRACE_STATUS_ERROR,
                error=f"final synthesis failed: {exc.message}",
            )
            return ""
        except Exception as exc:  # noqa: BLE001 - defensive
            add_trace(
                TRACE_KIND_ERROR,
                TRACE_STATUS_ERROR,
                error=f"final synthesis failed ({exc.__class__.__name__})",
            )
            return ""
        text = (content or "").strip()
        if text:
            add_trace(
                TRACE_KIND_FINAL,
                TRACE_STATUS_OK,
                result_summary="final synthesis produced",
            )
        return text

    def _extract_artifacts(self) -> Day20Artifacts | None:
        """Read the artifact descriptor captured by the Reports MCP client."""
        client = self._registry.get_client("reports")
        captured = getattr(client, "last_artifacts", None) if client else None
        if not isinstance(captured, dict):
            return None
        investigation_id = captured.get("investigation_id")
        if not investigation_id:
            return None
        return Day20Artifacts(
            investigation_id=str(investigation_id),
            root=str(captured.get("root", "")),
            investigation=str(captured.get("investigation", "investigation.md")),
            metadata=str(captured.get("metadata", "metadata.json")),
        )

    def _failed(
        self,
        request: Day20InvestigateRequest,
        trace: list[Day20TraceStep],
        error: str,
    ) -> Day20InvestigateResponse:
        return Day20InvestigateResponse(
            investigation_id="",
            status=STATUS_FAILED,
            answer="",
            trace=trace,
            error=error,
        )


class Day20Service:
    """Builds the MCP registry and exposes discovery + investigation."""

    def __init__(
        self,
        settings: Settings,
        *,
        deepseek: Any | None = None,
        client_factory: Callable[[ReportsToolset], dict[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._deepseek = deepseek or DeepSeekService(settings)
        self._client_factory = client_factory
        self._now = clock or _utc_now

    # ------------------------------------------------------------------
    # Registry construction
    # ------------------------------------------------------------------
    def _build_registry(
        self, reports: ReportsToolset | None = None
    ) -> tuple[MCPServerRegistry, ReportsToolset]:
        toolset = reports or ReportsToolset(self._settings)
        registry = MCPServerRegistry()
        if self._client_factory is not None:
            clients = self._client_factory(toolset)
            for name, client in clients.items():
                label = {
                    "victorialogs": "VictoriaLogs MCP",
                    "gitea": "Gitea MCP",
                    "reports": "Reports MCP",
                }.get(name, name)
                transport = "in-memory" if name == "reports" else "stdio"
                registry.register(
                    name, client, label=label, transport=transport
                )
            return registry, toolset

        registry.register(
            "victorialogs",
            MCPClient(victorialogs_server_config(env=self._victorialogs_env())),
            label="VictoriaLogs MCP",
            transport="stdio",
        )
        registry.register(
            "gitea",
            MCPClient(gitea_server_config(env=self._gitea_env())),
            label="Gitea MCP",
            transport="stdio",
        )
        registry.register(
            "reports",
            ReportsMCPClient(toolset),
            label="Reports MCP",
            transport="in-memory",
        )
        return registry, toolset

    def _victorialogs_env(self) -> dict[str, str]:
        s = self._settings
        return {
            "VICTORIA_LOGS_BASE_URL": s.victoria_logs_base_url,
            "VICTORIA_LOGS_SERVICE_FIELD": s.victoria_logs_service_field,
            "VICTORIA_LOGS_TIMEOUT_SECONDS": str(s.victoria_logs_timeout_seconds),
            "VICTORIA_LOGS_VERIFY_SSL": "1" if s.victoria_logs_verify_ssl else "0",
            "VICTORIA_LOGS_CA_BUNDLE": s.victoria_logs_ca_bundle,
        }

    def _gitea_env(self) -> dict[str, str]:
        s = self._settings
        return {
            "GITEA_BASE_URL": s.gitea_base_url,
            "GITEA_TOKEN": s.gitea_token,
            "GITEA_REPOSITORY_OWNER": s.gitea_repository_owner,
            "GITEA_REPOSITORY_NAME": s.gitea_repository_name,
            "GITEA_DEFAULT_BRANCH": s.gitea_default_branch,
            "GITEA_TIMEOUT_SECONDS": str(s.gitea_timeout_seconds),
            "GITEA_MAX_DIFF_CHARS": str(s.gitea_max_diff_chars),
            "GITEA_MAX_COMMITS": str(s.gitea_max_commits),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def discover(self) -> Day20MCPStatusResponse:
        """Run unified ``tools/list`` across every MCP server."""
        registry, _toolset = self._build_registry()
        try:
            discovery = await registry.discover_all()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Day 20 discovery failed")
            return Day20MCPStatusResponse(connected=False, error=str(exc))

        servers = [
            Day20ServerInfo(
                name=server.name,
                label=server.label,
                transport=server.transport,
                connected=server.connected,
                tools_count=server.tools_count,
                tools=[tool.model_dump() for tool in server.tools],
                error=server.error,
            )
            for server in discovery.servers
        ]
        routes = [
            Day20ToolRoute(**row) for row in registry.routing(discovery)
        ]
        error = None
        if discovery.collisions:
            error = "; ".join(
                f"{tool}: {a} vs {b}" for tool, a, b in discovery.collisions
            )
        return Day20MCPStatusResponse(
            connected=discovery.connected_any,
            servers=servers,
            routing=routes,
            tools_count=len(discovery.tools),
            error=error,
        )

    async def investigate(
        self, request: Day20InvestigateRequest
    ) -> Day20InvestigateResponse:
        """Run one full multi-server investigation."""
        registry, _toolset = self._build_registry()
        orchestrator = Day20Orchestrator(
            self._settings, registry, deepseek=self._deepseek, clock=self._now
        )
        return await orchestrator.investigate(request)

    def read_artifact(self, investigation_id: str, filename: str) -> str:
        """Read one known investigation artifact (safe: root+name enforced)."""
        store = ReportsToolset(self._settings).store
        return store.read(investigation_id, filename)


__all__ = [
    "Day20Orchestrator",
    "Day20Service",
    "SYSTEM_PROMPT",
    "safe_arguments",
    "summarize_result",
]
