"""Day 17 — VictoriaLogs MCP server (local stdio transport).

Run standalone for a manual protocol check::

    python -m app.services.mcp.victorialogs.server

Like the Day 16 demo server this is NOT an HTTP service: it speaks the MCP
JSON-RPC protocol over stdin/stdout and is spawned as a subprocess by
``MCPClient``. The server itself performs the outbound HTTPS request to the
VictoriaLogs HTTP API (``/select/logsql/query``).

Exactly one tool is registered: ``search_logs``. The LLM supplies *structured*
filters (service / level / text / time window / limit); the server builds a
safe LogsQL query, calls VictoriaLogs, normalizes + sanitizes the JSON Lines
result and returns it. Raw LogsQL is never accepted from the caller.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field, ValidationError

from app.services.mcp.victorialogs.client import (
    VictoriaLogsClient,
    VictoriaLogsError,
    format_time,
    normalize_logs,
)
from app.services.mcp.victorialogs.query_builder import QueryBuildError, build_logsql
from app.services.mcp.victorialogs.sanitizer import sanitize_value
from app.services.mcp.victorialogs.schemas import (
    MAX_LIMIT,
    MAX_SINCE_MINUTES,
    MAX_WINDOW_MINUTES,
    MIN_LIMIT,
    MIN_SINCE_MINUTES,
    SearchLogsInput,
    _as_utc,
)

SERVER_NAME = "VictoriaLogs MCP"
SERVER_VERSION = "1.0"

TOOL_DESCRIPTION = (
    "Search recent stage logs stored in VictoriaLogs using safe structured "
    "filters (service, time window, level and optional message text). Returns "
    "normalized log rows (timestamp, message, fields) plus the generated query "
    "and a count. Use this when the user asks about service logs, errors or "
    "incidents. The time window is limited and results are capped."
)


def _service_field() -> str:
    """Resolve the configured service field inside the server process."""
    import os

    return (os.environ.get("VICTORIA_LOGS_SERVICE_FIELD") or "service").strip() or "service"


def build_server() -> FastMCP:
    """Create the VictoriaLogs MCP server with the ``search_logs`` tool.

    Kept as a factory so tests can introspect the registered tools without
    starting a subprocess.
    """
    server = FastMCP(SERVER_NAME)

    @server.tool(description=TOOL_DESCRIPTION)
    async def search_logs(
        service: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description="Service name to filter by (exact match on the service field).",
        ),
        since_minutes: int = Field(
            15,
            ge=MIN_SINCE_MINUTES,
            le=MAX_SINCE_MINUTES,
            description="How many minutes back from now (1-360). Ignored when start+end are set.",
        ),
        level: str | None = Field(
            None,
            max_length=32,
            description="Optional log level filter, e.g. ERROR or WARN.",
        ),
        text_contains: str | None = Field(
            None,
            max_length=200,
            description="Optional substring the log message must contain.",
        ),
        limit: int = Field(
            100,
            ge=MIN_LIMIT,
            le=MAX_LIMIT,
            description="Maximum number of log rows to return (1-500).",
        ),
        start: datetime | None = Field(
            None,
            description="Optional ISO8601 window start (UTC). Used together with end.",
        ),
        end: datetime | None = Field(
            None,
            description="Optional ISO8601 window end (UTC). Used together with start.",
        ),
    ) -> dict[str, Any]:
        """Search recent stage logs in VictoriaLogs using safe structured filters."""
        # Re-validate through the Pydantic model (belt and suspenders: the
        # MCP layer already validated the published JSON schema).
        try:
            params = SearchLogsInput(
                service=service,
                since_minutes=since_minutes,
                level=level,
                text_contains=text_contains,
                limit=limit,
                start=start,
                end=end,
            )
        except ValidationError as exc:
            return {"error": str(exc), "error_kind": "query"}

        client = VictoriaLogsClient.from_env()
        service_field = _service_field()

        try:
            query = build_logsql(
                service=params.service,
                service_field=service_field,
                level=params.level,
                text_contains=params.text_contains,
            )
        except QueryBuildError as exc:
            return {"error": str(exc), "error_kind": "query"}

        if params.start is not None and params.end is not None:
            start_dt = _as_utc(params.start)
            end_dt = _as_utc(params.end)
            if end_dt - start_dt > timedelta(minutes=MAX_WINDOW_MINUTES):
                return {
                    "error": (
                        f"time window must not exceed {MAX_WINDOW_MINUTES} minutes"
                    ),
                    "error_kind": "query",
                }
        else:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(minutes=params.since_minutes)

        try:
            rows, malformed = await client.search(
                query=query,
                start=start_dt,
                end=end_dt,
                limit=params.limit,
            )
        except VictoriaLogsError as exc:
            # Controlled, secret-free error returned to the agent instead of a
            # traceback. The agent can explain the failure to the user.
            return {
                "error": exc.message,
                "error_kind": exc.kind,
                "query": query,
            }

        logs = normalize_logs(rows, sanitize=sanitize_value)
        return {
            "service": params.service,
            "since_minutes": params.since_minutes,
            "start": format_time(start_dt),
            "end": format_time(end_dt),
            "level": params.level,
            "text_contains": params.text_contains,
            "query": query,
            "count": len(logs),
            "limit": params.limit,
            "truncated": len(logs) >= params.limit,
            "malformed_lines": malformed,
            "logs": logs,
        }

    return server


mcp = build_server()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    mcp.run(transport="stdio")
