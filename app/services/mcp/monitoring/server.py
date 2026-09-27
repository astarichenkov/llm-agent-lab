"""Day 18 — the monitoring MCP server.

This is a REAL MCP server (``FastMCP``) that exposes the four Day 18 tools::

    start_monitoring
    get_monitoring_status
    get_monitoring_summary
    stop_monitoring

Why a SEPARATE server and an IN-MEMORY transport?

* The scheduler and the SQLite persistence must live in the long-running
  FastAPI process. A short-lived stdio MCP subprocess (the Day 16/17 pattern)
  cannot own that lifecycle, and bridging a subprocess back to the app would
  require an HTTP/socket hop.
* MCP's standard in-memory transport (``create_connected_server_and_client_session``)
  still runs the real MCP protocol: ``tools/list``, JSON-schema validation and
  ``tools/call``. Only the transport differs, so this remains a genuine MCP
  tool-calling path while keeping the architecture simple.

The tool handlers delegate to the in-process :class:`MonitoringService`. The
server itself contains no scheduler and no HTTP code.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field, ValidationError

from app.schemas.day18 import (
    DEFAULT_LOOKBACK_MINUTES,
    DEFAULT_MONITORING_LIMIT,
    DEFAULT_SUMMARY_RUNS,
    INTERVAL_SECONDS_PRESETS,
    MAX_LOOKBACK_MINUTES,
    MAX_MONITORING_LIMIT,
    MAX_SUMMARY_RUNS,
    MIN_LOOKBACK_MINUTES,
    MIN_MONITORING_LIMIT,
    IntervalSeconds,
    StartMonitoringInput,
)

SERVER_NAME = "Monitoring MCP"
SERVER_VERSION = "1.0"

_INTERVAL_HELP = ", ".join(str(value) for value in INTERVAL_SECONDS_PRESETS)

START_DESCRIPTION = (
    "Start a recurring background monitoring job for a service in "
    "VictoriaLogs. The scheduler runs it every interval_seconds and stores "
    "an aggregate (log count) per run. interval_seconds must be one of: "
    f"{_INTERVAL_HELP}. Lookback is independent of the interval."
)
STATUS_DESCRIPTION = (
    "Return the current status of a monitoring job: status, service, "
    "interval, lookback, last/next run and how many runs were recorded."
)
SUMMARY_DESCRIPTION = (
    "Return an aggregated summary of the most recent monitoring runs for a "
    "job (runs, successes/failures, total/average/min/max logs, trend). "
    "Use this to answer 'покажи сводку мониторинга <job_id>'."
)
STOP_DESCRIPTION = (
    "Stop a monitoring job: unregister it from the scheduler and mark it "
    "stopped in SQLite. Existing run history is preserved."
)


def build_monitoring_server(monitoring: Any) -> FastMCP:
    """Create the monitoring MCP server bound to a ``MonitoringService``.

    ``monitoring`` is duck-typed (no import of the service module here) to
    keep the MCP layer decoupled from the Day 18 service implementation.
    """
    server = FastMCP(SERVER_NAME)

    @server.tool(description=START_DESCRIPTION)
    async def start_monitoring(
        service: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description="Service name to monitor.",
        ),
        level: str | None = Field(
            default=None,
            max_length=32,
            description="Optional log level filter, e.g. ERROR or WARN.",
        ),
        interval_seconds: IntervalSeconds = Field(
            default=30,
            description=f"How often to run. One of: {_INTERVAL_HELP}.",
        ),
        lookback_minutes: int = Field(
            default=DEFAULT_LOOKBACK_MINUTES,
            ge=MIN_LOOKBACK_MINUTES,
            le=MAX_LOOKBACK_MINUTES,
            description="How many minutes back each run searches (1-360).",
        ),
        text_contains: str | None = Field(
            default=None,
            max_length=200,
            description="Optional substring the log message must contain.",
        ),
        limit: int = Field(
            default=DEFAULT_MONITORING_LIMIT,
            ge=MIN_MONITORING_LIMIT,
            le=MAX_MONITORING_LIMIT,
            description="Maximum log rows per run (1-500).",
        ),
    ) -> dict[str, Any]:
        """Create and schedule a new monitoring job, running it once immediately."""
        try:
            params = StartMonitoringInput(
                service=service,
                level=level,
                interval_seconds=interval_seconds,
                lookback_minutes=lookback_minutes,
                text_contains=text_contains,
                limit=limit,
            )
        except ValidationError as exc:
            return {"error": str(exc), "error_kind": "validation"}

        try:
            result = await monitoring.start_job(params)
        except Exception as exc:  # noqa: BLE001 - controlled tool error
            return {"error": str(exc), "error_kind": "internal"}

        payload = result.model_dump(mode="json")
        payload["message"] = (
            "Monitoring job created. The first run was executed "
            "immediately; subsequent runs happen every "
            f"{result.interval_seconds} seconds."
        )
        return payload

    @server.tool(description=STATUS_DESCRIPTION)
    async def get_monitoring_status(job_id: str) -> dict[str, Any]:
        """Return the status of a monitoring job."""
        status = monitoring.get_status(job_id)
        if status is None:
            return {
                "error": f"unknown monitoring job: {job_id}",
                "error_kind": "not_found",
                "job_id": job_id,
            }
        return status.model_dump(mode="json")

    @server.tool(description=SUMMARY_DESCRIPTION)
    async def get_monitoring_summary(
        job_id: str,
        last_runs: int = Field(
            default=DEFAULT_SUMMARY_RUNS,
            ge=1,
            le=MAX_SUMMARY_RUNS,
            description="How many recent runs to aggregate (1-100).",
        ),
    ) -> dict[str, Any]:
        """Return an aggregated summary of the most recent runs."""
        summary = monitoring.get_summary(job_id, last_runs=last_runs)
        if summary is None:
            return {
                "error": f"unknown monitoring job: {job_id}",
                "error_kind": "not_found",
                "job_id": job_id,
            }
        return summary.model_dump(mode="json")

    @server.tool(description=STOP_DESCRIPTION)
    async def stop_monitoring(job_id: str) -> dict[str, Any]:
        """Stop a monitoring job and keep its history."""
        result = monitoring.stop_job(job_id)
        if result is None:
            return {
                "error": f"unknown monitoring job: {job_id}",
                "error_kind": "not_found",
                "job_id": job_id,
            }
        return result.model_dump(mode="json")

    return server
