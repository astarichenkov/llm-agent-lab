"""Pydantic schemas for Day 18 — scheduled monitoring.

Day 18 turns the Day 17 one-shot VictoriaLogs search into a *recurring* job:

    start_monitoring -> monitoring job (SQLite) -> scheduler -> VictoriaLogs
                                           \\-> aggregated run history (SQLite)

The models here are shared by three callers:

* the MCP monitoring tools (``start_monitoring`` ... ), which publish derived
  JSON schemas to the LLM;
* the HTTP dashboard endpoints;
* the SQLite repository, which stores the job parameters and run aggregates.

Only aggregate data is persisted. Raw log rows and secrets are NEVER stored.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------------
# Interval / lookback presets
# ----------------------------------------------------------------------
# The UI may only pick one of these values and the backend re-validates the
# whitelist. 10 / 30 / 60 seconds exist for the recorded demo; in real
# monitoring longer intervals are recommended.
INTERVAL_SECONDS_PRESETS: tuple[int, ...] = (10, 30, 60, 300, 600, 1800, 3600)
INTERVAL_LABELS: dict[int, str] = {
    10: "10 sec",
    30: "30 sec",
    60: "1 min",
    300: "5 min",
    600: "10 min",
    1800: "30 min",
    3600: "1 hour",
}
# Demo-friendly intervals (short). Documented explicitly; there is no hidden
# demo mode — it is just the lower end of the same whitelist.
DEMO_INTERVAL_SECONDS: tuple[int, ...] = (10, 30, 60)
DEFAULT_INTERVAL_SECONDS = 30

# Interval and lookback are INDEPENDENT parameters: how often a run happens
# vs. how far back each run looks. Windows may overlap by design.
LOOKBACK_MINUTES_PRESETS: tuple[int, ...] = (1, 5, 10, 15, 30, 60)
LOOKBACK_LABELS: dict[int, str] = {
    1: "1 min",
    5: "5 min",
    10: "10 min",
    15: "15 min",
    30: "30 min",
    60: "1 hour",
}
DEFAULT_LOOKBACK_MINUTES = 5
MIN_LOOKBACK_MINUTES = 1
MAX_LOOKBACK_MINUTES = 360

# Reuse the Day 17 limit bounds (1..500).
DEFAULT_MONITORING_LIMIT = 100
MIN_MONITORING_LIMIT = 1
MAX_MONITORING_LIMIT = 500

# Summary window bounds.
DEFAULT_SUMMARY_RUNS = 10
MAX_SUMMARY_RUNS = 100

# Job / run status values.
JOB_STATUS_ACTIVE = "active"
JOB_STATUS_STOPPED = "stopped"
JOB_STATUS_ERROR = "error"
JOB_STATUSES: tuple[str, ...] = (
    JOB_STATUS_ACTIVE,
    JOB_STATUS_STOPPED,
    JOB_STATUS_ERROR,
)
RUN_STATUS_SUCCESS = "success"
RUN_STATUS_ERROR = "error"

# The whitelist enforced in the schema (and re-checked in the service).
IntervalSeconds = Literal[10, 30, 60, 300, 600, 1800, 3600]


# ----------------------------------------------------------------------
# Input models
# ----------------------------------------------------------------------
class StartMonitoringInput(BaseModel):
    """Validated parameters for a new monitoring job.

    ``interval_seconds`` is constrained to the preset whitelist both here and
    in the MCP tool schema, so a caller cannot schedule an arbitrary interval.
    """

    service: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Service name to monitor (exact match on the service field).",
    )
    level: str | None = Field(
        default=None,
        max_length=32,
        description="Optional log level filter, e.g. ERROR or WARN.",
    )
    interval_seconds: IntervalSeconds = Field(
        default=DEFAULT_INTERVAL_SECONDS,
        description=(
            "How often the job runs. One of: 10, 30, 60, 300, 600, 1800, 3600."
        ),
    )
    lookback_minutes: int = Field(
        default=DEFAULT_LOOKBACK_MINUTES,
        ge=MIN_LOOKBACK_MINUTES,
        le=MAX_LOOKBACK_MINUTES,
        description="How many minutes back each run searches (1-360).",
    )
    text_contains: str | None = Field(
        default=None,
        max_length=200,
        description="Optional substring the log message must contain.",
    )
    limit: int = Field(
        default=DEFAULT_MONITORING_LIMIT,
        ge=MIN_MONITORING_LIMIT,
        le=MAX_MONITORING_LIMIT,
        description="Maximum number of log rows each run requests (1-500).",
    )

    @field_validator("level", "text_contains")
    @classmethod
    def _normalize_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


# ----------------------------------------------------------------------
# Stored entities
# ----------------------------------------------------------------------
class MonitoringJob(BaseModel):
    """One monitoring job as persisted in SQLite."""

    job_id: str
    service: str
    level: str | None = None
    text_contains: str | None = None
    interval_seconds: int
    lookback_minutes: int
    limit: int = DEFAULT_MONITORING_LIMIT
    status: str = JOB_STATUS_ACTIVE
    created_at: datetime
    updated_at: datetime
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None


class MonitoringRun(BaseModel):
    """One scheduled execution. Stores AGGREGATES only — never raw log rows."""

    id: int | None = None
    job_id: str
    started_at: datetime
    finished_at: datetime
    logs_count: int = 0
    malformed_count: int = 0
    status: str = RUN_STATUS_SUCCESS
    error_message: str | None = None
    duration_ms: int = 0


# ----------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------
class StartMonitoringResult(BaseModel):
    """Result of ``start_monitoring`` (MCP tool and HTTP endpoint)."""

    job_id: str
    status: str
    service: str
    level: str | None = None
    interval_seconds: int
    lookback_minutes: int
    next_run_at: datetime | None = None
    first_run: MonitoringRun | None = None
    message: str = ""


class MonitoringStatusResponse(BaseModel):
    """Result of ``get_monitoring_status`` / the job card."""

    job_id: str
    status: str
    service: str
    level: str | None = None
    interval_seconds: int
    lookback_minutes: int
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    runs_count: int = 0


class MonitoringRunListResponse(BaseModel):
    """Run history for one job (most recent first)."""

    job_id: str
    runs: list[MonitoringRun] = Field(default_factory=list)


class MonitoringSummaryResponse(BaseModel):
    """Deterministic aggregate over the most recent ``last_runs`` runs.

    All aggregates are computed by the backend; the frontend only renders them.
    """

    job_id: str
    service: str
    runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_logs: int = 0
    average_logs_per_run: float = 0.0
    min_logs_per_run: int = 0
    max_logs_per_run: int = 0
    last_run_logs: int = 0
    first_run_at: datetime | None = None
    last_run_at: datetime | None = None
    # Simple, deterministic trend of the last run vs. the window average.
    trend: str = "stable"


class StopMonitoringResult(BaseModel):
    """Result of ``stop_monitoring``."""

    job_id: str
    status: str


# ----------------------------------------------------------------------
# Agent tool-calling flow
# ----------------------------------------------------------------------
MAX_DAY18_MESSAGE_LENGTH = 2000


class Day18ChatRequest(BaseModel):
    """One Day 18 user message."""

    message: str = Field(..., min_length=1, max_length=MAX_DAY18_MESSAGE_LENGTH)


class Day18ToolCallInfo(BaseModel):
    """One MCP monitoring tool call performed during the agent turn."""

    server: str = "monitoring"
    tool: str
    arguments: dict = Field(default_factory=dict)
    result_summary: dict = Field(default_factory=dict)


class Day18TraceStep(BaseModel):
    """One real execution step rendered by the UI as an MCP/agent trace."""

    step: str
    status: str  # "ok" | "error"
    message: str


class Day18ChatResponse(BaseModel):
    """Result of ``POST /api/week4/day18/chat``."""

    answer: str = ""
    tool_calls: list[Day18ToolCallInfo] = Field(default_factory=list)
    trace: list[Day18TraceStep] = Field(default_factory=list)
    # Controlled, secret-free failure message (no stack traces).
    error: str | None = None
