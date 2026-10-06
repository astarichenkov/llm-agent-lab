"""Pydantic schemas for Week 4 / Day 20 — multi-server MCP orchestration.

Day 20 registers SEVERAL MCP servers at once and lets the LLM choose which
tool to call next based on the previous tool results::

    user request
        -> LLM chooses a tool
        -> MCP tool (VictoriaLogs / Gitea / Reports)
        -> result returned to the LLM
        -> LLM chooses the next tool (possibly from another server)
        -> ... -> final answer + saved report

The models here are shared by:

* the Day 20 Reports MCP server tools;
* the orchestration service (agent loop + trace);
* the HTTP endpoints and the UI.

Only sanitized, bounded data ever appears in these models. Secrets
(GITEA_TOKEN, Authorization headers, VictoriaLogs credentials) are never
included.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------------
# Investigation storage
# ----------------------------------------------------------------------
# Mirrors the Day 19 run-id policy: a single safe path segment only. This is
# what makes path traversal / absolute paths impossible.
INVESTIGATION_ID_PATTERN = r"[A-Za-z0-9_-]{1,64}"
INVESTIGATION_ARTIFACT = "investigation.md"
METADATA_ARTIFACT = "metadata.json"
ARTIFACT_FILENAMES: tuple[str, ...] = (INVESTIGATION_ARTIFACT, METADATA_ARTIFACT)

MAX_INVESTIGATION_QUESTION_LENGTH = 2000
MAX_REPORT_FIELD_CHARS = 8000

# Orchestration status values.
STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

# Trace step kinds.
TRACE_KIND_DISCOVERY = "discovery"
TRACE_KIND_TOOL_CALL = "tool_call"
TRACE_KIND_LLM = "llm"
TRACE_KIND_FINAL = "final"
TRACE_KIND_ERROR = "error"

# Trace statuses.
TRACE_STATUS_OK = "ok"
TRACE_STATUS_ERROR = "error"
TRACE_STATUS_TIMEOUT = "timeout"
TRACE_STATUS_SKIPPED = "skipped"


class InvestigationReportInput(BaseModel):
    """Input of the Reports MCP ``save_report`` tool.

    The LLM supplies the human-readable sections; the backend owns the file
    format, the investigation id and the technical metadata. Section text is
    bounded so a runaway model can never write an unbounded report.
    """

    title: str = Field(default="Incident Investigation", max_length=300)
    request: str = Field(default="", max_length=MAX_INVESTIGATION_QUESTION_LENGTH)
    incident: str = Field(default="", max_length=MAX_REPORT_FIELD_CHARS)
    symptoms: str = Field(default="", max_length=MAX_REPORT_FIELD_CHARS)
    relevant_commits: str = Field(default="", max_length=MAX_REPORT_FIELD_CHARS)
    relevant_diff: str = Field(default="", max_length=MAX_REPORT_FIELD_CHARS)
    correlation: str = Field(default="", max_length=MAX_REPORT_FIELD_CHARS)
    hypothesis: str = Field(default="", max_length=MAX_REPORT_FIELD_CHARS)
    verification_required: str = Field(default="", max_length=MAX_REPORT_FIELD_CHARS)
    summary: str = Field(default="", max_length=MAX_REPORT_FIELD_CHARS)

    @field_validator(
        "title",
        "request",
        "incident",
        "symptoms",
        "relevant_commits",
        "relevant_diff",
        "correlation",
        "hypothesis",
        "verification_required",
        "summary",
    )
    @classmethod
    def _strip(cls, value: str) -> str:
        return (value or "").strip()


# ----------------------------------------------------------------------
# MCP server registry metadata
# ----------------------------------------------------------------------
class Day20ServerInfo(BaseModel):
    """One registered MCP server and its discovered tools."""

    name: str
    label: str
    transport: str = "stdio"
    connected: bool = False
    tools_count: int = 0
    tools: list[Any] = Field(default_factory=list)
    error: str | None = None


class Day20ToolRoute(BaseModel):
    """The resolved mapping ``tool_name -> MCP server``."""

    tool: str
    server: str
    label: str = ""
    description: str = ""


class Day20MCPStatusResponse(BaseModel):
    """Result of the Day 20 unified MCP discovery."""

    connected: bool = False
    servers: list[Day20ServerInfo] = Field(default_factory=list)
    routing: list[Day20ToolRoute] = Field(default_factory=list)
    tools_count: int = 0
    error: str | None = None


# ----------------------------------------------------------------------
# Orchestration request / response
# ----------------------------------------------------------------------
class Day20InvestigateRequest(BaseModel):
    """One incident-investigation request.

    The repository and all service endpoints live in server configuration;
    the caller only supplies the natural-language request.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INVESTIGATION_QUESTION_LENGTH,
        description="Natural-language incident investigation request.",
    )
    service: str | None = Field(
        default=None,
        max_length=200,
        description="Optional service hint for the log search.",
    )
    since_minutes: int = Field(
        default=60,
        ge=1,
        le=360,
        description="Log lookback window hint in minutes.",
    )
    save_report: bool = Field(
        default=True,
        description="Allow the agent to persist the investigation report.",
    )
    mask_data: bool = Field(
        default=False,
        description=(
            "When true, mask URLs, service/container/host names and other "
            "identifiers in the tool results sent to the LLM, the final "
            "answer, the trace and the saved report."
        ),
    )


class Day20TraceStep(BaseModel):
    """One real orchestration step, rendered by the UI as a trace.

    ``arguments_safe`` is a bounded, secret-masked view of the tool arguments;
    it never contains a token or Authorization header.
    """

    sequence: int
    kind: str
    server: str | None = None
    server_label: str | None = None
    tool: str | None = None
    arguments_safe: dict[str, Any] = Field(default_factory=dict)
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int | None = None
    result_summary: str = ""
    reason: str = ""
    error: str | None = None


class Day20ToolCallInfo(BaseModel):
    """Compact record of one executed tool call (for UI/routing display)."""

    sequence: int
    server: str
    server_label: str = ""
    tool: str
    status: str
    duration_ms: int | None = None
    result_summary: str = ""


class Day20Artifacts(BaseModel):
    """Locations of the artifacts written by the Reports MCP ``save_report``."""

    investigation_id: str
    root: str
    investigation: str = INVESTIGATION_ARTIFACT
    metadata: str = METADATA_ARTIFACT


class Day20InvestigateResponse(BaseModel):
    """Result of ``POST /api/week4/day20/investigate``."""

    investigation_id: str
    status: str = STATUS_COMPLETED
    answer: str = ""
    servers_used: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    tool_calls: list[Day20ToolCallInfo] = Field(default_factory=list)
    trace: list[Day20TraceStep] = Field(default_factory=list)
    artifacts: Day20Artifacts | None = None
    report_saved: bool = False
    iterations: int = 0
    error: str | None = None
