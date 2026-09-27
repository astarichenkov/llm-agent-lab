"""Pydantic schemas for Week 4 / Day 19 — MCP tool composition pipeline.

Day 19 composes THREE MCP tools into one automatic pipeline::

    search_logs  ->  analyze_logs  ->  save_report

The models here are shared by three callers:

* the Day 19 pipeline MCP server tools (``app/services/mcp/pipeline/server.py``),
  which publish their JSON schemas to the client and validate the arguments;
* the Day 19 toolset that really performs search / analysis / persistence;
* the HTTP endpoint and the UI.

Only sanitized, normalized log rows ever appear in these models. The pipeline
is bounded on purpose: a single request can never push an unbounded payload
into the LLM or onto disk.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Reuse the Day 17 normalized row shape and search bounds. Re-using the exact
# same model guarantees the pipeline ``search_logs`` tool accepts and returns
# the same shape as the Day 17 VictoriaLogs tool (no divergence).
from app.services.mcp.victorialogs.schemas import (
    MAX_LIMIT,
    MAX_SINCE_MINUTES,
    MIN_LIMIT,
    MIN_SINCE_MINUTES,
    NormalizedLog,
)

# ----------------------------------------------------------------------
# Hard analysis limits (enforced in code, not just in the prompt)
# ----------------------------------------------------------------------
# The model is only ever allowed to see a bounded number of log rows, each
# truncated to a bounded number of characters. This prevents accidentally
# sending 500 huge lines to the provider.
MAX_ANALYSIS_LOGS = 100
MAX_LOG_MESSAGE_CHARS = 2000
MAX_ANALYSIS_INPUT_CHARS = 200_000
# Bounded lists in the structured analysis.
MAX_ERROR_GROUPS = 25
MAX_ANALYSIS_ITEMS = 25
MAX_PATTERN_CHARS = 400
MAX_ITEM_CHARS = 800

# Day 19 artifact names. They are the ONLY filenames the artifact endpoint will
# ever serve; everything else is rejected.
RAW_ARTIFACT = "raw.jsonl"
ANALYSIS_ARTIFACT = "analysis.md"
METADATA_ARTIFACT = "metadata.json"
ARTIFACT_FILENAMES: tuple[str, ...] = (
    RAW_ARTIFACT,
    ANALYSIS_ARTIFACT,
    METADATA_ARTIFACT,
)

# A run id only ever contains ``[A-Za-z0-9_-]``. This is the single check that
# makes path traversal, absolute paths and drive letters impossible.
RUN_ID_PATTERN = r"[A-Za-z0-9_-]{1,64}"

DEFAULT_ANALYSIS_QUESTION = "Найди основные типы ошибок и возможные причины"
MAX_DAY19_QUESTION_LENGTH = 2000

PIPELINE_STATUS_COMPLETED = "completed"
PIPELINE_STATUS_FAILED = "failed"
TRACE_STATUS_OK = "ok"
TRACE_STATUS_ERROR = "error"
TRACE_STATUS_SKIPPED = "skipped"


class Day19PipelineRequest(BaseModel):
    """Validated parameters for one Day 19 pipeline run."""

    service: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Service whose logs are searched.",
    )
    level: str | None = Field(
        default=None,
        max_length=32,
        description="Optional log level filter, e.g. ERROR.",
    )
    since_minutes: int = Field(
        default=30,
        ge=MIN_SINCE_MINUTES,
        le=MAX_SINCE_MINUTES,
        description="How many minutes back to search (1-360).",
    )
    limit: int = Field(
        default=100,
        ge=MIN_LIMIT,
        le=MAX_LIMIT,
        description="Maximum log rows to request (1-500). Default 100 keeps the LLM context small.",
    )
    text_contains: str | None = Field(
        default=None,
        max_length=200,
        description="Optional substring the log message must contain.",
    )
    question: str = Field(
        default=DEFAULT_ANALYSIS_QUESTION,
        min_length=1,
        max_length=MAX_DAY19_QUESTION_LENGTH,
        description="What the structured analysis should focus on.",
    )

    @field_validator("level", "text_contains")
    @classmethod
    def _normalize_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("question")
    @classmethod
    def _normalize_question(cls, value: str) -> str:
        cleaned = (value or "").strip()
        return cleaned or DEFAULT_ANALYSIS_QUESTION


# ----------------------------------------------------------------------
# Pipeline MCP tool inputs
# ----------------------------------------------------------------------
class SearchLogsPipelineInput(BaseModel):
    """Input of the pipeline ``search_logs`` tool (same shape as Day 17)."""

    service: str = Field(..., min_length=1, max_length=200)
    since_minutes: int = Field(default=30, ge=MIN_SINCE_MINUTES, le=MAX_SINCE_MINUTES)
    level: str | None = Field(default=None, max_length=32)
    text_contains: str | None = Field(default=None, max_length=200)
    limit: int = Field(default=100, ge=MIN_LIMIT, le=MAX_LIMIT)


class AnalyzeLogsInput(BaseModel):
    """Input of the pipeline ``analyze_logs`` tool.

    ``logs`` MUST be the sanitized rows produced by the previous
    ``search_logs`` step. The tool never re-queries VictoriaLogs.
    """

    service: str = Field(..., min_length=1, max_length=200)
    question: str = Field(default=DEFAULT_ANALYSIS_QUESTION, min_length=1)
    logs: list[NormalizedLog] = Field(default_factory=list)
    since_minutes: int | None = None
    level: str | None = None
    start: str | None = None
    end: str | None = None
    logs_received: int | None = None

    @model_validator(mode="after")
    def _require_logs(self) -> "AnalyzeLogsInput":
        # An empty search result is still a valid scenario but it must be
        # explicit: the backend uses this to produce a controlled analysis
        # instead of silently inventing content.
        return self


class LogAnalysis(BaseModel):
    """Validated structured analysis returned by ``analyze_logs``."""

    summary: str = Field(..., min_length=1, max_length=4000)
    error_groups: list["ErrorGroup"] = Field(default_factory=list)
    possible_causes: list[str] = Field(default_factory=list)
    notable_patterns: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)

    @field_validator("error_groups")
    @classmethod
    def _cap_groups(cls, value: list["ErrorGroup"]) -> list["ErrorGroup"]:
        return value[:MAX_ERROR_GROUPS]

    @field_validator("possible_causes", "notable_patterns", "recommended_checks")
    @classmethod
    def _cap_items(cls, value: list[str]) -> list[str]:
        bounded: list[str] = []
        for item in value[:MAX_ANALYSIS_ITEMS]:
            text = str(item).strip()
            if text:
                bounded.append(text[:MAX_ITEM_CHARS])
        return bounded


class ErrorGroup(BaseModel):
    """One recurring error pattern found in the analyzed logs."""

    pattern: str = Field(..., min_length=1, max_length=MAX_PATTERN_CHARS)
    count: int = Field(default=0, ge=0)
    examples: list[str] = Field(default_factory=list)

    @field_validator("examples")
    @classmethod
    def _cap_examples(cls, value: list[str]) -> list[str]:
        return [str(item).strip()[:MAX_ITEM_CHARS] for item in value[:3] if str(item).strip()]


LogAnalysis.model_rebuild()


class SaveReportInput(BaseModel):
    """Input of the pipeline ``save_report`` tool (deterministic, no LLM).

    ``logs`` and ``analysis`` are exactly the objects handed over by the two
    previous pipeline steps. ``run_id`` is generated by the backend and is
    re-validated here so it can never control the filesystem path.
    """

    run_id: str = Field(..., min_length=1, max_length=64)
    pipeline_id: str | None = Field(default=None, max_length=64)
    service: str = Field(..., min_length=1, max_length=200)
    since_minutes: int | None = None
    level: str | None = None
    text_contains: str | None = None
    start: str | None = None
    end: str | None = None
    logs_received: int = Field(default=0, ge=0)
    logs_analyzed: int = Field(default=0, ge=0)
    malformed_lines: int = Field(default=0, ge=0)
    truncated: bool = False
    analysis_truncated: bool = False
    model: str | None = Field(default=None, max_length=200)
    duration_ms: int | None = Field(default=None, ge=0)
    logs: list[NormalizedLog] = Field(default_factory=list)
    analysis: LogAnalysis


# ----------------------------------------------------------------------
# Pipeline MCP tool outputs / HTTP responses
# ----------------------------------------------------------------------
class Day19SearchSummary(BaseModel):
    """Compact, safe summary of the search step (no raw body is exposed)."""

    service: str
    since_minutes: int
    level: str | None = None
    count: int = 0
    limit: int = 100
    truncated: bool = False
    malformed_lines: int = 0
    start: str | None = None
    end: str | None = None


class Day19AnalysisSummary(BaseModel):
    """Structured analysis exposed to the API/UI (already validated)."""

    summary: str
    error_groups_count: int = 0
    error_groups: list[ErrorGroup] = Field(default_factory=list)
    possible_causes: list[str] = Field(default_factory=list)
    notable_patterns: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)
    logs_analyzed: int = 0
    analysis_truncated: bool = False


class Day19Artifacts(BaseModel):
    """Locations of the artifacts written by ``save_report``."""

    run_id: str
    root: str
    raw: str = RAW_ARTIFACT
    analysis: str = ANALYSIS_ARTIFACT
    metadata: str = METADATA_ARTIFACT


class Day19TraceStep(BaseModel):
    """One real execution step of the pipeline (backend-generated)."""

    step: str
    status: str  # "ok" | "error" | "skipped"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class Day19PipelineResponse(BaseModel):
    """Result of ``POST /api/week4/day19/pipeline``."""

    pipeline_id: str
    status: str
    answer: str = ""
    search: Day19SearchSummary | None = None
    analysis: Day19AnalysisSummary | None = None
    artifacts: Day19Artifacts | None = None
    trace: list[Day19TraceStep] = Field(default_factory=list)
    # Controlled, secret-free failure message (no stack traces).
    error: str | None = None
