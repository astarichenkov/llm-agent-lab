"""Day 19 — the Pipeline MCP server (in-memory transport).

This is a REAL MCP server (``FastMCP``) that exposes exactly three tools::

    search_logs   -> get data
    analyze_logs  -> process data
    save_report   -> save data + analysis

Why a SEPARATE server from Day 17/18?

* Day 19 is explicitly about *composing several MCP tools*. Keeping a
  dedicated ``Pipeline MCP`` server makes the composition boundary obvious and
  keeps the Day 20 (multi-server) work cleanly separated.
* The pipeline needs a hard, byte-exact data handoff between steps and it
  writes files. That state lives in the long-running app process, so the
  in-memory MCP transport (the same one Day 18 uses) is the right fit.
* ``search_logs`` here does NOT re-implement VictoriaLogs HTTP: the tool
  handler delegates to :class:`Day19Toolset`, which reuses the Day 17
  ``VictoriaLogsClient`` / ``build_logsql`` / sanitizer / schemas.

The server itself contains no HTTP client, no LLM prompt and no filesystem
path logic — only the three tool definitions.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field, ValidationError

from app.schemas.day19 import (
    DEFAULT_ANALYSIS_QUESTION,
    AnalyzeLogsInput,
    LogAnalysis,
    SaveReportInput,
    SearchLogsPipelineInput,
)
from app.services.mcp.victorialogs.schemas import (
    MAX_LIMIT,
    MAX_SINCE_MINUTES,
    MIN_LIMIT,
    MIN_SINCE_MINUTES,
    NormalizedLog,
)

SERVER_NAME = "Pipeline MCP"
SERVER_VERSION = "1.0"

SEARCH_DESCRIPTION = (
    "Step 1 of the Day 19 pipeline. Search recent VictoriaLogs rows for one "
    "service using safe structured filters (service, time window, level, "
    "optional text). Returns normalized, sanitized rows. This tool only GETS "
    "data; call analyze_logs next with the returned rows."
)

ANALYZE_DESCRIPTION = (
    "Step 2 of the Day 19 pipeline. Analyse the rows produced by search_logs "
    "and return a STRUCTURED analysis (summary, error groups with counts and "
    "examples, possible causes as hypotheses, notable patterns, recommended "
    "checks). It analyses exactly the provided rows and never queries "
    "VictoriaLogs again."
)

SAVE_DESCRIPTION = (
    "Step 3 of the Day 19 pipeline. Deterministically save the sanitized rows "
    "and the structured analysis to disk (raw.jsonl, analysis.md, "
    "metadata.json). No LLM is used. The run_id is validated server-side; a "
    "caller can never choose an arbitrary filesystem path."
)


def build_pipeline_server(toolset: Any) -> FastMCP:
    """Create the Day 19 pipeline MCP server bound to a ``Day19Toolset``.

    ``toolset`` is duck-typed to keep the MCP layer decoupled from the Day 19
    service implementation.
    """
    server = FastMCP(SERVER_NAME)

    @server.tool(description=SEARCH_DESCRIPTION)
    async def search_logs(
        service: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description="Service name to search (exact match on the service field).",
        ),
        since_minutes: int = Field(
            30,
            ge=MIN_SINCE_MINUTES,
            le=MAX_SINCE_MINUTES,
            description="How many minutes back from now (1-360).",
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
            description="Maximum number of rows to return (1-500).",
        ),
    ) -> dict[str, Any]:
        """Search recent stage logs for one service."""
        try:
            params = SearchLogsPipelineInput(
                service=service,
                since_minutes=since_minutes,
                level=level,
                text_contains=text_contains,
                limit=limit,
            )
        except ValidationError as exc:
            return {"error": str(exc), "error_kind": "validation"}
        return await toolset.search_logs(params)

    @server.tool(description=ANALYZE_DESCRIPTION)
    async def analyze_logs(
        service: str = Field(..., min_length=1, max_length=200),
        logs: list[NormalizedLog] = Field(
            default_factory=list,
            description="Sanitized rows returned by the previous search_logs step.",
        ),
        question: str = Field(
            DEFAULT_ANALYSIS_QUESTION,
            min_length=1,
            description="What the analysis should focus on.",
        ),
        since_minutes: int | None = Field(
            None, description="Period metadata copied from search_logs."
        ),
        level: str | None = Field(
            None, description="Level metadata copied from search_logs."
        ),
        start: str | None = None,
        end: str | None = None,
        logs_received: int | None = Field(
            None, description="How many rows search_logs originally received."
        ),
    ) -> dict[str, Any]:
        """Analyse the provided log rows and return a structured analysis."""
        try:
            params = AnalyzeLogsInput(
                service=service,
                question=question,
                logs=logs,
                since_minutes=since_minutes,
                level=level,
                start=start,
                end=end,
                logs_received=logs_received,
            )
        except ValidationError as exc:
            return {"error": str(exc), "error_kind": "validation"}
        return await toolset.analyze_logs(params)

    @server.tool(description=SAVE_DESCRIPTION)
    async def save_report(
        run_id: str = Field(
            ...,
            min_length=1,
            max_length=64,
            description="Backend-generated safe run id (letters, digits, '_' and '-').",
        ),
        service: str = Field(..., min_length=1, max_length=200),
        analysis: LogAnalysis = Field(
            ..., description="The structured analysis from analyze_logs."
        ),
        logs: list[NormalizedLog] = Field(
            default_factory=list,
            description="The sanitized rows from search_logs.",
        ),
        pipeline_id: str | None = None,
        since_minutes: int | None = None,
        level: str | None = None,
        text_contains: str | None = None,
        start: str | None = None,
        end: str | None = None,
        logs_received: int = 0,
        logs_analyzed: int = 0,
        malformed_lines: int = 0,
        truncated: bool = False,
        analysis_truncated: bool = False,
        model: str | None = None,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        """Deterministically save the rows and analysis as artifacts."""
        try:
            params = SaveReportInput(
                run_id=run_id,
                pipeline_id=pipeline_id,
                service=service,
                since_minutes=since_minutes,
                level=level,
                text_contains=text_contains,
                start=start,
                end=end,
                logs_received=logs_received,
                logs_analyzed=logs_analyzed,
                malformed_lines=malformed_lines,
                truncated=truncated,
                analysis_truncated=analysis_truncated,
                model=model,
                duration_ms=duration_ms,
                logs=logs,
                analysis=analysis,
            )
        except ValidationError as exc:
            return {"error": str(exc), "error_kind": "validation"}
        return await toolset.save_report(params)

    return server


__all__ = [
    "SERVER_NAME",
    "ANALYZE_DESCRIPTION",
    "SAVE_DESCRIPTION",
    "SEARCH_DESCRIPTION",
    "build_pipeline_server",
]
