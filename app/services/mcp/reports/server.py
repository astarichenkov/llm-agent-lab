"""Day 20 — Reports MCP server (in-memory transport).

A REAL MCP server (``FastMCP``) exposing exactly one tool::

    save_report  ->  deterministically persist the investigation report

Keeping Reports as its OWN MCP server (rather than reusing the Day 19
pipeline server) makes the multi-server orchestration explicit:

    VictoriaLogs MCP   (search_logs)
    Gitea MCP          (recent_commits / get_commit / get_commit_diff)
    Reports MCP        (save_report)

The server owns no HTTP client and no LLM prompt; it delegates to
:class:`ReportsToolset`.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field, ValidationError

from app.schemas.day20 import InvestigationReportInput

SERVER_NAME = "Reports MCP"
SERVER_VERSION = "1.0"

SAVE_REPORT_DESCRIPTION = (
    "Save the final incident investigation report. Call this ONCE, after you "
    "have gathered evidence from the other MCP tools and formed the "
    "correlation / hypothesis / verification sections. This is deterministic "
    "and read-only for the repository: it only writes a local investigation "
    "report. Provide the human-readable sections; the backend owns the file "
    "format and location."
)


def build_reports_server(toolset: Any) -> FastMCP:
    """Create the Reports MCP server bound to a ``ReportsToolset``."""
    server = FastMCP(SERVER_NAME)

    @server.tool(description=SAVE_REPORT_DESCRIPTION)
    async def save_report(
        title: str = Field(
            "Incident Investigation",
            max_length=300,
            description="Report title.",
        ),
        request: str = Field(
            "", max_length=2000, description="The original user request."
        ),
        incident: str = Field(
            "",
            max_length=8000,
            description="What happened: first observed error, timeframe.",
        ),
        symptoms: str = Field(
            "", max_length=8000, description="Main error patterns / symptoms."
        ),
        relevant_commits: str = Field(
            "", max_length=8000, description="Relevant commits near the incident."
        ),
        relevant_diff: str = Field(
            "", max_length=8000, description="Relevant code/config changes found."
        ),
        correlation: str = Field(
            "", max_length=8000, description="Observed correlation (NOT causation)."
        ),
        hypothesis: str = Field(
            "", max_length=8000, description="Hypothesis that still needs verification."
        ),
        verification_required: str = Field(
            "",
            max_length=8000,
            description="What must be verified before any causal conclusion.",
        ),
        summary: str = Field("", max_length=8000, description="Short summary."),
    ) -> dict[str, Any]:
        """Persist the investigation report (deterministic, no LLM)."""
        try:
            params = InvestigationReportInput(
                title=title,
                request=request,
                incident=incident,
                symptoms=symptoms,
                relevant_commits=relevant_commits,
                relevant_diff=relevant_diff,
                correlation=correlation,
                hypothesis=hypothesis,
                verification_required=verification_required,
                summary=summary,
            )
        except ValidationError as exc:
            return {"error": str(exc), "error_kind": "validation"}
        return await toolset.save_report(params)

    return server


__all__ = ["SERVER_NAME", "SAVE_REPORT_DESCRIPTION", "build_reports_server"]
