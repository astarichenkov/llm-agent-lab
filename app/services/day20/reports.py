"""Day 20 — the Reports MCP toolset (``save_report``).

The Reports MCP server (``app.services.mcp.reports.server``) is a thin MCP
wrapper around this toolset. The toolset:

* owns the safe :class:`InvestigationStore`;
* keeps the technical investigation context (id, servers/tools used, counts)
  that the orchestrator updates as tool calls happen;
* renders and persists the investigation report.

The LLM only supplies the human-readable report sections. It never chooses a
filesystem path or the investigation id.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.schemas.day20 import InvestigationReportInput
from app.services.day20.artifacts import (
    InvestigationArtifactError,
    InvestigationStore,
    generate_investigation_id,
)
from app.services.mcp.victorialogs.sanitizer import sanitize_text

logger = logging.getLogger("app.services.day20.reports")


class ReportsToolset:
    """Implements the Reports MCP ``save_report`` tool."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: InvestigationStore | None = None,
    ) -> None:
        self._settings = settings
        self._store = store or InvestigationStore(settings.day20_artifact_root)
        self._investigation_id: str | None = None
        self._context: dict[str, Any] = {}
        # Optional opt-in masker (Day 20 "Маскировать данные").
        self._masker: Any | None = None

    def set_masker(self, masker: Any | None) -> None:
        """Install the opt-in identifier masker used before saving the report."""
        self._masker = masker

    # ------------------------------------------------------------------
    # Orchestrator-managed state
    # ------------------------------------------------------------------
    @property
    def store(self) -> InvestigationStore:
        return self._store

    @property
    def investigation_id(self) -> str | None:
        return self._investigation_id

    def begin_investigation(
        self,
        *,
        request: str,
        investigation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Start a new investigation and return its server-generated id."""
        self._investigation_id = investigation_id or generate_investigation_id()
        self._context = {
            "request": sanitize_text(request or ""),
            "servers_used": [],
            "tools_used": [],
            "tool_calls": 0,
            "tool_calls_detail": [],
            "iterations": 0,
            "status": "in_progress",
        }
        if context:
            self._context.update(context)
        return self._investigation_id

    def update_context(self, **updates: Any) -> None:
        """Merge orchestrator state into the context written to metadata."""
        self._context.update(deepcopy(updates))

    def context(self) -> dict[str, Any]:
        return deepcopy(self._context)

    # ------------------------------------------------------------------
    # Tool implementation
    # ------------------------------------------------------------------
    async def save_report(self, params: InvestigationReportInput) -> dict[str, Any]:
        """Persist the investigation report and return its artifact descriptor."""
        if not self._investigation_id:
            return {
                "error": "no active investigation context",
                "error_kind": "artifact",
            }

        safe_report = InvestigationReportInput(
            **{
                key: sanitize_text(value)
                for key, value in params.model_dump().items()
            }
        )
        if self._masker is not None:
            # Mask identifiers (URLs, service/container/host names, IPs, …) in
            # the human-readable report before it is rendered to disk.
            safe_report = InvestigationReportInput(
                **self._masker.mask_value(safe_report.model_dump())
            )
        now = datetime.now(timezone.utc)
        metadata = {
            "status": self._context.get("status", "completed"),
            "request": self._context.get("request", ""),
            "servers_used": list(self._context.get("servers_used", [])),
            "tools_used": list(self._context.get("tools_used", [])),
            "tool_calls": int(self._context.get("tool_calls", 0)),
            "tool_calls_detail": list(self._context.get("tool_calls_detail", [])),
            "iterations": int(self._context.get("iterations", 0)),
            "started_at": self._context.get("started_at"),
            "finished_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": self._context.get("model"),
        }
        try:
            return self._store.save(
                investigation_id=self._investigation_id,
                report=safe_report,
                metadata=metadata,
            )
        except InvestigationArtifactError as exc:
            return {"error": exc.message, "error_kind": exc.kind}
        except OSError as exc:
            logger.exception("Day 20 save_report failed")
            return {
                "error": f"could not write investigation ({exc.__class__.__name__})",
                "error_kind": "artifact",
            }


__all__ = ["ReportsToolset"]
