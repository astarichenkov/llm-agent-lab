"""Day 20 — safe investigation storage and deterministic Markdown rendering.

Every report written by the Reports MCP ``save_report`` tool lives under ONE
git-ignored root (``data/day20/investigations`` by default). The caller may
supply an investigation id, but the FILESYSTEM PATH is always derived and
validated here:

* the id must match ``[A-Za-z0-9_-]{1,64}`` — this alone rejects ``../``,
  absolute paths, drive letters and arbitrary filenames;
* the resolved directory must stay inside the artifact root.

Only sanitized, bounded report content is written. Secrets never reach this
module; the orchestrator masks tool arguments before they are persisted.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.day20 import (
    INVESTIGATION_ARTIFACT,
    METADATA_ARTIFACT,
    INVESTIGATION_ID_PATTERN,
    InvestigationReportInput,
)
from app.services.mcp.victorialogs.sanitizer import sanitize_value

_ID_RE = re.compile(rf"^(?:{INVESTIGATION_ID_PATTERN})$")


class InvestigationArtifactError(Exception):
    """Controlled investigation storage failure (safe to expose)."""

    kind = "artifact"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def generate_investigation_id(now: datetime | None = None) -> str:
    """Return a safe, sortable, unique id, e.g. ``inv_20260928_120530_ab12cd``."""
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"inv_{stamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def validate_investigation_id(investigation_id: str) -> str:
    """Return the id when it is a safe single path segment, else raise."""
    candidate = (investigation_id or "").strip()
    if not _ID_RE.match(candidate):
        raise InvestigationArtifactError(
            "invalid investigation_id: only letters, digits, '_' and '-' are allowed"
        )
    return candidate


def _section(title: str, body: str) -> list[str]:
    return [f"## {title}", "", body.strip() or "_Not provided._", ""]


def render_investigation_markdown(
    *,
    investigation_id: str,
    created_at: datetime,
    report: InvestigationReportInput,
) -> str:
    """Render ``investigation.md`` deterministically from the report sections."""
    lines: list[str] = [
        f"# {report.title.strip() or 'Incident Investigation'}",
        "",
        f"Investigation ID: {investigation_id}",
        f"Created: {created_at.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
    ]
    lines += _section("Request", report.request)
    lines += _section("Incident", report.incident)
    lines += _section("Symptoms", report.symptoms)
    lines += _section("Relevant commits", report.relevant_commits)
    lines += _section("Relevant diff", report.relevant_diff)
    lines += _section("Correlation", report.correlation)
    lines += _section("Hypothesis", report.hypothesis)
    lines += _section("Verification required", report.verification_required)
    lines += _section("Summary", report.summary)
    lines += [
        "> Correlation is not causation. Commit timestamps do not prove",
        "> deployment time; the points under \"Verification required\" must be",
        "> checked before any causal conclusion is drawn.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


class InvestigationStore:
    """Writes Day 20 investigation artifacts below one git-ignored root."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def investigation_dir(self, investigation_id: str) -> Path:
        safe_id = validate_investigation_id(investigation_id)
        root = self._root.resolve()
        candidate = (root / safe_id).resolve()
        if candidate != root and not candidate.is_relative_to(root):
            raise InvestigationArtifactError("resolved path escapes the artifact root")
        return candidate

    def save(
        self,
        *,
        investigation_id: str,
        report: InvestigationReportInput,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Write ``investigation.md`` + ``metadata.json``.

        Returns a small, secret-free descriptor with the id, root and names.
        """
        directory = self.investigation_dir(investigation_id)
        directory.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc)
        markdown = render_investigation_markdown(
            investigation_id=investigation_id,
            created_at=created_at,
            report=report,
        )
        (directory / INVESTIGATION_ARTIFACT).write_text(markdown, encoding="utf-8")

        payload = dict(metadata)
        payload["investigation_id"] = investigation_id
        payload["created_at"] = created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["artifacts"] = {
            "investigation": INVESTIGATION_ARTIFACT,
            "metadata": METADATA_ARTIFACT,
        }
        (directory / METADATA_ARTIFACT).write_text(
            json.dumps(sanitize_value(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        return {
            "investigation_id": investigation_id,
            "root": str(self._root),
            "investigation": INVESTIGATION_ARTIFACT,
            "metadata": METADATA_ARTIFACT,
        }

    def read(self, investigation_id: str, filename: str) -> str:
        """Read one known artifact, rejecting unknown names and traversal."""
        if filename not in (INVESTIGATION_ARTIFACT, METADATA_ARTIFACT):
            raise InvestigationArtifactError("unknown artifact filename")
        path = self.investigation_dir(investigation_id) / filename
        if not path.is_file():
            raise InvestigationArtifactError("artifact not found")
        return path.read_text(encoding="utf-8")


__all__ = [
    "InvestigationArtifactError",
    "InvestigationStore",
    "generate_investigation_id",
    "render_investigation_markdown",
    "validate_investigation_id",
]
