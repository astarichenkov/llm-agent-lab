"""Day 19 — safe artifact storage and deterministic Markdown rendering.

Every artifact written by ``save_report`` lives under ONE git-ignored root
(``data/day19/runs`` by default). The caller supplies a ``run_id`` but the
FILESYSTEM PATH is always derived and validated here:

* ``run_id`` must match ``[A-Za-z0-9_-]{1,64}`` — this alone rejects
  ``../``, absolute paths, drive letters and arbitrary filenames;
* the resolved directory must stay inside the artifact root (defence in
  depth, in case the root itself is a symlink).

Only sanitized, normalized log rows are written. Raw HTTP bytes, headers and
credentials never reach this module.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.schemas.day19 import (
    ANALYSIS_ARTIFACT,
    METADATA_ARTIFACT,
    RAW_ARTIFACT,
    ErrorGroup,
    LogAnalysis,
    NormalizedLog,
    RUN_ID_PATTERN,
)
from app.services.mcp.victorialogs.sanitizer import sanitize_value

_RUN_ID_RE = re.compile(rf"^(?:{RUN_ID_PATTERN})$")


class ArtifactError(Exception):
    """Controlled artifact failure (safe to expose)."""

    kind = "artifact"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def generate_run_id(now: datetime | None = None) -> str:
    """Return a safe, sortable, unique run id, e.g. ``run_20260928_120530_ab12cd``."""
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"run_{stamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` when it is a safe single path segment, else raise."""
    candidate = (run_id or "").strip()
    if not _RUN_ID_RE.match(candidate):
        raise ArtifactError(
            "invalid run_id: only letters, digits, '_' and '-' are allowed"
        )
    return candidate


def _sanitize_logs(logs: Iterable[NormalizedLog | dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize + sanitize every row before it is written to disk."""
    cleaned: list[dict[str, Any]] = []
    for item in logs or []:
        if isinstance(item, NormalizedLog):
            row = item.model_dump()
        elif isinstance(item, dict):
            row = item
        else:
            continue
        cleaned.append(
            {
                "timestamp": sanitize_value(row.get("timestamp")),
                "message": sanitize_value(row.get("message")),
                "fields": sanitize_value(row.get("fields") or {}),
            }
        )
    return cleaned


def render_analysis_markdown(
    *,
    run_id: str,
    created_at: datetime,
    service: str,
    since_minutes: int | None,
    level: str | None,
    logs_received: int,
    logs_analyzed: int,
    malformed_lines: int,
    analysis: LogAnalysis,
) -> str:
    """Render ``analysis.md`` deterministically from the structured analysis.

    The LLM never returns the Markdown file itself; this renderer owns the
    format so the artifact stays stable and easy to diff.
    """
    period = (
        f"last {since_minutes} minutes"
        if since_minutes is not None
        else "selected window"
    )
    lines: list[str] = [
        "# Stage Log Analysis",
        "",
        f"Run ID: {run_id}",
        f"Created: {created_at.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Service: {service}",
        f"Period: {period}",
        f"Level: {level or 'ANY'}",
        f"Logs received: {logs_received}",
        f"Logs analyzed: {logs_analyzed}",
    ]
    if malformed_lines:
        lines.append(f"Malformed lines skipped: {malformed_lines}")
    lines += [
        "",
        "## Summary",
        "",
        analysis.summary.strip(),
        "",
        "## Error groups",
        "",
    ]

    if not analysis.error_groups:
        lines += ["_No recurring error groups were identified in the analyzed rows._", ""]
    for group in analysis.error_groups:
        lines += [
            f"### {group.pattern}",
            "",
            f"Occurrences: {group.count}",
            "",
        ]
        for example in group.examples:
            lines.append(f"- `{example}`")
        lines.append("")

    lines += ["## Possible causes", ""]
    lines += _markdown_list(analysis.possible_causes)
    lines += ["## Notable patterns", ""]
    lines += _markdown_list(analysis.notable_patterns)
    lines += ["## Recommended checks", ""]
    lines += _markdown_list(analysis.recommended_checks)
    return "\n".join(lines).rstrip() + "\n"


def _markdown_list(items: list[str]) -> list[str]:
    if not items:
        return ["_None._", ""]
    return [f"- {item}" for item in items] + [""]


class ArtifactStore:
    """Writes Day 19 artifacts below a single, git-ignored root directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def run_dir(self, run_id: str) -> Path:
        """Return the validated run directory (never outside the root)."""
        safe_id = validate_run_id(run_id)
        root = self._root.resolve()
        candidate = (root / safe_id).resolve()
        if candidate != root and not candidate.is_relative_to(root):
            raise ArtifactError("resolved path escapes the artifact root")
        return candidate

    def save(
        self,
        *,
        run_id: str,
        metadata: dict[str, Any],
        logs: Iterable[NormalizedLog | dict[str, Any]],
        analysis: LogAnalysis,
    ) -> dict[str, Any]:
        """Write ``raw.jsonl``, ``analysis.md`` and ``metadata.json``.

        Returns a small, secret-free descriptor with the run id, the root and
        the three filenames.
        """
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        clean_logs = _sanitize_logs(logs)
        raw_path = run_dir / RAW_ARTIFACT
        with raw_path.open("w", encoding="utf-8") as handle:
            for row in clean_logs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        created_at = datetime.now(timezone.utc)
        markdown = render_analysis_markdown(
            run_id=run_id,
            created_at=created_at,
            service=str(metadata.get("service") or ""),
            since_minutes=metadata.get("since_minutes"),
            level=metadata.get("level"),
            logs_received=int(metadata.get("logs_received") or 0),
            logs_analyzed=int(metadata.get("logs_analyzed") or 0),
            malformed_lines=int(metadata.get("malformed_lines") or 0),
            analysis=analysis,
        )
        (run_dir / ANALYSIS_ARTIFACT).write_text(markdown, encoding="utf-8")

        payload = dict(metadata)
        payload["run_id"] = run_id
        payload["created_at"] = created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["artifacts"] = {
            "raw": RAW_ARTIFACT,
            "analysis": ANALYSIS_ARTIFACT,
            "metadata": METADATA_ARTIFACT,
        }
        (run_dir / METADATA_ARTIFACT).write_text(
            json.dumps(sanitize_value(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        return {
            "run_id": run_id,
            "root": str(self._root),
            "raw": RAW_ARTIFACT,
            "analysis": ANALYSIS_ARTIFACT,
            "metadata": METADATA_ARTIFACT,
        }

    def read(self, run_id: str, filename: str) -> str:
        """Read one known artifact, rejecting unknown names and traversal."""
        if filename not in (RAW_ARTIFACT, ANALYSIS_ARTIFACT, METADATA_ARTIFACT):
            raise ArtifactError("unknown artifact filename")
        path = self.run_dir(run_id) / filename
        if not path.is_file():
            raise ArtifactError("artifact not found")
        return path.read_text(encoding="utf-8")


# Re-export for convenience of the pipeline toolset.
__all__ = [
    "ArtifactError",
    "ArtifactStore",
    "ErrorGroup",
    "generate_run_id",
    "render_analysis_markdown",
    "validate_run_id",
]
