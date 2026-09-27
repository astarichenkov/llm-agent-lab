"""Day 19 — bounded, sanitized, structured LLM log analysis.

``analyze_logs`` is the only pipeline step that calls the LLM. This module is
the pure, unit-testable core of it:

* apply hard limits to the incoming rows (``MAX_ANALYSIS_LOGS``,
  ``MAX_LOG_MESSAGE_CHARS``, ``MAX_ANALYSIS_INPUT_CHARS``);
* build a strict prompt that separates observations from hypotheses and
  demands a single JSON object;
* parse + validate the provider answer with the ``LogAnalysis`` Pydantic
  schema (a free-form JSON answer is never trusted);
* map ANY failure (invalid JSON, schema violation, empty answer, timeout,
  API error) to a controlled ``AnalysisError`` so the pipeline can stop
  before the deterministic ``save_report`` step.

The provider call is injected, so tests never touch the real DeepSeek API.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from pydantic import ValidationError

from app.schemas.day19 import (
    MAX_ANALYSIS_INPUT_CHARS,
    MAX_ANALYSIS_LOGS,
    MAX_LOG_MESSAGE_CHARS,
    LogAnalysis,
    NormalizedLog,
)
from app.services.mcp.victorialogs.sanitizer import sanitize_value
from app.services.structured_json import parse_json_object


class AnalysisError(Exception):
    """Controlled analysis failure (safe to expose, no stack traces)."""

    kind = "analysis"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


ANALYSIS_SYSTEM_PROMPT = (
    "You are an SRE log-analysis assistant. You receive a BOUNDED, already "
    "sanitized set of log rows for ONE service and ONE time window, plus the "
    "user's question. Analyse ONLY the provided rows.\n\n"
    "Rules:\n"
    "- Use ONLY facts visible in the provided logs. Never invent log lines, "
    "timestamps, hosts, error codes or counts.\n"
    "- Clearly separate OBSERVATIONS (what is directly present in the logs) "
    "from HYPOTHESES (possible causes).\n"
    "- 'possible_causes' MUST be phrased as hypotheses/possibilities, never "
    "as confirmed facts.\n"
    "- Do not include secrets, tokens, credentials, cookies or URLs.\n"
    "- 'error_groups[].pattern' is a short normalized description of a "
    "recurring problem; 'count' is the number of rows you actually matched "
    "(integer >= 0); 'examples' are 0-3 short sanitized excerpts.\n"
    "- Return a SINGLE strict JSON object with EXACTLY this shape and no "
    "extra keys:\n"
    "{\n"
    '  "summary": "string",\n'
    '  "error_groups": ['
    '{"pattern": "string", "count": 0, "examples": ["string"]}],\n'
    '  "possible_causes": ["string"],\n'
    '  "notable_patterns": ["string"],\n'
    '  "recommended_checks": ["string"]\n'
    "}\n"
    "- Output ONLY the JSON object: no markdown fences, no prose.\n"
)


def _bounded_message(message: Any) -> str:
    """Truncate one log message and normalize ``None`` to an empty string."""
    if message is None:
        return ""
    text = str(message)
    if len(text) > MAX_LOG_MESSAGE_CHARS:
        return text[:MAX_LOG_MESSAGE_CHARS]
    return text


def select_logs_for_analysis(
    logs: Iterable[NormalizedLog | dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, bool]:
    """Apply the analysis limits.

    Returns ``(selected_rows, logs_analyzed, truncated)`` where each selected
    row has a sanitized, length-bounded message. Rows are taken in the order
    returned by ``search_logs`` (VictoriaLogs returns newest-first in the
    deployments we target); the hard character budget can stop the selection
    earlier than the row cap.
    """
    selected: list[dict[str, Any]] = []
    used_chars = 0
    truncated = False
    for item in logs or []:
        if len(selected) >= MAX_ANALYSIS_LOGS:
            truncated = True
            break
        if isinstance(item, NormalizedLog):
            row = item.model_dump()
        elif isinstance(item, dict):
            row = item
        else:
            continue
        message = _bounded_message(sanitize_value(row.get("message")))
        if len(message) < len(str(row.get("message") or "")):
            truncated = True
        entry = {
            "timestamp": sanitize_value(row.get("timestamp")),
            "message": message,
            "fields": sanitize_value(row.get("fields") or {}),
        }
        entry_chars = len(json.dumps(entry, ensure_ascii=False))
        if selected and used_chars + entry_chars > MAX_ANALYSIS_INPUT_CHARS:
            truncated = True
            break
        selected.append(entry)
        used_chars += entry_chars
    return selected, len(selected), truncated


def build_analysis_messages(
    *,
    service: str,
    question: str,
    logs: Iterable[NormalizedLog | dict[str, Any]],
    since_minutes: int | None = None,
    level: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[list[dict[str, str]], int, bool]:
    """Build one analysis request and report how many rows were used."""
    selected, analyzed, truncated = select_logs_for_analysis(logs)
    window = {
        "since_minutes": since_minutes,
        "level": level,
        "start": start,
        "end": end,
    }
    user_content = (
        f"Service: {service}\n"
        f"Window: {json.dumps(window, ensure_ascii=False)}\n"
        f"User question: {question}\n"
        f"Rows provided: {len(selected)}\n\n"
        "Log rows (JSON array, already sanitized):\n"
        + json.dumps(selected, ensure_ascii=False)
        + "\n\nReturn the JSON analysis now."
    )
    return (
        [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        analyzed,
        truncated,
    )


def parse_analysis(raw: str) -> LogAnalysis:
    """Parse + validate the provider answer; raise ``AnalysisError`` on error."""
    if raw is None or not str(raw).strip():
        raise AnalysisError("LLM returned an empty analysis")
    try:
        data = parse_json_object(raw)
    except ValueError as exc:
        raise AnalysisError("LLM returned invalid JSON for the analysis") from exc
    try:
        return LogAnalysis(**data)
    except ValidationError as exc:
        raise AnalysisError(
            "LLM analysis did not match the required schema"
        ) from exc


async def run_structured_analysis(
    provider: Any,
    *,
    messages: list[dict[str, str]],
    model: str | None,
    max_tokens: int = 1500,
) -> LogAnalysis:
    """Call the provider and return a validated ``LogAnalysis``.

    Provider failures are mapped to ``AnalysisError``. The raw provider text
    is never returned to the caller.
    """
    try:
        content, _finish, _usage = await provider.generate(
            messages,
            model=model,
            temperature=0.2,
            max_tokens=max_tokens,
            thinking=False,
        )
    except Exception as exc:  # noqa: BLE001 - controlled pipeline failure
        raise AnalysisError(
            f"analysis model call failed ({exc.__class__.__name__})"
        ) from exc
    return parse_analysis(content)


__all__ = [
    "AnalysisError",
    "ANALYSIS_SYSTEM_PROMPT",
    "build_analysis_messages",
    "parse_analysis",
    "run_structured_analysis",
    "select_logs_for_analysis",
]
