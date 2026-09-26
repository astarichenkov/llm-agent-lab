"""Typed input/output schemas for the VictoriaLogs ``search_logs`` MCP tool.

``SearchLogsInput`` is the validated, *structured* input the LLM is allowed to
send. It deliberately does not expose raw LogsQL. The tool function declares
the same constraints via ``pydantic.Field`` (so ``tools/list`` publishes them to
the LLM) and then re-validates through this model before building the query.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Hard limits enforced server-side. They are intentionally conservative: the
# MCP tool must never ask VictoriaLogs for an unbounded amount of data.
MIN_SINCE_MINUTES = 1
MAX_SINCE_MINUTES = 360
# Explicit range (start/end) allows the UI presets up to 24 hours, but never
# more: the LLM must not be able to request a multi-month window.
MAX_WINDOW_MINUTES = 24 * 60
DEFAULT_LIMIT = 100
MIN_LIMIT = 1
MAX_LIMIT = 500


def _as_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime (naive input treated as UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class SearchLogsInput(BaseModel):
    """Structured filters accepted by ``search_logs``."""

    service: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Service name to filter by (exact match on the service field).",
    )
    since_minutes: int = Field(
        default=15,
        ge=MIN_SINCE_MINUTES,
        le=MAX_SINCE_MINUTES,
        description="How many minutes back from now to search (1-360).",
    )
    # Explicit window selected in the UI. When BOTH are set they take
    # precedence over ``since_minutes``. Capped at 24 hours.
    start: datetime | None = Field(
        default=None,
        description="Optional ISO8601 window start (UTC). Used with end.",
    )
    end: datetime | None = Field(
        default=None,
        description="Optional ISO8601 window end (UTC). Used with start.",
    )
    level: str | None = Field(
        default=None,
        max_length=32,
        description="Optional log level filter, e.g. ERROR or WARN.",
    )
    text_contains: str | None = Field(
        default=None,
        max_length=200,
        description="Optional substring the log message must contain.",
    )
    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=MIN_LIMIT,
        le=MAX_LIMIT,
        description="Maximum number of log rows to return (1-500).",
    )

    @model_validator(mode="after")
    def _validate_window(self) -> "SearchLogsInput":
        """Validate the explicit start/end window when it is supplied."""
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be provided together")
        if self.start is not None and self.end is not None:
            start = _as_utc(self.start)
            end = _as_utc(self.end)
            if end <= start:
                raise ValueError("end must be after start")
            if end - start > timedelta(minutes=MAX_WINDOW_MINUTES):
                raise ValueError(
                    f"time window must not exceed {MAX_WINDOW_MINUTES} minutes"
                )
        return self


class NormalizedLog(BaseModel):
    """One VictoriaLogs row in the normalized shape returned to the LLM."""

    timestamp: str | None = None
    message: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)


class SearchLogsResult(BaseModel):
    """Normalized result of a ``search_logs`` call."""

    service: str
    since_minutes: int
    start: str | None = None
    end: str | None = None
    level: str | None = None
    text_contains: str | None = None
    query: str
    count: int
    limit: int
    truncated: bool = False
    malformed_lines: int = 0
    logs: list[NormalizedLog] = Field(default_factory=list)
