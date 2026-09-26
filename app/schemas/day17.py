"""Pydantic schemas for Day 17 — VictoriaLogs MCP tool + agent flow.

The response deliberately exposes the *real* MCP tool call (server, tool,
arguments) and a real execution trace, so the UI can prove that the answer
came from an MCP ``tools/call`` and not from a direct HTTP call.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator

MAX_DAY17_MESSAGE_LENGTH = 2000
MAX_DAY17_WINDOW_MINUTES = 24 * 60


class Day17ChatRequest(BaseModel):
    """One Day 17 user message plus the UI-selected time window."""

    message: str = Field(..., min_length=1, max_length=MAX_DAY17_MESSAGE_LENGTH)
    # Optional explicit window selected in the UI. When supplied, the backend
    # guarantees search_logs is called with exactly this start/end.
    start: datetime | None = None
    end: datetime | None = None

    @model_validator(mode="after")
    def _validate_window(self) -> "Day17ChatRequest":
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be provided together")
        if self.start is not None and self.end is not None:
            start = self.start
            end = self.end
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            if end <= start:
                raise ValueError("end must be after start")
            window_minutes = (end - start).total_seconds() / 60
            if window_minutes > MAX_DAY17_WINDOW_MINUTES:
                raise ValueError(
                    f"time window must not exceed {MAX_DAY17_WINDOW_MINUTES} minutes"
                )
        return self


class Day17ToolCallInfo(BaseModel):
    """One MCP tool call performed during the agent turn."""

    server: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)


class Day17TraceStep(BaseModel):
    """One real execution step, rendered by the UI as an MCP trace."""

    step: str
    status: str  # "ok" | "error"
    message: str


class Day17ChatResponse(BaseModel):
    """Result of ``POST /api/week4/day17/chat``."""

    answer: str = ""
    tool_calls: list[Day17ToolCallInfo] = Field(default_factory=list)
    trace: list[Day17TraceStep] = Field(default_factory=list)
    # Controlled, secret-free failure message (no stack traces).
    error: str | None = None
