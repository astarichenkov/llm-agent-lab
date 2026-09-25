"""Pydantic schemas for Day 16 — MCP connection & tool discovery.

Day 16 connects the backend to a LOCAL MCP server over ``stdio``, performs the
MCP initialization handshake and calls ``tools/list``. These models are the
transport-agnostic shape returned to the frontend: the tool list is NEVER
hardcoded in the UI, it is exactly what the MCP server reported.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MCPToolInfo(BaseModel):
    """One tool as reported by the MCP server's ``tools/list`` response."""

    name: str
    description: str = ""
    # The tool's JSON Schema, passed through unchanged from the MCP server.
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPTraceStep(BaseModel):
    """One stage of the Day 16 connection, for the educational MCP Trace."""

    step: str
    status: str  # "ok" | "error"
    message: str


class MCPServerInfo(BaseModel):
    """Identity of the MCP server the backend connected to."""

    name: str
    transport: str = "stdio"


class MCPStatusResponse(BaseModel):
    """Result of a Day 16 MCP connection + ``tools/list`` call."""

    connected: bool
    server: MCPServerInfo
    tools_count: int = 0
    tools: list[MCPToolInfo] = Field(default_factory=list)
    trace: list[MCPTraceStep] = Field(default_factory=list)
    # Human-readable, controlled failure message (no stack traces).
    error: str | None = None
