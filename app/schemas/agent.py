"""Pydantic schemas for the Agent APIs (Day 6 stateless + Day 7 persistent)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.agents.models import Message
from app.schemas.chat import ChatRequest


class AgentInfo(BaseModel):
    """Safe, read-only view of an agent's configuration."""

    agent_id: str
    name: str
    provider: str
    model: str | None = None
    system_prompt: str = ""
    temperature: float
    max_tokens: int
    # ``False`` means reasoning/thinking is disabled for this agent.
    thinking: bool = False


class AgentMessage(BaseModel):
    """One persisted dialog turn."""

    role: str
    content: str
    created_at: str | None = None


class AgentHistoryResponse(BaseModel):
    """Response for ``GET /api/chat/history`` and the generic variant."""

    agent: AgentInfo
    messages: list[Message]
    count: int


class AgentChatResponse(BaseModel):
    """Response for agent-backed chat endpoints."""

    agent_id: str
    answer: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: dict | None = None
    message_count: int


class AgentListResponse(BaseModel):
    agents: list[AgentInfo]


class Day6ChatRequest(ChatRequest):
    """Request for the stateless Day 6 agent.

    Reuses the chat message validation (non-blank, max length). ``provider``
    and ``model`` are optional overrides; when omitted the default agent
    configuration from the environment is used.
    """

    provider: Literal["deepseek", "openrouter"] | None = None
    model: str | None = None


class Day6ChatResponse(BaseModel):
    """Response for ``POST /api/day6/agent/chat``.

    ``stateless`` is always ``True`` here: the request is handled by a fresh
    transient Agent and never touches the persistent Day 7 history.
    """

    agent_id: str
    answer: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: dict | None = None
    stateless: bool = True


class Day6AgentMeta(BaseModel):
    """Metadata for the default Day 6 transient agent (``GET /api/day6/agent``)."""

    agent_id: str
    name: str
    provider: str
    model: str
    max_tokens: int
    thinking: bool = False
    stateless: bool = True
