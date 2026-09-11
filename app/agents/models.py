"""Domain models for the persistent Agent layer."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Current time as an ISO-8601 UTC string (used for ``created_at``)."""
    return datetime.now(timezone.utc).isoformat()


class Message(BaseModel):
    """One conversation turn that is persisted and replayed to the LLM.

    Only ``user`` and ``assistant`` roles are stored. The system prompt lives
    in :class:`~app.agents.config.AgentConfig` and is prepended at request
    time, never persisted as a dialog turn.
    """

    role: Literal["user", "assistant"]
    content: str
    created_at: str = Field(default_factory=utc_now_iso)

    def as_payload(self) -> dict[str, str]:
        """The minimal ``{"role", "content"}`` shape the provider expects."""
        return {"role": self.role, "content": self.content}
