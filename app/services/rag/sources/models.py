"""Internal models for the Day 24 source-link resolver."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResolvedMessageLink:
    """One Telegram message and its resolved permalink."""

    message_id: int
    url: str | None = None
    is_cited: bool = False


@dataclass
class ResolvedSourceLink:
    """Backend-resolved link for one validated citation.

    ``url`` is ``None`` when no link can be built reliably — the UI then shows
    the source as plain (non-clickable) text instead of inventing a URL.

    ``link_kind``:
    * ``document``     — a manual PDF (optionally with a ``#page=`` fragment);
    * ``message``      — a Telegram permalink to the exact cited message;
    * ``conversation`` — a Telegram anchor to the first message of the chunk
      (the quote could not be mapped to one message).

    ``context_links`` lists every message id of the Telegram chunk with its
    permalink; ``is_cited`` marks the main one.
    """

    url: str | None = None
    cited_message_id: int | None = None
    topic_id: int | None = None
    link_kind: str = ""
    note: str = ""
    context_links: list[ResolvedMessageLink] = field(default_factory=list)
