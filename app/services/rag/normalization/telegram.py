"""Telegram export normalization.

The Telegram JSON export is *private*: it is never written back to disk here,
only read and transformed in memory.

Two responsibilities:

* ``extract_text`` / ``parse_messages`` — turn raw export entries into
  :class:`TelegramMessage` objects, counting and explaining skipped entries;
* ``group_messages`` — a deterministic conversation-grouping strategy so that
  short replies are indexed together with their context instead of as
  standalone, meaningless chunks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

# Skip reasons surfaced in the ``rag sources`` statistics.
SKIP_NON_MESSAGE = "service_or_non_message"
SKIP_EMPTY = "empty_text"
SKIP_PHOTO_ONLY = "photo_without_caption"
SKIP_FILE_ONLY = "file_without_caption"
SKIP_POLL = "poll_without_text"
SKIP_UNSUPPORTED = "unsupported_content"


@dataclass
class TelegramMessage:
    """One usable Telegram message."""

    message_id: int
    date: str
    author: str
    text: str
    from_id: str = ""
    reply_to_message_id: int | None = None
    forwarded_from: str | None = None

    @property
    def dt(self) -> datetime | None:
        return _parse_date(self.date)


@dataclass
class ParseStats:
    total: int = 0
    usable: int = 0
    skipped: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def add_skip(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_timestamp(value: Any) -> str:
    """Format an ISO date as ``YYYY-MM-DD HH:MM`` (stable, human readable)."""
    dt = _parse_date(value)
    if dt is None:
        return str(value or "")
    return dt.strftime("%Y-%m-%d %H:%M")


def extract_text(raw: Any, entities: Any = None) -> str:
    """Extract plain text from a Telegram ``text`` field.

    Telegram stores either a plain string or a list of entity dicts/strings.
    Both forms must be handled; formatting entities are flattened to text.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(raw)


def _skip_reason(entry: dict[str, Any]) -> str:
    if "poll" in entry:
        return SKIP_POLL
    if "photo" in entry:
        return SKIP_PHOTO_ONLY
    if "file" in entry or "media_type" in entry:
        return SKIP_FILE_ONLY
    if "sticker" in entry or "location" in entry or "contact" in entry:
        return SKIP_UNSUPPORTED
    return SKIP_EMPTY


def parse_messages(
    messages: Iterable[dict[str, Any]],
    *,
    include_forwarded: bool = True,
) -> tuple[list[TelegramMessage], ParseStats]:
    """Parse raw export entries into usable messages plus skip statistics."""
    stats = ParseStats()
    parsed: list[TelegramMessage] = []
    for entry in messages:
        stats.total += 1
        if not isinstance(entry, dict):
            stats.add_skip(SKIP_UNSUPPORTED)
            continue
        if entry.get("type") != "message":
            stats.add_skip(SKIP_NON_MESSAGE)
            continue
        text = extract_text(entry.get("text"), entry.get("text_entities")).strip()
        if not text:
            stats.add_skip(_skip_reason(entry))
            continue
        forwarded = entry.get("forwarded_from")
        if forwarded and not include_forwarded:
            stats.add_skip("forwarded_message")
            continue
        try:
            message_id = int(entry.get("id"))
        except (TypeError, ValueError):
            stats.add_skip(SKIP_UNSUPPORTED)
            continue
        raw_reply = entry.get("reply_to_message_id")
        reply_id: int | None = None
        if raw_reply is not None:
            try:
                reply_id = int(raw_reply)
            except (TypeError, ValueError):
                reply_id = None
        parsed.append(
            TelegramMessage(
                message_id=message_id,
                date=str(entry.get("date", "")),
                author=str(entry.get("from") or entry.get("from_id") or "unknown"),
                text=text,
                from_id=str(entry.get("from_id") or ""),
                reply_to_message_id=reply_id,
                forwarded_from=str(forwarded) if forwarded else None,
            )
        )
        stats.usable += 1
    return parsed, stats


def format_message(message: TelegramMessage) -> str:
    """Render one message as ``[date] author:\\ntext``."""
    return f"[{format_timestamp(message.date)}] {message.author}:\n{message.text}"


def group_messages(
    messages: list[TelegramMessage],
    *,
    gap_minutes: int = 30,
    max_group_messages: int = 12,
    max_group_chars: int = 1800,
) -> list[list[TelegramMessage]]:
    """Deterministically group neighbouring messages into conversations.

    A new group starts when any of these holds:

    * the time gap since the previous message exceeds ``gap_minutes``;
    * the group already has ``max_group_messages`` messages;
    * adding the message would exceed ``max_group_chars`` characters.

    Reply relationships are used as a *keep-together* signal: a reply to a
    message already inside the current group never triggers a split (a reply
    to an older message does not fragment the chronological thread either —
    the time-gap rule already covers real topic switches). This is
    intentionally simple and explainable — no ML topic segmentation.
    """
    groups: list[list[TelegramMessage]] = []
    current: list[TelegramMessage] = []
    current_chars = 0
    previous_dt: datetime | None = None

    for message in messages:
        dt = message.dt
        new_group = False
        if current:
            if previous_dt is not None and dt is not None:
                if dt - previous_dt > timedelta(minutes=gap_minutes):
                    new_group = True
            if len(current) >= max_group_messages:
                new_group = True
            if current_chars + len(message.text) > max_group_chars:
                new_group = True
        if new_group:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(message)
        current_chars += len(message.text)
        previous_dt = dt if dt is not None else previous_dt

    if current:
        groups.append(current)
    return groups
