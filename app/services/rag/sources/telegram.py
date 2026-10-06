"""Telegram source links.

The LLM never produces a URL. The backend reads the **real** Telegram export
configured by ``RAG_TELEGRAM_EXPORT_PATH`` and resolves links from validated
evidence only.

URL strategies, in priority order:

* **A — configured base** (``RAG_TELEGRAM_CHAT_LINK_BASE``):
  ``<base>/<message_id>``. Highest priority and the most reliable option.
* **B — public username** taken from the export's top-level ``username``
  (never guessed from the chat title): ``https://t.me/<username>/<mid>``.
* **C — private/supergroup form**: ``https://t.me/c/<internal_id>/<mid>``.
  Only applied when the export's ``type`` is a supergroup/channel and the id
  is a real Telegram id (a ``-100`` prefix is stripped; basic groups / user
  ids are rejected).
* **D — unavailable**: return ``None`` instead of a potentially wrong link.

Quote -> message mapping: the loader renders each Telegram message as
``[date] author:\\ntext`` and the chunk text is the concatenation of those
blocks. To find the message a quote came from, the resolver reconstructs the
formatted blocks for the chunk's ``message_ids`` and matches the (whitespace
normalized) quote against each one. If no single message contains the quote,
the first message id is used as a *conversation anchor* (link kind
``conversation``) — clearly labelled as such, never as the cited message.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from app.services.rag.normalization.telegram import format_message, parse_messages
from app.services.rag.normalization.whitespace import normalize_whitespace

logger = logging.getLogger("app.services.rag.sources.telegram")

_SUPERGROUP_TYPES = {
    "supergroup",
    "public_supergroup",
    "private_supergroup",
    "channel",
    "public_channel",
    "private_channel",
}
_USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{3,31}$")

TELEGRAM_NOTE_PRIVATE = "Requires access to the Telegram chat"


def is_valid_username(value: str) -> bool:
    """A plausible public Telegram username (never derived from the title)."""
    return bool(value) and bool(_USERNAME_RE.match(value))


def internal_chat_id(chat_id, chat_type: str) -> str | None:
    """Return the numeric id used in ``t.me/c/<id>`` links, or None.

    Only supergroup/channel exports are accepted. A ``-100`` supergroup
    prefix is stripped; a bare negative id (basic group / user) is rejected
    because it must not be guessed.
    """
    if not chat_id:
        return None
    if str(chat_type or "").strip().lower() not in _SUPERGROUP_TYPES:
        return None
    value = str(chat_id).strip()
    if value.startswith("-100"):
        value = value[4:]
    elif value.startswith("-"):
        return None
    if not value.isdigit() or len(value) < 5:
        return None
    return value


def is_http_url(value: str) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith(
        ("https://", "http://")
    )


class TelegramSourceIndex:
    """Cached, read-only view over the configured Telegram export.

    Loading is lazy: the export is only read when a Telegram citation is
    actually resolved. Nothing from the export is logged.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.chat_id = None
        self.chat_type = ""
        self.chat_name = ""
        self.username = ""
        self.loaded = False
        self.error = ""
        self._messages: dict[int, str] = {}

    def load(self) -> None:
        if self.loaded:
            return
        self.loaded = True
        if not self.path.exists():
            self.error = f"Telegram export not found: {self.path}"
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.error = f"Cannot read Telegram export: {exc}"
            return
        if isinstance(raw, dict):
            self.chat_id = raw.get("id")
            self.chat_type = str(raw.get("type") or "")
            self.chat_name = str(raw.get("name") or "")
            self.username = str(raw.get("username") or "")
            messages = raw.get("messages") or []
        elif isinstance(raw, list):
            messages = raw
        else:
            self.error = "Unsupported Telegram export root type"
            return
        parsed, _ = parse_messages(messages)
        for message in parsed:
            self._messages[message.message_id] = format_message(message)

    def formatted_message(self, message_id: int) -> str | None:
        self.load()
        try:
            return self._messages.get(int(message_id))
        except (TypeError, ValueError):
            return None

    def find_message_for_quote(
        self, message_ids: list[int], quote: str
    ) -> int | None:
        """Return the message id whose formatted text contains *quote*."""
        normalized_quote = normalize_whitespace(quote)
        if not normalized_quote:
            return None
        for message_id in message_ids or []:
            text = self.formatted_message(message_id)
            if text and normalized_quote in normalize_whitespace(text):
                try:
                    return int(message_id)
                except (TypeError, ValueError):
                    continue
        return None


class TelegramSourceResolver:
    """Build Telegram permalinks from validated citations."""

    def __init__(
        self,
        index: TelegramSourceIndex,
        *,
        link_base: str = "",
    ) -> None:
        self.index = index
        self.link_base = (link_base or "").strip().rstrip("/")

    def build_url(
        self, message_id: int, topic_id: int | None = None
    ) -> tuple[str | None, str]:
        """Build a permalink. Forum topics use ``/<topic>/<message>``."""
        suffix = str(int(message_id))
        if topic_id is not None:
            try:
                topic = int(topic_id)
            except (TypeError, ValueError):
                topic = 0
            if topic > 0:
                suffix = f"{topic}/{suffix}"
        # A — explicit configured base (highest priority).
        if is_http_url(self.link_base):
            return f"{self.link_base}/{suffix}", ""
        # B — real public username from the export (never from the title).
        self.index.load()
        if is_valid_username(self.index.username):
            return f"https://t.me/{self.index.username}/{suffix}", ""
        # C — private / supergroup form.
        internal = internal_chat_id(self.index.chat_id, self.index.chat_type)
        if internal:
            return f"https://t.me/c/{internal}/{suffix}", TELEGRAM_NOTE_PRIVATE
        return None, ""


__all__ = [
    "TelegramSourceIndex",
    "TelegramSourceResolver",
    "TELEGRAM_NOTE_PRIVATE",
    "internal_chat_id",
    "is_http_url",
    "is_valid_username",
]
