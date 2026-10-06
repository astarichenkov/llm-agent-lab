"""Telegram chat export loader (JSON).

The loader accepts either:

* a single ``result.json`` file, or
* a **directory** containing one or many ``*.json`` exports (recursively).

Multi-file support matters for Telegram forum/supergroup exports: each file is
an export of a different topic/chat. A general ("общий") export may already
contain the messages of the specialized topic exports, so loading all files
naively would index the same message twice. The loader therefore **deduplicates
messages by ``(chat_id, message_id)``**: the file with fewer messages (a more
specific topic) wins the attribution, while the general export only keeps the
messages no topic claimed. Duplicates are counted as ``duplicate_message``.

The export is PRIVATE: it is only read, never modified.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from app.services.rag.loaders.base import LoaderError
from app.services.rag.models import Document, LoaderStats, Segment
from app.services.rag.normalization.telegram import (
    TelegramMessage,
    format_message,
    group_messages,
    parse_messages,
)

logger = logging.getLogger("app.services.rag.loaders.telegram")

DUPLICATE_REASON = "duplicate_message"


class TelegramLoader:
    """Load one or many Telegram JSON exports as conversation documents."""

    source_type = "telegram"

    def __init__(
        self,
        path: str | Path,
        *,
        gap_minutes: int = 30,
        max_group_messages: int = 12,
        max_group_chars: int = 1800,
    ) -> None:
        self.path = Path(path)
        self.gap_minutes = gap_minutes
        self.max_group_messages = max_group_messages
        self.max_group_chars = max_group_chars
        self._stats = LoaderStats(source_type=self.source_type)

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    def _files(self) -> list[Path]:
        if not self.path.exists():
            return []
        if self.path.is_file():
            return [self.path] if self.path.suffix.lower() == ".json" else []
        return sorted(p for p in self.path.rglob("*.json") if p.is_file())

    def _relative(self, file: Path) -> str:
        root = self.path if self.path.is_dir() else self.path.parent
        if root.is_dir():
            try:
                return file.relative_to(root).as_posix()
            except ValueError:
                pass
        return file.name

    # ------------------------------------------------------------------
    # reading / normalization
    # ------------------------------------------------------------------
    def _read_file(self, file: Path) -> tuple[str, str, list[dict[str, Any]]]:
        """Return ``(chat_id, chat_name, raw_messages)`` for one export."""
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LoaderError(f"Cannot read Telegram export {file}: {exc}") from exc
        if isinstance(raw, dict):
            chat_id = str(raw.get("id") or file.stem)
            chat_name = str(raw.get("name") or file.stem)
            messages = raw.get("messages") or []
        elif isinstance(raw, list):
            chat_id = file.stem
            chat_name = file.stem
            messages = raw
        else:  # pragma: no cover - unexpected root type
            raise LoaderError(f"Unsupported Telegram export root type: {file}")
        if not isinstance(messages, list):
            raise LoaderError(f"Telegram export 'messages' is not a list: {file}")
        return chat_id, chat_name, messages

    def _prepare(
        self,
    ) -> tuple[dict[tuple[str, str], list[TelegramMessage]], LoaderStats]:
        """Read every export, deduplicate, and return messages grouped by source."""
        stats = LoaderStats(source_type=self.source_type)
        entries: list[tuple[Path, str, str, list[TelegramMessage]]] = []

        for file in self._files():
            stats.files += 1
            try:
                chat_id, chat_name, raw_messages = self._read_file(file)
            except LoaderError as exc:
                logger.warning("%s", exc.message)
                continue
            messages, parse_stats = parse_messages(raw_messages)
            stats.messages += parse_stats.total
            stats.usable_messages += parse_stats.usable
            stats.skipped_messages += parse_stats.skipped
            for reason, count in parse_stats.reasons.items():
                stats.skip_reasons[reason] = stats.skip_reasons.get(reason, 0) + count
            entries.append((file, chat_id, chat_name, messages))

        # Deduplicate by (chat_id, message_id). Files are processed from the
        # largest to the smallest so a smaller, more specific topic export
        # overrides the attribution of an already-seen message.
        chosen: dict[tuple[str, int], tuple[TelegramMessage, str, str]] = {}
        for file, chat_id, chat_name, messages in sorted(
            entries, key=lambda item: len(item[3]), reverse=True
        ):
            source = self._relative(file)
            for message in messages:
                chosen[(chat_id, message.message_id)] = (message, source, chat_name)

        duplicates = stats.usable_messages - len(chosen)
        if duplicates > 0:
            stats.usable_messages -= duplicates
            stats.skipped_messages += duplicates
            stats.skip_reasons[DUPLICATE_REASON] = (
                stats.skip_reasons.get(DUPLICATE_REASON, 0) + duplicates
            )

        grouped: dict[tuple[str, str], list[TelegramMessage]] = {}
        for message, source, chat_name in chosen.values():
            grouped.setdefault((source, chat_name), []).append(message)
        for messages in grouped.values():
            messages.sort(key=lambda m: (m.date, m.message_id))
        return grouped, stats

    # ------------------------------------------------------------------
    # scan / load
    # ------------------------------------------------------------------
    def scan(self) -> LoaderStats:
        if not self.path.exists():
            return LoaderStats(source_type=self.source_type, files=0)
        _, stats = self._prepare()
        return stats

    def load(self) -> Iterable[Document]:
        grouped, stats = self._prepare()
        for (source, chat_name), messages in grouped.items():
            groups = group_messages(
                messages,
                gap_minutes=self.gap_minutes,
                max_group_messages=self.max_group_messages,
                max_group_chars=self.max_group_chars,
            )
            for index, group in enumerate(groups):
                yield self._build_document(chat_name, source, group, index)
        self._stats = stats

    def stats(self) -> LoaderStats:
        return self._stats

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_document(
        self,
        chat_name: str,
        source: str,
        group: list[TelegramMessage],
        index: int,
    ) -> Document:
        segments = [
            Segment(
                text=format_message(message),
                metadata=self._message_metadata(message),
            )
            for message in group
        ]
        full_text = "\n\n".join(segment.text for segment in segments)
        authors: list[str] = []
        for message in group:
            if message.author not in authors:
                authors.append(message.author)
        message_ids = [message.message_id for message in group]
        dates = [m.date for m in group if m.date]
        metadata = {
            "source_type": self.source_type,
            "source": source,
            "chat_name": chat_name,
            "title": chat_name,
            "conversation_index": index,
            "message_ids": message_ids,
            "date_from": min(dates) if dates else "",
            "date_to": max(dates) if dates else "",
            "authors": authors,
            "locator": f"{source}#{message_ids[0] if message_ids else index}",
        }
        return Document(text=full_text, metadata=metadata, segments=segments)

    @staticmethod
    def _message_metadata(message: TelegramMessage) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "message_id": message.message_id,
            "author": message.author,
            "date": message.date,
        }
        if message.reply_to_message_id is not None:
            metadata["reply_to_message_id"] = message.reply_to_message_id
        if message.forwarded_from:
            metadata["forwarded_from"] = message.forwarded_from
        return metadata
