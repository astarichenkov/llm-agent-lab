"""Day 11 — memory storage.

Three layers, three lifecycles:

* **short-term** — session scoped, in memory: the messages of the current
  dialog. A new session never sees the old session's short-term memory.
* **working**    — session/task scoped, in memory: structured state of the
  current task. Can be cleared independently.
* **long-term**  — process/session independent: durable key/value facts,
  persisted as a small JSON file so they survive a new session AND a restart.

No database, no Redis, no vector store: a single JSON file is enough for the
educational scope and matches the project's "minimal local storage" approach.
The write is atomic (tmp file + replace) so a crash cannot corrupt memory.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.schemas.day11 import LongTermMemory, WorkingMemory

logger = logging.getLogger("app.services.day11.store")


# ----------------------------------------------------------------------
# merge helpers (pure)
# ----------------------------------------------------------------------
def _merge_items(current: list[str], update: list[str]) -> list[str]:
    """Append new items, case-insensitively de-duplicated, preserving order."""
    merged = list(current)
    seen = {item.casefold() for item in merged}
    for item in update:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return merged


def apply_working_update(
    current: WorkingMemory, update: WorkingMemory
) -> WorkingMemory:
    """Merge a classifier delta into working memory.

    The goal is REPLACED when the classifier provides a new one; list
    categories are merged; ``data`` keys are upserted.
    """
    merged = current.model_copy(deep=True)
    if update.goal and update.goal.strip():
        merged.goal = update.goal.strip()
    merged.constraints = _merge_items(merged.constraints, update.constraints)
    merged.requirements = _merge_items(merged.requirements, update.requirements)
    merged.decisions = _merge_items(merged.decisions, update.decisions)
    for key, value in update.data.items():
        name = str(key).strip()
        if name:
            merged.data[name] = str(value).strip()
    return merged


def apply_long_term_update(
    current: LongTermMemory, updates: dict[str, str]
) -> LongTermMemory:
    """Upsert classifier entries into long-term memory (new value wins)."""
    merged = current.model_copy(deep=True)
    for key, value in updates.items():
        name = str(key).strip()
        if not name:
            continue
        merged.entries[name] = str(value).strip()
    return merged


# ----------------------------------------------------------------------
# long-term persistence
# ----------------------------------------------------------------------
class LongTermMemoryStore:
    """Minimal JSON-file persistence for LONG-TERM memory only."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> LongTermMemory:
        """Read the stored memory; a missing/corrupt file yields empty memory."""
        try:
            if not self._path.exists():
                return LongTermMemory()
            with open(self._path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            entries = data.get("entries", data) if isinstance(data, dict) else {}
            if not isinstance(entries, dict):
                entries = {}
            clean = {
                str(key): str(value)
                for key, value in entries.items()
                if value is not None and str(value).strip()
            }
            return LongTermMemory(entries=clean)
        except Exception as exc:  # noqa: BLE001 - memory must never crash the app
            logger.warning(
                "day11 long-term memory load failed (%s): %s", type(exc).__name__, exc
            )
            return LongTermMemory()

    def save(self, memory: LongTermMemory) -> None:
        """Atomically persist the long-term memory."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"entries": memory.entries}
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "day11 long-term memory save failed (%s): %s", type(exc).__name__, exc
            )

    def clear(self) -> None:
        """Delete the persisted long-term memory (file removed)."""
        try:
            if self._path.exists():
                self._path.unlink()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "day11 long-term memory clear failed (%s): %s", type(exc).__name__, exc
            )


# ----------------------------------------------------------------------
# session scoped memory
# ----------------------------------------------------------------------
@dataclass
class MemorySession:
    """Short-term + working memory of ONE session/task."""

    id: str
    short_term: list[dict[str, str]] = field(default_factory=list)
    working: WorkingMemory = field(default_factory=WorkingMemory)
