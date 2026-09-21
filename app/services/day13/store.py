"""Day 13 — persistence for the current Task State.

The Task State must survive separate HTTP requests (and, ideally, a restart).
Like Day 11's long-term memory and Day 12's user profile, it is stored as a
small atomic JSON file: no DB/Redis is needed for the educational scope.

Limitations made explicit:

* there is exactly ONE current task (single-user educational app), so a new
  task replaces the previous one;
* the file lives on the local filesystem (``data/day13_task_state.json`` by
  default) and is ignored by git through the existing ``data/`` rules.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app.schemas.day13 import TaskState

logger = logging.getLogger("app.services.day13.store")


class TaskStore:
    """Minimal, atomic JSON-file persistence for the current Task State."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> TaskState | None:
        """Return the stored task, or ``None`` when absent/corrupt."""
        try:
            if not self._path.exists():
                return None
            with open(self._path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return None
            # Accept both {"task": {...}} and a bare TaskState object.
            payload = data.get("task", data)
            return TaskState.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - state must never crash the app
            logger.warning(
                "day13 task state load failed (%s): %s", type(exc).__name__, exc
            )
            return None

    def save(self, state: TaskState) -> None:
        """Atomically persist the current Task State."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"task": state.model_dump()}
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "day13 task state save failed (%s): %s", type(exc).__name__, exc
            )

    def clear(self) -> None:
        """Delete the persisted task (used by tests / reset)."""
        try:
            if self._path.exists():
                self._path.unlink()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "day13 task state clear failed (%s): %s", type(exc).__name__, exc
            )
