"""Day 15 — persistence for the controlled lifecycle state.

The state (including the full transition history) is stored as a small atomic
JSON file, exactly like Days 11-14. This is what makes the ``pause``/``resume``
demonstration work across separate HTTP requests and even a restart: the
lifecycle — state, step, guards and history — is reloaded from disk.

There is exactly ONE current task (single-user educational app), so creating a
new task replaces the previous one.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app.schemas.day15 import Day15TaskState

logger = logging.getLogger("app.services.day15.store")


class LifecycleStore:
    """Minimal, atomic JSON-file persistence for the Day 15 lifecycle state."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Day15TaskState | None:
        """Return the stored task, or ``None`` when absent/corrupt."""
        try:
            if not self._path.exists():
                return None
            with open(self._path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return None
            payload = data.get("task", data)
            return Day15TaskState.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - state must never crash the app
            logger.warning(
                "day15 lifecycle load failed (%s): %s", type(exc).__name__, exc
            )
            return None

    def save(self, state: Day15TaskState) -> None:
        """Atomically persist the current lifecycle state."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"task": state.model_dump()}
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "day15 lifecycle save failed (%s): %s", type(exc).__name__, exc
            )

    def clear(self) -> None:
        """Delete the persisted state (used by tests / reset)."""
        try:
            if self._path.exists():
                self._path.unlink()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "day15 lifecycle clear failed (%s): %s", type(exc).__name__, exc
            )
