"""Day 14 — persistence for the active invariants.

Invariants are stored in their OWN small JSON file, completely separate from
the conversation history (which lives in memory in the service). This is the
concrete, testable proof of the assignment's first requirement: invariants are
not chat messages.

Matching the project's minimal local-storage approach, no database is used.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app.schemas.day14 import Invariant
from app.services.day14.invariants import DEFAULT_INVARIANTS

logger = logging.getLogger("app.services.day14.store")


class InvariantStore:
    """Atomic JSON-file persistence for the active invariants."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[Invariant]:
        """Return stored invariants; missing/corrupt file -> defaults."""
        try:
            if not self._path.exists():
                return [inv.model_copy(deep=True) for inv in DEFAULT_INVARIANTS]
            with open(self._path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return [inv.model_copy(deep=True) for inv in DEFAULT_INVARIANTS]
            payload = data.get("invariants", [])
            return [Invariant.model_validate(item) for item in payload]
        except Exception as exc:  # noqa: BLE001 - state must never crash the app
            logger.warning(
                "day14 invariants load failed (%s): %s", type(exc).__name__, exc
            )
            return [inv.model_copy(deep=True) for inv in DEFAULT_INVARIANTS]

    def save(self, invariants: list[Invariant]) -> None:
        """Atomically persist the invariants."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"invariants": [inv.model_dump() for inv in invariants]}
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "day14 invariants save failed (%s): %s", type(exc).__name__, exc
            )
