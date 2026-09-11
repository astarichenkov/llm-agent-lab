"""Context storage for agents.

The :class:`ContextRepository` abstraction keeps SQL out of :class:`Agent`.
:class:`SQLiteContextRepository` is the default persistent implementation.

SQLite is used instead of a JSON file because it naturally supports several
agents with independent histories (`agent_id` column) and safe incremental
writes. The database is a single file (``Settings.agent_db_path``, default
``data/agents.db``) that survives process/container restarts when the host
directory is persisted (see ``docker-compose.yml``).
"""
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from pathlib import Path

from app.agents.config import AgentConfig
from app.agents.models import Message, utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_agent_id ON messages(agent_id, id);

CREATE TABLE IF NOT EXISTS agents (
    agent_id    TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


class ContextRepository(ABC):
    """Storage contract for agent history and (optionally) agent configs."""

    @abstractmethod
    async def load_messages(self, agent_id: str) -> list[Message]:
        """Return every stored message for ``agent_id`` in insertion order."""

    @abstractmethod
    async def add_message(self, agent_id: str, message: Message) -> None:
        """Append one message to an agent's history."""

    @abstractmethod
    async def clear_messages(self, agent_id: str) -> None:
        """Delete an agent's history (the agent config is kept)."""

    @abstractmethod
    async def save_config(self, agent_id: str, config: AgentConfig) -> None:
        """Persist (upsert) an agent configuration."""

    @abstractmethod
    async def load_config(self, agent_id: str) -> AgentConfig | None:
        """Return the stored configuration, or ``None`` if unknown."""

    @abstractmethod
    async def list_agent_ids(self) -> list[str]:
        """All agent ids known to storage (configs and/or messages)."""


class InMemoryContextRepository(ContextRepository):
    """Transient, per-instance context storage.

    Used by the Day 6 stateless agent: a fresh instance per request means no
    history ever survives between calls, and nothing is written to SQLite.
    Implemented with plain dicts — one repository belongs to exactly one
    ``Agent``.
    """

    def __init__(self) -> None:
        self._messages: dict[str, list[Message]] = {}
        self._configs: dict[str, AgentConfig] = {}

    async def load_messages(self, agent_id: str) -> list[Message]:
        return list(self._messages.get(agent_id, []))

    async def add_message(self, agent_id: str, message: Message) -> None:
        self._messages.setdefault(agent_id, []).append(message)

    async def clear_messages(self, agent_id: str) -> None:
        self._messages.pop(agent_id, None)

    async def save_config(self, agent_id: str, config: AgentConfig) -> None:
        self._configs[agent_id] = config

    async def load_config(self, agent_id: str) -> AgentConfig | None:
        return self._configs.get(agent_id)

    async def list_agent_ids(self) -> list[str]:
        return sorted(set(self._messages) | set(self._configs))


class SQLiteContextRepository(ContextRepository):
    """SQLite-backed :class:`ContextRepository`.

    A fresh connection is opened per operation (cheap for SQLite); this keeps
    the class safe to share across requests/tasks without a global lock. The
    schema is created lazily on first use, so constructing the repository (and
    therefore the FastAPI app) never touches the filesystem.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._schema_ready = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if not self._schema_ready:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self._db_path, timeout=10)) as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
            self._schema_ready = True
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------
    async def load_messages(self, agent_id: str) -> list[Message]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE agent_id = ? ORDER BY id ASC",
                (agent_id,),
            ).fetchall()
        return [
            Message(role=row["role"], content=row["content"], created_at=row["created_at"])
            for row in rows
        ]

    async def add_message(self, agent_id: str, message: Message) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO messages (agent_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (agent_id, message.role, message.content, message.created_at),
            )
            conn.commit()

    async def clear_messages(self, agent_id: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM messages WHERE agent_id = ?", (agent_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # configs
    # ------------------------------------------------------------------
    async def save_config(self, agent_id: str, config: AgentConfig) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO agents (agent_id, config_json, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(agent_id) DO UPDATE SET "
                "config_json = excluded.config_json, updated_at = excluded.updated_at",
                (agent_id, config.model_dump_json(), utc_now_iso()),
            )
            conn.commit()

    async def load_config(self, agent_id: str) -> AgentConfig | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT config_json FROM agents WHERE agent_id = ?", (agent_id,)
            ).fetchone()
        if row is None:
            return None
        return AgentConfig.model_validate(json.loads(row["config_json"]))

    async def list_agent_ids(self) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT agent_id FROM agents "
                "UNION SELECT agent_id FROM messages ORDER BY agent_id"
            ).fetchall()
        return [row["agent_id"] for row in rows]
