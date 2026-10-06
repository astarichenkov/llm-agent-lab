"""Day 25 — SQLite persistence for chat sessions, messages, evidence and state.

The Day 25 chat database lives SEPARATELY from the Day 21 vector index
(default ``data/day25/chat.sqlite3``). Task state is normal application state
stored in a JSON column; it is NEVER embedded and never mixed with manuals or
Telegram chunks.

Schema init is deterministic (``CREATE TABLE IF NOT EXISTS``) and runs lazily
on first use, mirroring the existing Day 18/Agent repositories: constructing
the application never touches disk.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.day25 import (
    MSG_STATUS_OK,
    ROLE_USER,
    ChatMessage,
    ChatSession,
    FactItem,
    MessageEvidence,
    TaskState,
    VehicleState,
)

logger = logging.getLogger("app.services.day25.repository")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ok',
    created_at  TEXT NOT NULL,
    trace_json  TEXT,
    ordinal     INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, ordinal);

CREATE TABLE IF NOT EXISTS task_states (
    session_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS message_evidence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id      TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,
    source_type     TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT '',
    section         TEXT NOT NULL DEFAULT '',
    page            INTEGER,
    page_from       INTEGER,
    page_to         INTEGER,
    message_ids_json TEXT NOT NULL DEFAULT '[]',
    date_from       TEXT NOT NULL DEFAULT '',
    date_to         TEXT NOT NULL DEFAULT '',
    chat_name       TEXT NOT NULL DEFAULT '',
    quote           TEXT NOT NULL DEFAULT '',
    quote_valid     INTEGER NOT NULL DEFAULT 0,
    chunk_text      TEXT NOT NULL DEFAULT '',
    ordinal         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_message_evidence_message
    ON message_evidence(message_id, ordinal);

CREATE TABLE IF NOT EXISTS day25_schema (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

SCHEMA_VERSION = "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def _row_to_session(row: sqlite3.Row) -> ChatSession:
    return ChatSession(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class ChatRepository:
    """Thin synchronous SQLite gateway for the Day 25 tables."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._schema_ready = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # connection / schema
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if not self._schema_ready:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self._db_path, timeout=10)) as conn:
                conn.executescript(_SCHEMA)
                conn.execute(
                    "INSERT OR IGNORE INTO day25_schema (key, value) VALUES (?, ?)",
                    ("version", SCHEMA_VERSION),
                )
                conn.commit()
            self._schema_ready = True
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------
    def create_session(self, title: str | None = None) -> ChatSession:
        now = _now_iso()
        session = ChatSession(
            id=new_id(),
            title=(title or "").strip() or "Новый чат",
            created_at=now,
            updated_at=now,
        )
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (session.id, session.title, session.created_at, session.updated_at),
            )
            conn.execute(
                "INSERT INTO task_states (session_id, state_json, updated_at) "
                "VALUES (?, ?, ?)",
                (
                    session.id,
                    TaskState().model_dump_json(),
                    now,
                ),
            )
            conn.commit()
        return session

    def get_session(self, session_id: str) -> ChatSession | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return _row_to_session(row) if row else None

    def list_sessions(self) -> list[ChatSession]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [_row_to_session(row) for row in rows]

    def set_title(self, session_id: str, title: str) -> ChatSession | None:
        now = _now_iso()
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )
            conn.commit()
        return self.get_session(session_id)

    def touch_session(self, session_id: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (_now_iso(), session_id),
            )
            conn.commit()

    def delete_session(self, session_id: str) -> bool:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "DELETE FROM chat_sessions WHERE id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM task_states WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM message_evidence WHERE session_id = ?", (session_id,)
            )
            conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        status: str = MSG_STATUS_OK,
        trace: dict | None = None,
    ) -> ChatMessage:
        now = _now_iso()
        message = ChatMessage(
            id=new_id(),
            session_id=session_id,
            role=role,
            content=content,
            status=status,
            created_at=now,
            trace=trace,
        )
        with closing(self._connect()) as conn:
            ordinal = conn.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM chat_messages "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO chat_messages "
                "(id, session_id, role, content, status, created_at, trace_json, ordinal) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    session_id,
                    role,
                    content,
                    status,
                    now,
                    json.dumps(trace, ensure_ascii=False) if trace else None,
                    ordinal,
                ),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
        return message

    def _row_to_message(self, row: sqlite3.Row) -> ChatMessage:
        trace = None
        if row["trace_json"]:
            try:
                trace = json.loads(row["trace_json"])
            except (ValueError, TypeError):
                trace = None
        return ChatMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            status=row["status"],
            created_at=row["created_at"],
            trace=trace,
        )

    def list_messages(
        self, session_id: str, *, limit: int | None = None
    ) -> list[ChatMessage]:
        query = "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY ordinal ASC"
        params: list = [session_id]
        if limit is not None:
            query = (
                "SELECT * FROM (SELECT * FROM chat_messages WHERE session_id = ? "
                "ORDER BY ordinal DESC LIMIT ?) ORDER BY ordinal ASC"
            )
            params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_message(row) for row in rows]

    def message_count(self, session_id: str) -> int:
        with closing(self._connect()) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            )

    # ------------------------------------------------------------------
    # evidence
    # ------------------------------------------------------------------
    def add_evidence(
        self, session_id: str, message_id: str, evidence: list[MessageEvidence]
    ) -> None:
        if not evidence:
            return
        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT INTO message_evidence ("
                "message_id, session_id, chunk_id, source_type, source, section, "
                "page, page_from, page_to, message_ids_json, date_from, date_to, "
                "chat_name, quote, quote_valid, chunk_text, ordinal) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        message_id,
                        session_id,
                        item.chunk_id,
                        item.source_type,
                        item.source,
                        item.section,
                        item.page,
                        item.page_from,
                        item.page_to,
                        json.dumps(item.message_ids, ensure_ascii=False),
                        item.date_from,
                        item.date_to,
                        item.chat_name,
                        item.quote,
                        1 if item.quote_valid else 0,
                        item.chunk_text,
                        item.ordinal,
                    )
                    for item in evidence
                ],
            )
            conn.commit()

    def _row_to_evidence(self, row: sqlite3.Row) -> MessageEvidence:
        try:
            message_ids = json.loads(row["message_ids_json"] or "[]")
        except (ValueError, TypeError):
            message_ids = []
        return MessageEvidence(
            id=row["id"],
            message_id=row["message_id"],
            chunk_id=row["chunk_id"],
            source_type=row["source_type"] or "",
            source=row["source"] or "",
            section=row["section"] or "",
            page=row["page"],
            page_from=row["page_from"],
            page_to=row["page_to"],
            message_ids=message_ids,
            date_from=row["date_from"] or "",
            date_to=row["date_to"] or "",
            chat_name=row["chat_name"] or "",
            quote=row["quote"] or "",
            quote_valid=bool(row["quote_valid"]),
            chunk_text=row["chunk_text"] or "",
            ordinal=int(row["ordinal"] or 0),
        )

    def list_evidence(self, message_id: str) -> list[MessageEvidence]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM message_evidence WHERE message_id = ? "
                "ORDER BY ordinal ASC, id ASC",
                (message_id,),
            ).fetchall()
        return [self._row_to_evidence(row) for row in rows]

    def list_evidence_for_session(
        self, session_id: str
    ) -> dict[str, list[MessageEvidence]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM message_evidence WHERE session_id = ? "
                "ORDER BY message_id, ordinal ASC, id ASC",
                (session_id,),
            ).fetchall()
        grouped: dict[str, list[MessageEvidence]] = {}
        for row in rows:
            grouped.setdefault(row["message_id"], []).append(
                self._row_to_evidence(row)
            )
        return grouped

    # ------------------------------------------------------------------
    # task state
    # ------------------------------------------------------------------
    def get_state(self, session_id: str) -> TaskState:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT state_json FROM task_states WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return TaskState()
        try:
            return TaskState.model_validate(json.loads(row["state_json"]))
        except Exception:  # noqa: BLE001 - corrupt state must not wipe the chat
            logger.warning("Corrupt task state for session %s; using default", session_id)
            return TaskState()

    def get_state_updated_at(self, session_id: str) -> str:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT updated_at FROM task_states WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["updated_at"] if row else ""

    def save_state(self, session_id: str, state: TaskState) -> None:
        now = _now_iso()
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO task_states (session_id, state_json, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "state_json = excluded.state_json, updated_at = excluded.updated_at",
                (session_id, state.model_dump_json(), now),
            )
            conn.commit()
