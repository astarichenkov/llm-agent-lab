"""Day 18 — SQLite persistence for monitoring jobs and run aggregates.

Two tables live in one database file (default ``data/day18/monitoring.db``):

* ``monitor_jobs`` — job parameters + lifecycle timestamps;
* ``monitor_runs`` — one row per scheduled execution with COUNTS and status.

Design notes:

* A fresh connection is opened per operation (cheap for SQLite) and closed
  immediately, so no unsafe connection is shared between threads/tasks. This
  mirrors the existing ``SQLiteContextRepository`` pattern.
* All timestamps are stored as timezone-aware UTC ISO-8601 strings.
* A raw VictoriaLogs row is never written here: only ``logs_count`` /
  ``malformed_count`` aggregates. Secrets (URLs, tokens, headers) are never
  stored either; only the user-supplied ``text_contains`` filter is kept
  (documented in ``docs/week4/day18.md``).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.day18 import (
    JOB_STATUS_ACTIVE,
    RUN_STATUS_SUCCESS,
    MonitoringJob,
    MonitoringRun,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS monitor_jobs (
    id               TEXT    PRIMARY KEY,
    service          TEXT    NOT NULL,
    level            TEXT,
    text_contains    TEXT,
    interval_seconds INTEGER NOT NULL,
    lookback_minutes INTEGER NOT NULL,
    limit_count      INTEGER NOT NULL,
    status           TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL,
    last_run_at      TEXT,
    next_run_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_monitor_jobs_status ON monitor_jobs(status);

CREATE TABLE IF NOT EXISTS monitor_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT    NOT NULL,
    logs_count      INTEGER NOT NULL DEFAULT 0,
    malformed_count INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL,
    error_message   TEXT,
    duration_ms     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_monitor_runs_job ON monitor_runs(job_id, id);
"""


def _to_iso(value: datetime | None) -> str | None:
    """Serialize a timezone-aware datetime as a UTC ISO-8601 string."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: str | None) -> datetime | None:
    """Parse a stored ISO-8601 string back into an aware UTC datetime."""
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class MonitoringRepository:
    """Thin, synchronous SQLite gateway for the Day 18 tables."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._schema_ready = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # Lazily create the schema so constructing the app never touches disk.
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
    # Jobs
    # ------------------------------------------------------------------
    def create_job(self, job: MonitoringJob) -> MonitoringJob:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO monitor_jobs ("
                "id, service, level, text_contains, interval_seconds, "
                "lookback_minutes, limit_count, status, created_at, "
                "updated_at, last_run_at, next_run_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    job.service,
                    job.level,
                    job.text_contains,
                    job.interval_seconds,
                    job.lookback_minutes,
                    job.limit,
                    job.status,
                    _to_iso(job.created_at),
                    _to_iso(job.updated_at),
                    _to_iso(job.last_run_at),
                    _to_iso(job.next_run_at),
                ),
            )
            conn.commit()
        return job

    def update_job(self, job: MonitoringJob) -> MonitoringJob:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE monitor_jobs SET "
                "service = ?, level = ?, text_contains = ?, "
                "interval_seconds = ?, lookback_minutes = ?, limit_count = ?, "
                "status = ?, updated_at = ?, last_run_at = ?, next_run_at = ? "
                "WHERE id = ?",
                (
                    job.service,
                    job.level,
                    job.text_contains,
                    job.interval_seconds,
                    job.lookback_minutes,
                    job.limit,
                    job.status,
                    _to_iso(job.updated_at),
                    _to_iso(job.last_run_at),
                    _to_iso(job.next_run_at),
                    job.job_id,
                ),
            )
            conn.commit()
        return job

    def get_job(self, job_id: str) -> MonitoringJob | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM monitor_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return None if row is None else self._row_to_job(row)

    def list_jobs(self, status: str | None = None) -> list[MonitoringJob]:
        with closing(self._connect()) as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM monitor_jobs ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM monitor_jobs WHERE status = ? "
                    "ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def count_runs(self, job_id: str) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM monitor_runs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return int(row["n"]) if row is not None else 0

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def create_run(self, run: MonitoringRun) -> MonitoringRun:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO monitor_runs ("
                "job_id, started_at, finished_at, logs_count, malformed_count, "
                "status, error_message, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.job_id,
                    _to_iso(run.started_at),
                    _to_iso(run.finished_at),
                    int(run.logs_count),
                    int(run.malformed_count),
                    run.status,
                    run.error_message,
                    int(run.duration_ms),
                ),
            )
            conn.commit()
            new_id = cursor.lastrowid
        return run.model_copy(update={"id": int(new_id) if new_id else None})

    def list_runs(self, job_id: str, limit: int = 20) -> list[MonitoringRun]:
        limit = max(1, int(limit))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM monitor_runs WHERE job_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    # ------------------------------------------------------------------
    # Row mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> MonitoringJob:
        return MonitoringJob(
            job_id=row["id"],
            service=row["service"],
            level=row["level"],
            text_contains=row["text_contains"],
            interval_seconds=int(row["interval_seconds"]),
            lookback_minutes=int(row["lookback_minutes"]),
            limit=int(row["limit_count"]),
            status=row["status"] or JOB_STATUS_ACTIVE,
            created_at=_from_iso(row["created_at"]),
            updated_at=_from_iso(row["updated_at"]),
            last_run_at=_from_iso(row["last_run_at"]),
            next_run_at=_from_iso(row["next_run_at"]),
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> MonitoringRun:
        return MonitoringRun(
            id=int(row["id"]),
            job_id=row["job_id"],
            started_at=_from_iso(row["started_at"]),
            finished_at=_from_iso(row["finished_at"]),
            logs_count=int(row["logs_count"]),
            malformed_count=int(row["malformed_count"]),
            status=row["status"] or RUN_STATUS_SUCCESS,
            error_message=row["error_message"],
            duration_ms=int(row["duration_ms"]),
        )
