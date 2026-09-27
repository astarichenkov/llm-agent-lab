"""Day 18 — scheduler adapter around APScheduler.

The scheduler lives in the **FastAPI application process** (never in a
short-lived MCP subprocess). This wrapper isolates the APScheduler API behind
a tiny interface so the service and tests do not depend on it directly:

* one job per monitoring job, keyed by the ``mon_id``;
* ``max_instances=1`` + ``coalesce=True`` — an overlapping execution is
  skipped/misfired instead of running twice (critical for 10-second demo
  intervals);
* ``misfire_grace_time`` tolerates a briefly blocked event loop;
* register/unregister are idempotent.

Lifecycle (start/stop) is owned by the FastAPI ``lifespan`` handler, not by
this class.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger("app.services.day18.scheduler")

# How long after the scheduled time a run may still fire (event-loop hiccups).
MISFIRE_GRACE_TIME_SECONDS = 5


class MonitoringScheduler:
    """Small, testable facade over ``AsyncIOScheduler``."""

    def __init__(self, scheduler: AsyncIOScheduler | None = None) -> None:
        self._scheduler = scheduler or AsyncIOScheduler(timezone=timezone.utc)

    @property
    def raw(self) -> AsyncIOScheduler:
        """The underlying scheduler (used by tests to inspect job settings)."""
        return self._scheduler

    @property
    def running(self) -> bool:
        return bool(getattr(self._scheduler, "running", False))

    def start(self) -> None:
        """Start the scheduler (idempotent). Requires a running event loop."""
        if not self.running:
            self._scheduler.start()
            logger.info("Day 18 monitoring scheduler started")

    def shutdown(self, *, wait: bool = False) -> None:
        """Shut the scheduler down gracefully (idempotent)."""
        if self.running:
            self._scheduler.shutdown(wait=wait)
            logger.info("Day 18 monitoring scheduler stopped")

    def register(
        self,
        job_id: str,
        interval_seconds: int,
        callback: Callable[[], Awaitable[None]],
        next_run_at: datetime | None = None,
    ) -> None:
        """Register (or replace) one interval job.

        ``next_run_at`` allows the first execution to be scheduled immediately
        (``now``) or at a restored timestamp after a restart.
        """
        # With APScheduler 3.x, ``replace_existing`` does not purge a
        # *pending* job of the same id while the scheduler is stopped (it only
        # applies once the scheduler runs). Explicitly removing first keeps
        # register idempotent both before and after start.
        self.unregister(job_id)
        self._scheduler.add_job(
            callback,
            trigger="interval",
            seconds=int(interval_seconds),
            id=job_id,
            name=f"monitoring:{job_id}",
            replace_existing=True,
            next_run_time=next_run_at,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=MISFIRE_GRACE_TIME_SECONDS,
        )

    def unregister(self, job_id: str) -> None:
        """Remove a job if present (idempotent)."""
        try:
            self._scheduler.remove_job(job_id)
        except JobLookupError:
            return

    def is_registered(self, job_id: str) -> bool:
        return self._scheduler.get_job(job_id) is not None

    def next_run_time(self, job_id: str) -> datetime | None:
        job = self._scheduler.get_job(job_id)
        return None if job is None else job.next_run_time
