"""Day 18 — the monitoring service.

This is the single owner of the monitoring lifecycle:

    start_monitoring (MCP/HTTP)
            |
            v
    MonitoringService ──> MonitoringRepository (SQLite)
            |                    ^
            |                    |
            +──> MonitoringScheduler ──> run_job() ──> Day 17 VictoriaLogs client

Key guarantees:

* the FIRST run happens inline, so the dashboard shows a result immediately;
* every run is persisted as an aggregate row (``monitor_runs``);
* a scheduled run NEVER calls an LLM — it is a deterministic
  VictoriaLogs query + SQLite write;
* one failed run is recorded and the scheduler keeps going;
* a job's runs never overlap (``max_instances=1`` plus an in-process guard);
* active jobs are restored from SQLite after a restart.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from app.config import Settings
from app.schemas.day18 import (
    DEFAULT_SUMMARY_RUNS,
    JOB_STATUS_ACTIVE,
    JOB_STATUS_ERROR,
    JOB_STATUS_STOPPED,
    MAX_SUMMARY_RUNS,
    RUN_STATUS_ERROR,
    RUN_STATUS_SUCCESS,
    MonitoringJob,
    MonitoringRun,
    MonitoringStatusResponse,
    MonitoringSummaryResponse,
    StartMonitoringInput,
    StartMonitoringResult,
    StopMonitoringResult,
)
from app.services.day18.repository import MonitoringRepository
from app.services.day18.scheduler import MonitoringScheduler
from app.services.mcp.victorialogs.client import VictoriaLogsClient, VictoriaLogsError
from app.services.mcp.victorialogs.query_builder import QueryBuildError, build_logsql

logger = logging.getLogger("app.services.day18.service")

# Default window used when reading run history for the dashboard.
DEFAULT_RUN_HISTORY = 20

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MonitoringService:
    """Create, schedule, execute and aggregate monitoring jobs."""

    def __init__(
        self,
        settings: Settings,
        *,
        repository: MonitoringRepository | None = None,
        scheduler: MonitoringScheduler | None = None,
        client_factory: Callable[[], Any] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repository or MonitoringRepository(
            settings.day18_monitoring_db_path
        )
        self._scheduler = scheduler or MonitoringScheduler()
        self._client_factory = client_factory or self._default_client
        self._now = clock or _utc_now
        # Guards against overlapping runs even if the scheduler misfires.
        self._running: set[str] = set()

    # ------------------------------------------------------------------
    # Properties (tests / endpoints)
    # ------------------------------------------------------------------
    @property
    def repository(self) -> MonitoringRepository:
        return self._repo

    @property
    def scheduler(self) -> MonitoringScheduler:
        return self._scheduler

    # ------------------------------------------------------------------
    # Scheduler lifecycle
    # ------------------------------------------------------------------
    def start_scheduler(self) -> None:
        self._scheduler.start()

    def shutdown_scheduler(self) -> None:
        self._scheduler.shutdown()

    # ------------------------------------------------------------------
    # Job creation
    # ------------------------------------------------------------------
    async def start_job(self, params: StartMonitoringInput) -> StartMonitoringResult:
        """Persist a job, run it once immediately and register the schedule."""
        now = self._now()
        job = MonitoringJob(
            job_id=self._new_job_id(),
            service=params.service,
            level=params.level,
            text_contains=params.text_contains,
            interval_seconds=int(params.interval_seconds),
            lookback_minutes=int(params.lookback_minutes),
            limit=int(params.limit),
            status=JOB_STATUS_ACTIVE,
            created_at=now,
            updated_at=now,
        )
        self._repo.create_job(job)

        # First run happens inline; a VictoriaLogs/config failure is recorded
        # as a failed run but the job stays active for the next interval.
        first_run = await self.run_job(job.job_id)

        job = self._repo.get_job(job.job_id) or job
        next_run_at = job.next_run_at or (
            self._now() + timedelta(seconds=job.interval_seconds)
        )

        message = "Monitoring started. First run executed immediately."
        if job.status != JOB_STATUS_ACTIVE:
            # The job was stopped/errored while the first run was in flight;
            # do not schedule it.
            message = "Monitoring job is not active; schedule not registered."
        else:
            try:
                self._register(job.job_id, job.interval_seconds, next_run_at)
            except Exception:  # noqa: BLE001 - registration must not crash the API
                logger.exception("Day 18: failed to register job %s", job.job_id)
                job = job.model_copy(
                    update={
                        "status": JOB_STATUS_ERROR,
                        "updated_at": self._now(),
                    }
                )
                self._repo.update_job(job)
                message = "Monitoring job created, but scheduling failed."

        return StartMonitoringResult(
            job_id=job.job_id,
            status=job.status,
            service=job.service,
            level=job.level,
            interval_seconds=job.interval_seconds,
            lookback_minutes=job.lookback_minutes,
            next_run_at=job.next_run_at,
            first_run=first_run,
            message=message,
        )

    def _register(
        self, job_id: str, interval_seconds: int, next_run_at: datetime | None
    ) -> None:
        async def _callback() -> None:
            await self.run_job(job_id)

        self._scheduler.register(
            job_id, interval_seconds, _callback, next_run_at=next_run_at
        )

    def _new_job_id(self) -> str:
        for _ in range(10):
            candidate = "mon_" + uuid.uuid4().hex[:6]
            if self._repo.get_job(candidate) is None:
                return candidate
        return "mon_" + uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    # Scheduled execution
    # ------------------------------------------------------------------
    async def run_job(self, job_id: str) -> MonitoringRun | None:
        """Execute one monitoring run and persist the aggregate.

        Returns the persisted run, or ``None`` when the job is missing,
        stopped or already running (overlap skipped).
        """
        if job_id in self._running:
            logger.info("Day 18: skipping overlapping run for %s", job_id)
            return None
        self._running.add(job_id)
        try:
            job = self._repo.get_job(job_id)
            if job is None or job.status != JOB_STATUS_ACTIVE:
                return None

            started_at = self._now()
            logs_count = 0
            malformed_count = 0
            status = RUN_STATUS_SUCCESS
            error_message: str | None = None

            try:
                rows, malformed_count = await self._collect(job, started_at)
                logs_count = len(rows)
            except QueryBuildError as exc:
                status = RUN_STATUS_ERROR
                error_message = f"invalid query: {exc}"
            except VictoriaLogsError as exc:
                status = RUN_STATUS_ERROR
                error_message = exc.message
            except Exception as exc:  # noqa: BLE001 - one run must never crash
                logger.exception("Day 18: unexpected run failure for %s", job_id)
                status = RUN_STATUS_ERROR
                error_message = f"{exc.__class__.__name__}: {exc}"

            finished_at = self._now()
            duration_ms = max(
                0, int((finished_at - started_at).total_seconds() * 1000)
            )
            run = self._repo.create_run(
                MonitoringRun(
                    job_id=job_id,
                    started_at=started_at,
                    finished_at=finished_at,
                    logs_count=logs_count,
                    malformed_count=malformed_count,
                    status=status,
                    error_message=error_message,
                    duration_ms=duration_ms,
                )
            )

            # Re-read the job: it may have been stopped during the await.
            # Never clobber a stopped/error status with stale active state.
            current = self._repo.get_job(job_id) or job
            if current.status == JOB_STATUS_ACTIVE:
                next_run_at = finished_at + timedelta(seconds=current.interval_seconds)
                current = current.model_copy(
                    update={
                        "last_run_at": finished_at,
                        "next_run_at": next_run_at,
                        "updated_at": finished_at,
                    }
                )
            else:
                current = current.model_copy(
                    update={"last_run_at": finished_at, "updated_at": finished_at}
                )
            self._repo.update_job(current)
            return run
        finally:
            self._running.discard(job_id)

    async def _collect(
        self, job: MonitoringJob, reference: datetime
    ) -> tuple[list[dict[str, Any]], int]:
        """Run the existing Day 17 VictoriaLogs search for one job."""
        query = build_logsql(
            service=job.service,
            service_field=self._settings.victoria_logs_service_field,
            level=job.level,
            text_contains=job.text_contains,
        )
        start = reference - timedelta(minutes=job.lookback_minutes)
        client = self._client_factory()
        return await client.search(
            query=query, start=start, end=reference, limit=job.limit
        )

    def _default_client(self) -> VictoriaLogsClient:
        """Reuse the Day 17 client (no duplicated HTTP code)."""
        return VictoriaLogsClient(
            base_url=self._settings.victoria_logs_base_url,
            service_field=self._settings.victoria_logs_service_field,
            verify_ssl=self._settings.victoria_logs_verify_ssl,
            ca_bundle=self._settings.victoria_logs_ca_bundle,
            http_timeout_seconds=self._settings.victoria_logs_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_job(self, job_id: str) -> MonitoringJob | None:
        return self._repo.get_job(job_id)

    def list_jobs(self, status: str | None = None) -> list[MonitoringJob]:
        return self._repo.list_jobs(status=status)

    def get_status(self, job_id: str) -> MonitoringStatusResponse | None:
        job = self._repo.get_job(job_id)
        if job is None:
            return None
        return MonitoringStatusResponse(
            job_id=job.job_id,
            status=job.status,
            service=job.service,
            level=job.level,
            interval_seconds=job.interval_seconds,
            lookback_minutes=job.lookback_minutes,
            last_run_at=job.last_run_at,
            next_run_at=job.next_run_at,
            runs_count=self._repo.count_runs(job_id),
        )

    def list_runs(self, job_id: str, limit: int = DEFAULT_RUN_HISTORY) -> list[MonitoringRun]:
        return self._repo.list_runs(job_id, limit=limit)

    def get_summary(
        self, job_id: str, last_runs: int = DEFAULT_SUMMARY_RUNS
    ) -> MonitoringSummaryResponse | None:
        """Aggregate the most recent runs (deterministic, backend-side).

        Averages are rounded to two decimals and the ``trend`` compares the
        last run with the window average — no LLM is involved.
        """
        job = self._repo.get_job(job_id)
        if job is None:
            return None

        window = max(1, min(int(last_runs), MAX_SUMMARY_RUNS))
        runs = self._repo.list_runs(job_id, limit=window)  # most recent first
        if not runs:
            return MonitoringSummaryResponse(
                job_id=job.job_id,
                service=job.service,
                last_run_at=job.last_run_at,
            )

        counts = [run.logs_count for run in runs]
        total = sum(counts)
        successful = sum(1 for run in runs if run.status == RUN_STATUS_SUCCESS)
        failed = len(runs) - successful
        average = round(total / len(runs), 2)
        last = counts[0]
        average_for_trend = total / len(runs)
        if len(runs) < 2:
            trend = "stable"
        elif average_for_trend == 0:
            trend = "stable"
        elif last > average_for_trend * 1.2:
            trend = "rising"
        elif last < average_for_trend * 0.8:
            trend = "falling"
        else:
            trend = "stable"

        started = [run.started_at for run in runs]
        finished = [run.finished_at for run in runs]
        return MonitoringSummaryResponse(
            job_id=job.job_id,
            service=job.service,
            runs=len(runs),
            successful_runs=successful,
            failed_runs=failed,
            total_logs=total,
            average_logs_per_run=average,
            min_logs_per_run=min(counts),
            max_logs_per_run=max(counts),
            last_run_logs=last,
            first_run_at=min(started),
            last_run_at=max(finished),
            trend=trend,
        )

    # ------------------------------------------------------------------
    # Stopping
    # ------------------------------------------------------------------
    def stop_job(self, job_id: str) -> StopMonitoringResult | None:
        """Mark a job stopped, unregister the schedule, keep the history.

        Idempotent: stopping an already-stopped job returns the same result.
        """
        job = self._repo.get_job(job_id)
        if job is None:
            return None

        if job.status != JOB_STATUS_STOPPED:
            job = job.model_copy(
                update={
                    "status": JOB_STATUS_STOPPED,
                    "next_run_at": None,
                    "updated_at": self._now(),
                }
            )
            self._repo.update_job(job)

        self._scheduler.unregister(job_id)
        return StopMonitoringResult(job_id=job.job_id, status=job.status)

    # ------------------------------------------------------------------
    # Restart recovery
    # ------------------------------------------------------------------
    def restore_active_jobs(self) -> int:
        """Re-register every active job from SQLite (called on startup).

        No duplicates: ``replace_existing`` is used and SQLite is the source
        of truth, so the same job is restored after any number of restarts.
        """
        active = self._repo.list_jobs(status=JOB_STATUS_ACTIVE)
        now = self._now()
        restored = 0
        for job in active:
            next_run_at = job.next_run_at
            if next_run_at is None or next_run_at <= now:
                # The app was down past the next run: run as soon as possible.
                next_run_at = now
            try:
                self._register(job.job_id, job.interval_seconds, next_run_at)
                restored += 1
            except Exception:  # noqa: BLE001
                logger.exception("Day 18: failed to restore job %s", job.job_id)
        if restored:
            logger.info("Day 18: restored %d active monitoring job(s)", restored)
        return restored

    # ------------------------------------------------------------------
    # Introspection (used by tests)
    # ------------------------------------------------------------------
    def is_running(self, job_id: str) -> bool:
        return job_id in self._running
