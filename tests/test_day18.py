"""Day 18 — scheduled monitoring tests.

No test waits a real interval and no test contacts the real VictoriaLogs:

* scheduled execution is exercised by calling ``service.run_job()`` directly;
* VictoriaLogs is replaced by ``FakeVictoriaLogsClient``;
* the scheduler is a real ``AsyncIOScheduler`` inspected through the wrapper
  (never started, so no timers actually fire);
* restart recovery uses a second service/repository instance over the same
  temporary SQLite file.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes import get_monitoring_service
from app.config import Settings
from app.main import create_app
from app.schemas.day18 import (
    JOB_STATUS_ACTIVE,
    JOB_STATUS_STOPPED,
    MonitoringJob,
    MonitoringRun,
    StartMonitoringInput,
)
from app.services.day18 import (
    Day18MonitoringAgentService,
    MonitoringRepository,
    MonitoringService,
)
from app.services.mcp.monitoring import MonitoringMCPClient
from app.services.mcp.monitoring.server import SERVER_NAME, build_monitoring_server
from app.services.mcp.victorialogs.client import VictoriaLogsTimeoutError


# ----------------------------------------------------------------------
# Helpers / doubles
# ----------------------------------------------------------------------
class FakeVictoriaLogsClient:
    """Deterministic stand-in for the Day 17 VictoriaLogs client.

    ``results`` is a queue consumed per search: an ``int`` yields that many
    synthetic rows, an ``Exception`` is raised. This lets a test express
    "first run fails, second succeeds" without any network.
    """

    def __init__(self, results=None) -> None:
        self.results = list(results if results is not None else [3])
        self.calls: list[dict] = []

    async def search(self, *, query, start, end, limit):
        self.calls.append(
            {"query": query, "start": start, "end": end, "limit": limit}
        )
        if not self.results:
            return [], 0
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return (
            [{"_time": "t", "_msg": "message"} for _ in range(int(item))],
            0,
        )


def _monitoring_service(settings: Settings, client: FakeVictoriaLogsClient, **kwargs):
    return MonitoringService(
        settings, client_factory=lambda: client, **kwargs
    )


def _start_input(**overrides) -> StartMonitoringInput:
    base = {
        "service": "application",
        "level": "ERROR",
        "interval_seconds": 30,
        "lookback_minutes": 5,
        "limit": 100,
    }
    base.update(overrides)
    return StartMonitoringInput(**base)


# ----------------------------------------------------------------------
# 1. Interval whitelist
# ----------------------------------------------------------------------
@pytest.mark.parametrize("value", [10, 30, 60, 300, 600, 1800, 3600])
def test_interval_presets_accepted(value: int) -> None:
    params = StartMonitoringInput(service="application", interval_seconds=value)
    assert params.interval_seconds == value


@pytest.mark.parametrize("value", [11, 20, 120, 0, -10, 3601, 5])
def test_interval_presets_rejected(value: int) -> None:
    with pytest.raises(ValidationError):
        StartMonitoringInput(service="application", interval_seconds=value)


# ----------------------------------------------------------------------
# 2. Lookback bounds
# ----------------------------------------------------------------------
@pytest.mark.parametrize("value", [1, 5, 60, 360])
def test_lookback_bounds_accepted(value: int) -> None:
    params = StartMonitoringInput(service="application", lookback_minutes=value)
    assert params.lookback_minutes == value


@pytest.mark.parametrize("value", [0, -1, 361, 10000])
def test_lookback_bounds_rejected(value: int) -> None:
    with pytest.raises(ValidationError):
        StartMonitoringInput(service="application", lookback_minutes=value)


# ----------------------------------------------------------------------
# 3. Persistence
# ----------------------------------------------------------------------
async def test_job_and_runs_persist_across_instances(settings) -> None:
    client = FakeVictoriaLogsClient(results=[3])
    service = _monitoring_service(settings, client)
    result = await service.start_job(_start_input())

    assert result.first_run is not None
    assert result.first_run.status == "success"
    assert len(service.list_runs(result.job_id)) == 1

    # A brand new repository AND service over the same file must see it.
    repo = MonitoringRepository(settings.day18_monitoring_db_path)
    assert repo.get_job(result.job_id) is not None
    assert len(repo.list_runs(result.job_id)) == 1

    service2 = _monitoring_service(settings, FakeVictoriaLogsClient(results=[9]))
    status = service2.get_status(result.job_id)
    assert status is not None
    assert status.runs_count == 1
    assert status.service == "application"

    # One more run via the second instance is persisted.
    await service2.run_job(result.job_id)
    assert service2.get_status(result.job_id).runs_count == 2


async def test_stop_persists_and_keeps_history(settings) -> None:
    client = FakeVictoriaLogsClient(results=[2, 2])
    service = _monitoring_service(settings, client)
    result = await service.start_job(_start_input())
    await service.run_job(result.job_id)

    stopped = service.stop_job(result.job_id)
    assert stopped is not None
    assert stopped.status == JOB_STATUS_STOPPED

    service2 = _monitoring_service(settings, FakeVictoriaLogsClient())
    status = service2.get_status(result.job_id)
    assert status.status == JOB_STATUS_STOPPED
    assert status.runs_count == 2  # history preserved


# ----------------------------------------------------------------------
# 4. Aggregation
# ----------------------------------------------------------------------
def test_summary_aggregation_is_backend_computed(settings) -> None:
    service = MonitoringService(settings)
    repo = service.repository
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    job = MonitoringJob(
        job_id="mon_agg",
        service="application",
        interval_seconds=30,
        lookback_minutes=5,
        limit=100,
        status=JOB_STATUS_ACTIVE,
        created_at=now,
        updated_at=now,
    )
    repo.create_job(job)
    for index, count in enumerate([10, 20, 5, 25]):
        repo.create_run(
            MonitoringRun(
                job_id="mon_agg",
                started_at=now + timedelta(seconds=index),
                finished_at=now + timedelta(seconds=index, milliseconds=10),
                logs_count=count,
                status="success",
                duration_ms=10,
            )
        )

    summary = service.get_summary("mon_agg")
    assert summary is not None
    assert summary.runs == 4
    assert summary.successful_runs == 4
    assert summary.failed_runs == 0
    assert summary.total_logs == 60
    assert summary.average_logs_per_run == 15
    assert summary.min_logs_per_run == 5
    assert summary.max_logs_per_run == 25
    assert summary.last_run_logs == 25  # newest run first
    assert summary.trend in {"rising", "falling", "stable"}


def test_summary_counts_failures_and_empty_job(settings) -> None:
    service = MonitoringService(settings)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    service.repository.create_job(
        MonitoringJob(
            job_id="mon_mix",
            service="application",
            interval_seconds=60,
            lookback_minutes=5,
            limit=100,
            status=JOB_STATUS_ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    # no runs yet
    empty = service.get_summary("mon_mix")
    assert empty.runs == 0
    assert empty.total_logs == 0

    service.repository.create_run(
        MonitoringRun(
            job_id="mon_mix",
            started_at=now,
            finished_at=now,
            logs_count=0,
            status="error",
            error_message="boom",
            duration_ms=5,
        )
    )
    service.repository.create_run(
        MonitoringRun(
            job_id="mon_mix",
            started_at=now,
            finished_at=now,
            logs_count=4,
            status="success",
            duration_ms=5,
        )
    )
    summary = service.get_summary("mon_mix")
    assert summary.runs == 2
    assert summary.failed_runs == 1
    assert summary.successful_runs == 1
    assert summary.total_logs == 4


# ----------------------------------------------------------------------
# 5. First run + scheduler registration
# ----------------------------------------------------------------------
async def test_first_run_executes_immediately(settings) -> None:
    client = FakeVictoriaLogsClient(results=[7])
    service = _monitoring_service(settings, client)
    result = await service.start_job(_start_input(interval_seconds=30))

    assert result.first_run is not None
    assert result.first_run.logs_count == 7
    assert len(client.calls) == 1  # exactly one immediate run

    status = service.get_status(result.job_id)
    assert status.runs_count == 1
    assert status.last_run_at is not None
    assert status.next_run_at is not None
    assert status.next_run_at > status.last_run_at


async def test_job_registered_with_max_instances_one(settings) -> None:
    service = _monitoring_service(settings, FakeVictoriaLogsClient())
    result = await service.start_job(_start_input(interval_seconds=60))

    assert service.scheduler.is_registered(result.job_id)
    raw_job = service.scheduler.raw.get_job(result.job_id)
    assert raw_job is not None
    assert raw_job.max_instances == 1
    assert raw_job.coalesce is True
    assert raw_job.trigger.interval.total_seconds() == 60


async def test_periodic_run_calls_collection(settings) -> None:
    client = FakeVictoriaLogsClient(results=[2, 5, 8])
    service = _monitoring_service(settings, client)
    result = await service.start_job(_start_input())

    assert len(client.calls) == 1
    run = await service.run_job(result.job_id)
    assert run is not None and run.logs_count == 5
    assert len(client.calls) == 2
    assert service.get_status(result.job_id).runs_count == 2
    # The Day 17 query builder + window are reused.
    assert 'service:"application"' in client.calls[0]["query"]
    assert 'level:"ERROR"' in client.calls[0]["query"]
    window = client.calls[0]["end"] - client.calls[0]["start"]
    assert window == timedelta(minutes=5)


# ----------------------------------------------------------------------
# 6. Error behaviour
# ----------------------------------------------------------------------
async def test_failed_run_does_not_break_next_run(settings) -> None:
    client = FakeVictoriaLogsClient(
        results=[VictoriaLogsTimeoutError("VictoriaLogs request timed out"), 7]
    )
    service = _monitoring_service(settings, client)
    result = await service.start_job(_start_input())

    assert result.first_run is not None
    assert result.first_run.status == "error"
    assert "timed out" in (result.first_run.error_message or "")

    second = await service.run_job(result.job_id)
    assert second is not None
    assert second.status == "success"
    assert second.logs_count == 7
    assert service.get_status(result.job_id).runs_count == 2


async def test_missing_base_url_is_recorded_not_raised(settings) -> None:
    # Default client (no injected fake) with an empty VictoriaLogs URL.
    service = MonitoringService(
        settings.model_copy(update={"victoria_logs_base_url": ""})
    )
    result = await service.start_job(_start_input())
    assert result.first_run is not None
    assert result.first_run.status == "error"
    assert "not configured" in (result.first_run.error_message or "")
    assert result.status == JOB_STATUS_ACTIVE  # job keeps running/retrying


# ----------------------------------------------------------------------
# 7. Concurrency / overlap
# ----------------------------------------------------------------------
async def test_overlapping_run_is_skipped(settings) -> None:
    client = FakeVictoriaLogsClient(results=[1, 1])
    service = _monitoring_service(settings, client)
    result = await service.start_job(_start_input())

    service._running.add(result.job_id)  # simulate an in-flight run
    skipped = await service.run_job(result.job_id)
    assert skipped is None
    assert len(client.calls) == 1  # no second collection
    service._running.discard(result.job_id)


# ----------------------------------------------------------------------
# 8. Stop / scheduler unregister
# ----------------------------------------------------------------------
async def test_stop_unregisters_and_is_idempotent(settings) -> None:
    service = _monitoring_service(settings, FakeVictoriaLogsClient(results=[1, 1]))
    result = await service.start_job(_start_input())
    assert service.scheduler.is_registered(result.job_id)

    first = service.stop_job(result.job_id)
    assert first.status == JOB_STATUS_STOPPED
    assert not service.scheduler.is_registered(result.job_id)

    # Duplicate stop is a no-op returning the same state.
    second = service.stop_job(result.job_id)
    assert second.status == JOB_STATUS_STOPPED

    # A stopped job is never executed again.
    assert await service.run_job(result.job_id) is None


def test_stop_unknown_job_returns_none(settings) -> None:
    service = MonitoringService(settings)
    assert service.stop_job("mon_missing") is None


async def test_stop_during_run_is_not_resurrected(settings) -> None:
    """A job stopped while its run is in flight must not become active again."""
    holder: dict = {}

    class StoppingClient:
        async def search(self, *, query, start, end, limit):
            # Stop the job mid-run (simulates a concurrent /stop request).
            holder["service"].stop_job(holder["job_id"])
            return [{"_time": "t", "_msg": "m"}], 0

    service = MonitoringService(settings, client_factory=lambda: StoppingClient())
    holder["service"] = service

    original_new_job_id = service._new_job_id

    def tracked_new_job_id() -> str:
        job_id = original_new_job_id()
        holder["job_id"] = job_id
        return job_id

    service._new_job_id = tracked_new_job_id  # type: ignore[method-assign]
    result = await service.start_job(_start_input())

    status = service.get_status(result.job_id)
    assert status.status == JOB_STATUS_STOPPED
    assert status.next_run_at is None
    assert status.runs_count == 1
    assert not service.scheduler.is_registered(result.job_id)


# ----------------------------------------------------------------------
# 9. Restart recovery
# ----------------------------------------------------------------------
async def test_restore_active_jobs_after_restart(settings) -> None:
    client = FakeVictoriaLogsClient(results=[1, 1, 1, 1])
    service = _monitoring_service(settings, client)
    active = await service.start_job(_start_input())
    stopped = await service.start_job(_start_input(service="other"))

    assert service.stop_job(stopped.job_id).status == JOB_STATUS_STOPPED

    # Simulate a restart: new scheduler, new service, same SQLite file.
    restarted = MonitoringService(
        settings, client_factory=lambda: FakeVictoriaLogsClient()
    )
    restored = restarted.restore_active_jobs()

    assert restored == 1
    assert restarted.scheduler.is_registered(active.job_id)
    assert not restarted.scheduler.is_registered(stopped.job_id)

    # Restoring twice never creates duplicates.
    assert restarted.restore_active_jobs() == 1
    assert len(restarted.scheduler.raw.get_jobs()) == 1


def test_restart_recovery_runs_through_app_lifespan(settings) -> None:
    # Seed an active job directly into the temporary DB, then start the app:
    # the lifespan must restore it into the real scheduler.
    now = datetime.now(timezone.utc)
    MonitoringRepository(settings.day18_monitoring_db_path).create_job(
        MonitoringJob(
            job_id="mon_restart",
            service="application",
            interval_seconds=3600,
            lookback_minutes=5,
            limit=100,
            status=JOB_STATUS_ACTIVE,
            created_at=now,
            updated_at=now,
            next_run_at=now + timedelta(hours=1),
        )
    )

    app = create_app(settings)
    with TestClient(app):
        service = app.state.monitoring_service
        assert service.scheduler.is_registered("mon_restart")
    # Scheduler is stopped on shutdown.
    assert not service.scheduler.running


# ----------------------------------------------------------------------
# 10. MCP tools: registration, schema and the real in-memory call path
# ----------------------------------------------------------------------
async def test_monitoring_tools_registered_with_schemas(settings) -> None:
    server = build_monitoring_server(MonitoringService(settings))
    tools = await server.list_tools()
    names = {tool.name for tool in tools}
    assert names == {
        "start_monitoring",
        "get_monitoring_status",
        "get_monitoring_summary",
        "stop_monitoring",
    }

    start = next(tool for tool in tools if tool.name == "start_monitoring")
    props = start.inputSchema["properties"]
    assert set(props) >= {
        "service",
        "level",
        "interval_seconds",
        "lookback_minutes",
        "text_contains",
        "limit",
    }
    assert start.inputSchema["required"] == ["service"]
    assert props["interval_seconds"]["enum"] == [10, 30, 60, 300, 600, 1800, 3600]
    assert props["lookback_minutes"]["minimum"] == 1
    assert props["lookback_minutes"]["maximum"] == 360
    assert props["limit"]["maximum"] == 500


async def test_mcp_start_status_summary_stop_via_real_call(settings) -> None:
    client = FakeVictoriaLogsClient(results=[4, 6])
    service = _monitoring_service(settings, client)
    mcp = MonitoringMCPClient(service)

    discovery = await mcp.discover_tools()
    assert discovery.connected is True
    assert discovery.server.name == SERVER_NAME
    assert discovery.server.transport == "in-memory"

    started = await mcp.call_tool(
        "start_monitoring",
        {
            "service": "application",
            "level": "ERROR",
            "interval_seconds": 30,
            "lookback_minutes": 5,
        },
    )
    assert started.is_error is False
    payload = started.as_payload()
    job_id = payload["job_id"]
    assert job_id.startswith("mon_")
    assert service.get_job(job_id) is not None

    # A second run then a summary through MCP.
    await service.run_job(job_id)
    summary_result = await mcp.call_tool(
        "get_monitoring_summary", {"job_id": job_id, "last_runs": 10}
    )
    summary = summary_result.as_payload()
    assert summary["runs"] == 2
    assert summary["total_logs"] == 10

    status_result = await mcp.call_tool(
        "get_monitoring_status", {"job_id": job_id}
    )
    assert status_result.as_payload()["runs_count"] == 2

    stopped = await mcp.call_tool("stop_monitoring", {"job_id": job_id})
    assert stopped.as_payload()["status"] == JOB_STATUS_STOPPED


async def test_mcp_rejects_invalid_interval(settings) -> None:
    service = MonitoringService(settings)
    mcp = MonitoringMCPClient(service)
    result = await mcp.call_tool(
        "start_monitoring", {"service": "application", "interval_seconds": 11}
    )
    assert result.is_error is True
    assert service.list_jobs() == []


async def test_mcp_unknown_job_is_controlled(settings) -> None:
    service = MonitoringService(settings)
    mcp = MonitoringMCPClient(service)
    result = await mcp.call_tool(
        "get_monitoring_status", {"job_id": "mon_does_not_exist"}
    )
    payload = result.as_payload()
    assert payload["error_kind"] == "not_found"
    assert "unknown" in payload["error"]


# ----------------------------------------------------------------------
# 11. Agent flow (fake LLM selects monitoring tools)
# ----------------------------------------------------------------------
async def test_agent_starts_monitoring(settings, fake_service) -> None:
    from conftest import FakeDeepSeekService

    assert isinstance(fake_service, FakeDeepSeekService)
    client = FakeVictoriaLogsClient(results=[5])
    service = _monitoring_service(settings, client)
    fake_service.tool_sequence = [
        {
            "content": "",
            "finish_reason": "tool_calls",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "start_monitoring",
                    "arguments": {
                        "service": "application",
                        "level": "ERROR",
                        "interval_seconds": 30,
                        "lookback_minutes": 5,
                    },
                }
            ],
        },
        {
            "content": "Мониторинг запущен. Job ID: mon_x",
            "finish_reason": "stop",
            "tool_calls": [],
        },
    ]
    agent = Day18MonitoringAgentService(
        settings, service, deepseek=fake_service
    )
    response = await agent.chat(
        "Начни мониторить ERROR для application каждые 30 секунд"
    )

    assert response.error is None
    assert "Мониторинг" in response.answer
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.server == "monitoring"
    assert call.tool == "start_monitoring"
    assert call.result_summary["job_id"].startswith("mon_")
    assert len(service.list_jobs()) == 1

    # The monitoring tool result reached the second LLM call as a tool message.
    second_messages = fake_service.generate_with_tools_calls[1]["messages"]
    tool_messages = [m for m in second_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "job_id" in tool_messages[0]["content"]

    steps = [step.step for step in response.trace]
    assert "discover_tools" in steps
    assert "llm_select" in steps
    assert "call_tool" in steps
    assert "monitoring_job" in steps
    assert "llm_final" in steps


async def test_agent_summary_and_stop(settings, fake_service) -> None:
    client = FakeVictoriaLogsClient(results=[3, 9])
    service = _monitoring_service(settings, client)
    started = await service.start_job(_start_input())
    await service.run_job(started.job_id)

    fake_service.tool_sequence = [
        {
            "content": "",
            "finish_reason": "tool_calls",
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "get_monitoring_summary",
                    "arguments": {"job_id": started.job_id},
                }
            ],
        },
        {"content": "Сводка готова.", "finish_reason": "stop", "tool_calls": []},
        {
            "content": "",
            "finish_reason": "tool_calls",
            "tool_calls": [
                {
                    "id": "c2",
                    "name": "stop_monitoring",
                    "arguments": {"job_id": started.job_id},
                }
            ],
        },
        {"content": "Мониторинг остановлен.", "finish_reason": "stop", "tool_calls": []},
    ]
    agent = Day18MonitoringAgentService(settings, service, deepseek=fake_service)

    summary_response = await agent.chat(
        f"Покажи сводку мониторинга {started.job_id}"
    )
    assert summary_response.error is None
    assert summary_response.tool_calls[0].tool == "get_monitoring_summary"
    assert summary_response.tool_calls[0].result_summary["runs"] == 2

    stop_response = await agent.chat(f"Останови мониторинг {started.job_id}")
    assert stop_response.error is None
    assert stop_response.tool_calls[0].tool == "stop_monitoring"
    assert stop_response.tool_calls[0].result_summary["status"] == JOB_STATUS_STOPPED
    assert service.get_status(started.job_id).status == JOB_STATUS_STOPPED


# ----------------------------------------------------------------------
# 12. HTTP dashboard API
# ----------------------------------------------------------------------
async def test_dashboard_api_start_run_summary_stop(client, app, settings) -> None:
    fake = FakeVictoriaLogsClient(results=[2, 4])
    service = _monitoring_service(settings, fake)
    app.dependency_overrides[get_monitoring_service] = lambda: service
    try:
        start = client.post(
            "/api/week4/day18/monitoring",
            json={
                "service": "application",
                "level": "ERROR",
                "interval_seconds": 30,
                "lookback_minutes": 5,
            },
        )
        assert start.status_code == 200
        job_id = start.json()["job_id"]

        # The dashboard refresh source: the job list must expose the job.
        jobs_response = client.get("/api/week4/day18/monitoring")
        assert jobs_response.status_code == 200
        listed = jobs_response.json()
        assert any(job["job_id"] == job_id for job in listed)

        await service.run_job(job_id)

        status_response = client.get(f"/api/week4/day18/monitoring/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["runs_count"] == 2

        runs_response = client.get(f"/api/week4/day18/monitoring/{job_id}/runs")
        assert len(runs_response.json()["runs"]) == 2

        summary_response = client.get(
            f"/api/week4/day18/monitoring/{job_id}/summary?last_runs=10"
        )
        assert summary_response.json()["total_logs"] == 6
        assert summary_response.json()["runs"] == 2

        stop_response = client.post(
            f"/api/week4/day18/monitoring/{job_id}/stop"
        )
        assert stop_response.json()["status"] == JOB_STATUS_STOPPED
    finally:
        app.dependency_overrides.pop(get_monitoring_service, None)


def test_dashboard_api_validation_and_404(client) -> None:
    bad_interval = client.post(
        "/api/week4/day18/monitoring",
        json={"service": "application", "interval_seconds": 11},
    )
    assert bad_interval.status_code == 422

    bad_lookback = client.post(
        "/api/week4/day18/monitoring",
        json={"service": "application", "lookback_minutes": 0},
    )
    assert bad_lookback.status_code == 422

    assert client.get("/api/week4/day18/monitoring/mon_none").status_code == 404
    assert client.get("/api/week4/day18/monitoring/mon_none/runs").status_code == 404
    assert (
        client.get("/api/week4/day18/monitoring/mon_none/summary").status_code == 404
    )
    assert (
        client.post("/api/week4/day18/monitoring/mon_none/stop").status_code == 404
    )


def test_day18_chat_endpoint(client) -> None:
    response = client.post(
        "/api/week4/day18/chat",
        json={"message": "Покажи сводку мониторинга"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "trace" in payload

    assert client.post("/api/week4/day18/chat", json={"message": ""}).status_code == 422


# ----------------------------------------------------------------------
# 13. Privacy: no raw logs / secrets in SQLite
# ----------------------------------------------------------------------
def test_sqlite_schema_stores_aggregates_only(settings) -> None:
    service = MonitoringService(settings)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    service.repository.create_job(
        MonitoringJob(
            job_id="mon_priv",
            service="application",
            interval_seconds=30,
            lookback_minutes=5,
            limit=100,
            status=JOB_STATUS_ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    service.repository.create_run(
        MonitoringRun(
            job_id="mon_priv",
            started_at=now,
            finished_at=now,
            logs_count=3,
            status="success",
        )
    )

    conn = sqlite3.connect(settings.day18_monitoring_db_path)
    try:
        run_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(monitor_runs)")
        ]
        job_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(monitor_jobs)")
        ]
    finally:
        conn.close()

    # No column can hold a raw log message/body or a secret. ``error_message``
    # is allowed: it stores a short controlled error string, never a log row.
    forbidden = {
        "message",
        "msg",
        "raw",
        "raw_log",
        "raw_logs",
        "log_message",
        "logs",
        "body",
        "url",
        "base_url",
        "token",
        "secret",
        "password",
        "authorization",
    }
    for column in run_columns + job_columns:
        assert column.lower() not in forbidden, f"sensitive column: {column}"


# ----------------------------------------------------------------------
# 14. UI
# ----------------------------------------------------------------------
def test_week4_day18_enabled_and_future_days_disabled(client) -> None:
    import re

    html = client.get("/").text
    assert "Day 18 — Scheduled Monitoring" in html
    assert 'id="panel-day18"' in html
    assert 'src="/static/js/day18.js"' in html

    day18 = re.search(r'<button[^>]*id="tab-day18"[^>]*>', html)
    assert day18 and "disabled" not in day18.group(0)
    # Day 19 is implemented and enabled; only Day 20 stays disabled.
    day19 = re.search(r'<button[^>]*id="tab-day19"[^>]*>', html)
    assert day19 and "disabled" not in day19.group(0)
    # Day 18 is the default (active) landing tab.
    assert 'id="tab-day18"' in html
    assert re.search(r'id="tab-day18"[^>]*aria-selected="true"', html)
    assert re.search(r'id="tab-day17"[^>]*aria-selected="false"', html)
    for day in ("day20",):
        btn = re.search(r'<button[^>]*id="tab-' + day + r'"[^>]*>', html)
        assert btn and "disabled" in btn.group(0), f"{day} must stay disabled"


def test_day18_interval_and_lookback_options(client) -> None:
    import re

    html = client.get("/").text
    for label in ("10 sec", "30 sec", "1 min", "5 min", "10 min", "30 min", "1 hour"):
        assert label in html
    # Default interval is 30 sec and default lookback is 5 min.
    assert re.search(r'<option value="30" selected>30 sec</option>', html)
    assert re.search(r'<option value="5" selected>5 min</option>', html)
    # Demo-friendly values are selectable.
    assert re.search(r'<option value="10">10 sec</option>', html)
    assert re.search(r'<option value="3600">1 hour</option>', html)
    # Interval and lookback are two separate controls.
    assert 'id="d18-interval"' in html
    assert 'id="d18-lookback"' in html


def test_day18_job_card_history_and_summary_elements(client) -> None:
    html = client.get("/").text
    for element in (
        'id="d18-job-section"',
        'id="d18-job-status"',
        'id="d18-job-interval"',
        'id="d18-runs-section"',
        'id="d18-runs-body"',
        'id="d18-summary-section"',
        'id="d18-sum-avg"',
        'id="d18-trace-section"',
    ):
        assert element in html


def test_day18_js_dom_references_exist(client) -> None:
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(
        re.findall(
            r'\$\("([^"]+)"\)',
            (base / "day18.js").read_text(encoding="utf-8"),
        )
    )
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day18.js references missing ids: {missing}"


def test_day18_frontend_does_not_hardcode_secrets_or_logs() -> None:
    from pathlib import Path

    js = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "static"
        / "js"
        / "day18.js"
    ).read_text(encoding="utf-8")
    assert "/api/week4/day18/monitoring" in js
    assert "/api/week4/day18/chat" in js
    assert "VICTORIA_LOGS" not in js
    assert "monitoring.db" not in js


def test_day18_frontend_renders_history_and_failures() -> None:
    from pathlib import Path

    js = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "static"
        / "js"
        / "day18.js"
    ).read_text(encoding="utf-8")
    # Run history rendering (job card + rows) and failure rendering exist.
    assert "renderRuns" in js
    assert "d18-run-error" in js
    assert "run.error_message" in js
    assert "renderSummary" in js
    # Polling is display-only and uses exactly one timer variable.
    assert "POLL_MS" in js
    assert "setInterval" in js
    # The dashboard restores itself on load and after the agent flow.
    assert "loadLatestJob" in js
    assert 'jsonFetch("/api/week4/day18/monitoring")' in js
