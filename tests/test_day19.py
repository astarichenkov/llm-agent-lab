"""Day 19 — MCP tool composition pipeline tests.

No test contacts the real VictoriaLogs or the real DeepSeek API:

* VictoriaLogs is replaced by ``FakeVictoriaLogs``;
* DeepSeek is replaced by the shared ``FakeDeepSeekService`` scripted via
  ``generate_sequence`` (analysis JSON, then the final textual answer);
* artifacts are written to a temporary directory
  (``settings.day19_artifact_root``).

The full MCP path (``initialize`` -> ``tools/list`` -> three ``tools/call``)
is exercised through the real in-memory MCP session, so the composition is
genuinely tested, not simulated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas.day19 import (
    MAX_ANALYSIS_LOGS,
    MAX_LOG_MESSAGE_CHARS,
    AnalyzeLogsInput,
    Day19PipelineRequest,
    ErrorGroup,
    LogAnalysis,
    SaveReportInput,
    SearchLogsPipelineInput,
)
from app.services.day19 import Day19PipelineService, Day19Toolset
from app.services.day19.analysis import (
    AnalysisError,
    parse_analysis,
    run_structured_analysis,
    select_logs_for_analysis,
)
from app.services.day19.artifacts import ArtifactError, ArtifactStore, validate_run_id
from app.services.deepseek import DeepSeekTimeoutError
from app.services.mcp.pipeline import PipelineMCPClient, build_pipeline_server
from app.services.mcp.pipeline.server import SERVER_NAME
from app.services.mcp.victorialogs.client import VictoriaLogsTimeoutError


# ----------------------------------------------------------------------
# Helpers / doubles
# ----------------------------------------------------------------------
VALID_ANALYSIS = {
    "summary": "За период наблюдались таймауты подключения к БД и ответы 502.",
    "error_groups": [
        {
            "pattern": "connection timeout",
            "count": 1,
            "examples": ["connection timeout to db"],
        }
    ],
    "possible_causes": ["Возможно, база данных недоступна или перегружена."],
    "notable_patterns": ["ошибки повторяются в течение всего окна"],
    "recommended_checks": ["Проверить доступность БД и её нагрузку."],
}


class FakeVictoriaLogs:
    """Deterministic stand-in for the Day 17 VictoriaLogs client."""

    def __init__(self, rows=None, error: Exception | None = None) -> None:
        self.rows = rows if rows is not None else [
            {
                "_time": "2026-09-28T10:00:00Z",
                "_msg": "connection timeout to db",
                "level": "ERROR",
            },
            {
                "_time": "2026-09-28T10:01:00Z",
                "_msg": "502 bad gateway",
                "level": "ERROR",
            },
        ]
        self.error = error
        self.calls: list[dict] = []

    async def search(self, *, query, start, end, limit):
        self.calls.append({"query": query, "start": start, "end": end, "limit": limit})
        if self.error is not None:
            raise self.error
        return list(self.rows), 0


class RecordingMCPClient:
    """Wraps a real ``PipelineMCPClient`` and records the call order/args."""

    def __init__(self, inner: PipelineMCPClient) -> None:
        self.inner = inner
        self.calls: list[tuple[str, dict]] = []

    async def discover_tools(self):
        return await self.inner.discover_tools()

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return await self.inner.call_tool(name, arguments)


def _toolset(settings, fake_service, rows=None, error=None) -> Day19Toolset:
    return Day19Toolset(
        settings,
        deepseek=fake_service,
        client_factory=lambda: FakeVictoriaLogs(rows, error),
    )


def _service(settings, fake_service, *, rows=None, error=None, recording=False):
    toolset = _toolset(settings, fake_service, rows, error)
    mcp = PipelineMCPClient(toolset)
    if recording:
        recorder = RecordingMCPClient(mcp)
        service = Day19PipelineService(
            settings, toolset=toolset, deepseek=fake_service, mcp_client=recorder
        )
        return service, recorder
    return Day19PipelineService(
        settings, toolset=toolset, deepseek=fake_service, mcp_client=mcp
    ), mcp


def _request(**overrides) -> Day19PipelineRequest:
    base = {
        "service": "application",
        "level": "ERROR",
        "since_minutes": 30,
        "limit": 100,
        "question": "Найди основные проблемы и возможные причины",
    }
    base.update(overrides)
    return Day19PipelineRequest(**base)


# ----------------------------------------------------------------------
# 1. MCP tool registration / descriptions / schemas
# ----------------------------------------------------------------------
async def test_pipeline_tools_registered_with_description_and_schema(settings) -> None:
    server = build_pipeline_server(_toolset(settings, None))
    tools = await server.list_tools()
    names = {tool.name for tool in tools}
    assert names == {"search_logs", "analyze_logs", "save_report"}
    for tool in tools:
        assert tool.description, f"{tool.name} has no description"

    search = next(tool for tool in tools if tool.name == "search_logs")
    props = search.inputSchema["properties"]
    assert set(props) >= {"service", "since_minutes", "level", "text_contains", "limit"}
    assert search.inputSchema["required"] == ["service"]
    assert props["limit"]["maximum"] == 500
    assert props["since_minutes"]["maximum"] == 360

    analyze = next(tool for tool in tools if tool.name == "analyze_logs")
    assert set(analyze.inputSchema["properties"]) >= {"service", "logs", "question"}

    save = next(tool for tool in tools if tool.name == "save_report")
    assert set(save.inputSchema["properties"]) >= {"run_id", "service", "analysis", "logs"}


def test_pipeline_server_name_constant() -> None:
    assert SERVER_NAME == "Pipeline MCP"


# ----------------------------------------------------------------------
# 2. Input validation / limits
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"service": "svc", "since_minutes": 0},
        {"service": "svc", "since_minutes": 361},
        {"service": "svc", "limit": 0},
        {"service": "svc", "limit": 501},
        {"service": ""},
    ],
)
def test_search_input_rejects_out_of_range(kwargs) -> None:
    with pytest.raises(ValidationError):
        SearchLogsPipelineInput(**kwargs)


def test_search_input_defaults() -> None:
    params = SearchLogsPipelineInput(service="svc")
    assert params.since_minutes == 30
    assert params.limit == 100


def test_analysis_selection_applies_hard_limits() -> None:
    rows = [
        {"timestamp": "t", "message": f"m{i}", "fields": {}} for i in range(150)
    ]
    selected, analyzed, truncated = select_logs_for_analysis(rows)
    assert len(selected) == MAX_ANALYSIS_LOGS
    assert analyzed == MAX_ANALYSIS_LOGS
    assert truncated is True


def test_analysis_selection_truncates_long_messages() -> None:
    rows = [{"timestamp": "t", "message": "x" * 5000, "fields": {}}]
    selected, analyzed, truncated = select_logs_for_analysis(rows)
    assert len(selected[0]["message"]) == MAX_LOG_MESSAGE_CHARS
    assert analyzed == 1
    assert truncated is True


# ----------------------------------------------------------------------
# 3. LLM structured analysis: valid, invalid JSON, missing field, error
# ----------------------------------------------------------------------
async def test_structured_analysis_valid(fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS)]
    analysis = await run_structured_analysis(
        fake_service, messages=[{"role": "user", "content": "x"}], model="m"
    )
    assert analysis.summary.startswith("За период")
    assert analysis.error_groups[0].pattern == "connection timeout"
    assert analysis.possible_causes


async def test_structured_analysis_invalid_json(fake_service) -> None:
    fake_service.generate_sequence = ["not a json object"]
    with pytest.raises(AnalysisError):
        await run_structured_analysis(
            fake_service, messages=[{"role": "user", "content": "x"}], model="m"
        )


async def test_structured_analysis_missing_required_field(fake_service) -> None:
    fake_service.generate_sequence = [json.dumps({"error_groups": []})]
    with pytest.raises(AnalysisError):
        await run_structured_analysis(
            fake_service, messages=[{"role": "user", "content": "x"}], model="m"
        )


async def test_structured_analysis_provider_error(fake_service) -> None:
    fake_service.raise_error = DeepSeekTimeoutError()
    with pytest.raises(AnalysisError):
        await run_structured_analysis(
            fake_service, messages=[{"role": "user", "content": "x"}], model="m"
        )


def test_parse_analysis_accepts_markdown_fenced_json() -> None:
    text = "```json\n" + json.dumps(VALID_ANALYSIS) + "\n```"
    analysis = parse_analysis(text)
    assert isinstance(analysis, LogAnalysis)


# ----------------------------------------------------------------------
# 4. Real MCP flow: initialize -> tools/list -> 3x tools/call
# ----------------------------------------------------------------------
async def test_real_mcp_chain_search_analyze_save(settings, fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS)]
    toolset = _toolset(settings, fake_service)
    mcp = PipelineMCPClient(toolset)

    discovery = await mcp.discover_tools()
    assert discovery.connected is True
    assert discovery.server.name == SERVER_NAME
    assert discovery.server.transport == "in-memory"
    assert discovery.tools_count == 3

    search = await mcp.call_tool(
        "search_logs", {"service": "application", "level": "ERROR"}
    )
    search_payload = search.as_payload()
    assert search_payload["count"] == 2

    analyze = await mcp.call_tool(
        "analyze_logs",
        {
            "service": "application",
            "question": "q",
            "logs": search_payload["logs"],
        },
    )
    analyze_payload = analyze.as_payload()
    assert analyze_payload["error_groups_count"] == 1

    run_id = toolset.new_run_id()
    save = await mcp.call_tool(
        "save_report",
        {
            "run_id": run_id,
            "service": "application",
            "analysis": toolset.analysis_model(analyze_payload).model_dump(),
            "logs": search_payload["logs"],
        },
    )
    save_payload = save.as_payload()
    run_dir = Path(settings.day19_artifact_root) / run_id
    assert (run_dir / "raw.jsonl").is_file()
    assert (run_dir / "analysis.md").is_file()
    assert (run_dir / "metadata.json").is_file()
    assert save_payload["analysis"] == "analysis.md"


# ----------------------------------------------------------------------
# 5. Data passing between tools (the key Day 19 requirement)
# ----------------------------------------------------------------------
async def test_pipeline_passes_exact_logs_and_analysis(settings, fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "готово"]
    service, recorder = _service(settings, fake_service, recording=True)

    response = await service.run(_request())
    assert response.status == "completed"

    names = [name for name, _ in recorder.calls]
    assert names == ["search_logs", "analyze_logs", "save_report"]

    search_args = recorder.calls[0][1]
    analyze_args = recorder.calls[1][1]
    save_args = recorder.calls[2][1]

    # The exact sanitized rows returned by search reach analyze.
    assert analyze_args["logs"] == [
        {
            "timestamp": "2026-09-28T10:00:00Z",
            "message": "connection timeout to db",
            "fields": {"level": "ERROR"},
        },
        {
            "timestamp": "2026-09-28T10:01:00Z",
            "message": "502 bad gateway",
            "fields": {"level": "ERROR"},
        },
    ]
    assert search_args["service"] == "application"

    # The exact structured analysis reaches save.
    assert save_args["analysis"]["summary"] == VALID_ANALYSIS["summary"]
    assert save_args["analysis"]["error_groups"][0]["pattern"] == "connection timeout"
    assert save_args["logs"] == analyze_args["logs"]


async def test_pipeline_trace_shows_data_handoff(settings, fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "готово"]
    service, _ = _service(settings, fake_service)
    response = await service.run(_request())

    steps = {step.step: step for step in response.trace}
    assert steps["handoff_search_logs_to_analyze_logs"].details["logs_passed"] == 2
    assert (
        steps["handoff_analyze_logs_to_save_report"].details["error_groups"] == 1
    )
    assert steps["search_logs"].details["logs_received"] == 2
    assert steps["save_report"].details["artifacts"] == [
        "raw.jsonl",
        "analysis.md",
        "metadata.json",
    ]


# ----------------------------------------------------------------------
# 6. Pipeline order
# ----------------------------------------------------------------------
async def test_pipeline_order_is_always_search_analyze_save(settings, fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "ok"]
    service, recorder = _service(settings, fake_service, recording=True)
    await service.run(_request())
    assert [name for name, _ in recorder.calls] == [
        "search_logs",
        "analyze_logs",
        "save_report",
    ]
    # Never save before analyze.
    assert recorder.calls[0][0] != "save_report"


# ----------------------------------------------------------------------
# 7. Failure behaviour (short-circuiting)
# ----------------------------------------------------------------------
async def test_search_failure_skips_analyze_and_save(settings, fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS)]
    service, recorder = _service(
        settings, fake_service, error=VictoriaLogsTimeoutError("timed out"), recording=True
    )
    response = await service.run(_request())

    assert response.status == "failed"
    assert "timed out" in (response.error or "")
    assert [name for name, _ in recorder.calls] == ["search_logs"]
    statuses = {step.step: step.status for step in response.trace}
    assert statuses["analyze_logs"] == "skipped"
    assert statuses["save_report"] == "skipped"


async def test_analyze_failure_skips_save(settings, fake_service) -> None:
    fake_service.generate_sequence = ["not json"]
    service, recorder = _service(settings, fake_service, recording=True)
    response = await service.run(_request())

    assert response.status == "failed"
    assert [name for name, _ in recorder.calls] == ["search_logs", "analyze_logs"]
    assert not any(name == "save_report" for name, _ in recorder.calls)
    statuses = {step.step: step.status for step in response.trace}
    assert statuses["analyze_logs"] == "error"
    assert statuses["save_report"] == "skipped"


async def test_save_failure_marks_pipeline_failed(settings, fake_service, monkeypatch) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "ok"]
    service, _ = _service(settings, fake_service)

    def boom(**kwargs):
        raise ArtifactError("disk full")

    monkeypatch.setattr(service.artifact_store, "save", boom)
    response = await service.run(_request())

    assert response.status == "failed"
    assert "disk full" in (response.error or "")
    assert response.artifacts is None
    statuses = {step.step: step.status for step in response.trace}
    assert statuses["save_report"] == "error"


# ----------------------------------------------------------------------
# 7b. Empty search result: explicit, no LLM, still a valid run
# ----------------------------------------------------------------------
async def test_empty_search_is_safe_and_does_not_call_llm(settings, fake_service) -> None:
    service, recorder = _service(settings, fake_service, rows=[], recording=True)
    response = await service.run(_request(since_minutes=1))

    assert response.status == "completed"
    assert response.search is not None and response.search.count == 0
    assert response.analysis is not None
    assert response.analysis.error_groups_count == 0
    assert response.analysis.logs_analyzed == 0
    assert "логи не найдены" in response.analysis.summary
    # No LLM call at all: neither analysis nor final answer.
    assert fake_service.generate_calls == []
    # The deterministic no-data answer is used instead.
    assert "логи не найдены" in response.answer

    # raw.jsonl exists but is accurately empty; metadata captures 0 logs.
    run_dir = Path(settings.day19_artifact_root) / response.artifacts.run_id
    assert (run_dir / "raw.jsonl").read_text(encoding="utf-8") == ""
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["logs_received"] == 0
    assert metadata["logs_analyzed"] == 0

    # The chain/order is unchanged even with no data.
    assert [name for name, _ in recorder.calls] == [
        "search_logs",
        "analyze_logs",
        "save_report",
    ]


# ----------------------------------------------------------------------
# 8. Artifacts
# ----------------------------------------------------------------------
async def test_artifacts_are_written_and_valid(settings, fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "ok"]
    service, _ = _service(settings, fake_service)
    response = await service.run(_request())

    assert response.artifacts is not None
    run_dir = Path(settings.day19_artifact_root) / response.artifacts.run_id
    assert (run_dir / "raw.jsonl").is_file()
    assert (run_dir / "analysis.md").is_file()
    assert (run_dir / "metadata.json").is_file()

    # raw.jsonl: one valid JSON object per line.
    raw_lines = (run_dir / "raw.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(raw_lines) == 2
    for line in raw_lines:
        parsed = json.loads(line)
        assert set(parsed) == {"timestamp", "message", "fields"}

    markdown = (run_dir / "analysis.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Stage Log Analysis")
    assert "## Summary" in markdown
    assert "## Error groups" in markdown
    assert "## Possible causes" in markdown
    assert "## Recommended checks" in markdown

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == response.artifacts.run_id
    assert metadata["service"] == "application"
    assert metadata["logs_received"] == 2
    assert metadata["logs_analyzed"] == 2
    assert metadata["artifacts"]["raw"] == "raw.jsonl"


# ----------------------------------------------------------------------
# 9. Sanitization: no secrets in artifacts or in the analyze input
# ----------------------------------------------------------------------
async def test_secrets_are_sanitized_before_analyze_and_save(settings, fake_service) -> None:
    secret_rows = [
        {
            "_time": "2026-09-28T10:00:00Z",
            "_msg": "Authorization: Bearer abc123 password=secret token=qwerty",
            "level": "ERROR",
        }
    ]
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "ok"]
    service, recorder = _service(settings, fake_service, rows=secret_rows, recording=True)
    response = await service.run(_request())
    assert response.status == "completed"

    analyze_logs = recorder.calls[1][1]["logs"]
    analyze_blob = json.dumps(analyze_logs)
    for secret in ("abc123", "secret", "qwerty"):
        assert secret not in analyze_blob
    assert "[REDACTED]" in analyze_blob

    run_dir = Path(settings.day19_artifact_root) / response.artifacts.run_id
    for name in ("raw.jsonl", "analysis.md", "metadata.json"):
        content = (run_dir / name).read_text(encoding="utf-8")
        for secret in ("abc123", "secret", "qwerty"):
            assert secret not in content, f"{secret} leaked into {name}"


def test_save_report_sanitizes_direct_input(settings, fake_service, tmp_path) -> None:
    # Even if a caller submits an unsanitized row directly, save_report masks it.
    store = ArtifactStore(tmp_path / "runs")
    toolset = Day19Toolset(settings, deepseek=fake_service, artifact_store=store)
    import asyncio

    params = SaveReportInput(
        run_id="run_direct",
        service="application",
        logs=[
            {
                "timestamp": "t",
                "message": "password=topsecret",
                "fields": {"authorization": "Bearer zzz999"},
            }
        ],
        analysis=LogAnalysis(**VALID_ANALYSIS),
    )
    asyncio.run(toolset.save_report(params))
    raw = (tmp_path / "runs" / "run_direct" / "raw.jsonl").read_text(encoding="utf-8")
    assert "topsecret" not in raw
    assert "zzz999" not in raw


# ----------------------------------------------------------------------
# 10. Path traversal / run id security
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        "../../secret",
        "..\\..\\secret",
        "C:\\Windows\\system32",
        "/etc/passwd",
        "a/b",
        "a\\b",
        ".",
        "..",
        "run id with spaces",
    ],
)
def test_validate_run_id_rejects_unsafe_values(bad: str) -> None:
    with pytest.raises(ArtifactError):
        validate_run_id(bad)


def test_artifact_store_rejects_traversal(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "runs")
    for bad in ("../../secret", "C:\\Windows", "/etc/passwd"):
        with pytest.raises(ArtifactError):
            store.run_dir(bad)


def test_artifact_store_rejects_unknown_filename(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "runs")
    with pytest.raises(ArtifactError):
        store.read("run_safe", "secrets.txt")


def test_save_report_rejects_bad_run_id(settings, fake_service) -> None:
    import asyncio

    toolset = _toolset(settings, fake_service)
    result = asyncio.run(
        toolset.save_report(
            SaveReportInput(
                run_id="../../escape",
                service="application",
                analysis=LogAnalysis(**VALID_ANALYSIS),
            )
        )
    )
    assert result["error_kind"] == "artifact"


# ----------------------------------------------------------------------
# 11. Full agent pipeline + final answer
# ----------------------------------------------------------------------
async def test_full_pipeline_returns_final_answer_and_trace(settings, fake_service) -> None:
    fake_service.generate_sequence = [
        json.dumps(VALID_ANALYSIS),
        "Анализ завершён. Найдена одна группа ошибок.",
    ]
    service, _ = _service(settings, fake_service)
    response = await service.run(_request())

    assert response.status == "completed"
    assert response.pipeline_id.startswith("run_")
    assert "одна группа" in response.answer
    steps = [step.step for step in response.trace]
    assert "pipeline_started" in steps
    assert "discover_tools" in steps
    assert "pipeline_completed" in steps


async def test_final_answer_falls_back_when_llm_fails(settings, fake_service) -> None:
    # First generate -> analysis JSON; second generate -> provider error.
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS)]
    # Make the NEXT call fail by consuming the sequence then raising.
    service, _ = _service(settings, fake_service)

    original_generate = fake_service.generate

    async def flaky_generate(*args, **kwargs):
        if not fake_service.generate_sequence:
            raise DeepSeekTimeoutError()
        return await original_generate(*args, **kwargs)

    fake_service.generate = flaky_generate  # type: ignore[method-assign]
    response = await service.run(_request())

    assert response.status == "completed"
    assert "Анализ завершён" in response.answer  # deterministic fallback


# ----------------------------------------------------------------------
# 12. HTTP API
# ----------------------------------------------------------------------
def test_day19_pipeline_endpoint(client, fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "готово"]
    response = client.post(
        "/api/week4/day19/pipeline",
        json={
            "service": "application",
            "level": "ERROR",
            "since_minutes": 30,
            "limit": 100,
            "question": "Найди основные проблемы",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["search"]["count"] == 2
    assert data["analysis"]["error_groups_count"] == 1
    assert data["artifacts"]["run_id"].startswith("run_")
    assert data["answer"] == "готово"
    assert data["trace"]


def test_day19_artifact_endpoint(client, fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "готово"]
    data = client.post(
        "/api/week4/day19/pipeline",
        json={"service": "application", "level": "ERROR"},
    ).json()
    run_id = data["artifacts"]["run_id"]

    analysis = client.get(f"/api/week4/day19/runs/{run_id}/artifacts/analysis.md")
    assert analysis.status_code == 200
    assert "# Stage Log Analysis" in analysis.text

    metadata = client.get(f"/api/week4/day19/runs/{run_id}/artifacts/metadata.json")
    assert metadata.status_code == 200
    assert json.loads(metadata.text)["run_id"] == run_id

    raw = client.get(f"/api/week4/day19/runs/{run_id}/artifacts/raw.jsonl")
    assert raw.status_code == 200

    # Unknown artifact and unknown run are rejected.
    assert (
        client.get(f"/api/week4/day19/runs/{run_id}/artifacts/secret.txt").status_code
        == 404
    )
    assert (
        client.get(
            "/api/week4/day19/runs/run_missing/artifacts/analysis.md"
        ).status_code
        == 404
    )


def test_day19_pipeline_validation(client) -> None:
    assert (
        client.post(
            "/api/week4/day19/pipeline",
            json={"service": "", "level": "ERROR"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/week4/day19/pipeline",
            json={"service": "application", "limit": 1000},
        ).status_code
        == 422
    )


# ----------------------------------------------------------------------
# 13. UI
# ----------------------------------------------------------------------
def test_week4_day19_enabled_and_day20_disabled(client) -> None:
    import re

    html = client.get("/").text
    assert "Day 19 — MCP Pipeline" in html
    assert 'id="panel-day19"' in html
    assert 'src="/static/js/day19.js"' in html

    day19 = re.search(r'<button[^>]*id="tab-day19"[^>]*>', html)
    assert day19 and "disabled" not in day19.group(0)
    day20 = re.search(r'<button[^>]*id="tab-day20"[^>]*>', html)
    assert day20 and "disabled" in day20.group(0)


def test_day19_form_and_result_elements(client) -> None:
    html = client.get("/").text
    for element in (
        'id="d19-service"',
        'id="d19-level"',
        'id="d19-since"',
        'id="d19-limit"',
        'id="d19-question"',
        'id="d19-run"',
        'id="d19-steps"',
        'id="d19-artifacts"',
        'id="d19-analysis-summary"',
        'id="d19-trace"',
    ):
        assert element in html, element
    # Short test intervals are selectable (1 min is useful for a quick smoke).
    assert '<option value="1">1 min</option>' in html
    assert '<option value="5">5 min</option>' in html


def test_day19_js_dom_references_exist(client) -> None:
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(
        re.findall(r'\$\("([^"]+)"\)', (base / "day19.js").read_text(encoding="utf-8"))
    )
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day19.js references missing ids: {missing}"


def test_day19_frontend_does_not_render_raw_logs_or_hardcode_secrets() -> None:
    from pathlib import Path

    js = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "static"
        / "js"
        / "day19.js"
    ).read_text(encoding="utf-8")
    assert "/api/week4/day19/pipeline" in js
    assert "/api/week4/day19/runs/" in js
    assert "VICTORIA_LOGS" not in js
    # The step renderer groups by tool name; raw log rows are never rendered.
    assert "renderSteps" in js
    assert "renderArtifacts" in js
    assert "search_logs" in js and "analyze_logs" in js and "save_report" in js
    assert "d19-mask" in js


# ----------------------------------------------------------------------
# 14. Opt-in data masking ("Маскировать данные")
# ----------------------------------------------------------------------
MASKED_ROWS = [
    {
        "_time": "2026-09-28T10:00:00Z",
        "_msg": (
            "upstream https://api.internal.example.com/v1 failed "
            "via db.internal:5432 from 10.0.0.7"
        ),
        "service": "payment-service",
        "container_name": "payments-7f9c2a",
        "host": "node-1.internal",
    }
]


async def test_mask_data_masks_identifiers_in_artifacts_and_output(
    settings, fake_service
) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "готово"]
    service, recorder = _service(
        settings, fake_service, rows=MASKED_ROWS, recording=True
    )
    response = await service.run(_request(mask_data=True, service="payment-service"))
    assert response.status == "completed"

    # The masked rows (not the originals) reach analyze_logs and save_report.
    analyze_logs = recorder.calls[1][1]["logs"]
    save_args = recorder.calls[2][1]
    assert analyze_logs[0]["fields"]["service"] == "[MASKED]"
    assert analyze_logs[0]["fields"]["container_name"] == "[MASKED]"
    assert "[URL]" in analyze_logs[0]["message"]
    assert "db.internal:5432" not in analyze_logs[0]["message"]
    assert "10.0.0.7" not in analyze_logs[0]["message"]
    assert analyze_logs == save_args["logs"]
    assert save_args["service"] == "[MASKED]"

    # The response exposes no original identifiers either.
    assert response.search.service == "[MASKED]"

    # Saved artifacts contain no original identifiers.
    run_dir = Path(settings.day19_artifact_root) / response.artifacts.run_id
    blob = "\n".join(
        (run_dir / name).read_text(encoding="utf-8")
        for name in ("raw.jsonl", "analysis.md", "metadata.json")
    )
    for secret in (
        "payment-service",
        "payments-7f9c2a",
        "node-1.internal",
        "api.internal.example.com",
        "db.internal",
        "10.0.0.7",
    ):
        assert secret not in blob, f"{secret} leaked with masking enabled"
    assert "[MASKED]" in blob
    assert "[URL]" in blob


async def test_mask_data_disabled_keeps_original_service(settings, fake_service) -> None:
    fake_service.generate_sequence = [json.dumps(VALID_ANALYSIS), "готово"]
    service, recorder = _service(
        settings, fake_service, rows=MASKED_ROWS, recording=True
    )
    response = await service.run(_request(mask_data=False, service="payment-service"))
    assert response.status == "completed"
    assert response.search.service == "payment-service"
    assert recorder.calls[1][1]["service"] == "payment-service"
    run_dir = Path(settings.day19_artifact_root) / response.artifacts.run_id
    assert "payment-service" in (run_dir / "metadata.json").read_text(encoding="utf-8")


def test_data_masker_masks_urls_hosts_ips_and_emails() -> None:
    from app.services.day19.masking import DataMasker

    masker = DataMasker(["payment-service"])
    text = (
        "see https://api.internal.example.com/v1, db.internal:5432, "
        "10.0.0.7 and ops@example.com for payment-service"
    )
    masked = masker.mask_text(text)
    assert "[URL]" in masked
    assert "db.internal:5432" not in masked
    assert "10.0.0.7" not in masked
    assert "ops@example.com" not in masked
    assert "payment-service" not in masked
    assert "[MASKED]" in masked


def test_data_masker_masks_identifier_field_keys() -> None:
    from app.services.day19.masking import DataMasker

    masker = DataMasker()
    value = {
        "service": "svc-a",
        "container_name": "c-1",
        "host": "node-1",
        "level": "ERROR",
        "message": "ok",
    }
    masked = masker.mask_value(value)
    assert masked["service"] == "[MASKED]"
    assert masked["container_name"] == "[MASKED]"
    assert masked["host"] == "[MASKED]"
    # Non-identifier keys keep their value.
    assert masked["level"] == "ERROR"
    assert masked["message"] == "ok"


def test_data_masker_keeps_timestamps_and_clock_times() -> None:
    from app.services.day19.masking import DataMasker

    masker = DataMasker()
    # Clock times and ISO timestamps must not be mistaken for host:port.
    assert masker.mask_text("2026-09-28T10:00:00Z") == "2026-09-28T10:00:00Z"
    assert masker.mask_text("timeout after 10:00 seconds") == (
        "timeout after 10:00 seconds"
    )
    # Real host:port values are still masked.
    assert masker.mask_text("db.internal:5432") == "[MASKED]"
    assert masker.mask_text("localhost:8080") == "[MASKED]"


def test_day19_mask_checkbox_present(client) -> None:
    html = client.get("/").text
    assert 'id="d19-mask"' in html
    assert "Маскировать данные" in html
