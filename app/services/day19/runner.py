"""Day 19 — the pipeline orchestrator.

Chosen strategy: **deterministic backend orchestration (Variant B)** with the
LLM participating inside ``analyze_logs`` and in the final answer.

Why this variant and not fully LLM-driven tool selection?

* The assignment's primary requirement is that the result of each tool is
  *really* passed to the next one and that the order is exactly
  ``search -> analyze -> save``. Large sanitized log arrays cannot be reliably
  round-tripped through LLM tool-call arguments without either dropping data
  or blowing up the context window.
* Reliability matters more than "magic": a deterministic chain makes the data
  handoff byte-exact, the failure short-circuit explicit, and the trace
  trustworthy.
* The LLM is still genuinely used: ``analyze_logs`` performs the structured
  analysis (with the user's question), and a final compact LLM call produces
  the natural-language answer. If that final call fails, a deterministic
  answer is used instead.

The chain is executed through REAL MCP ``tools/call`` requests (in-memory
transport of the dedicated Pipeline MCP server) — not by calling the toolset
functions directly.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from app.config import Settings
from app.schemas.day19 import (
    PIPELINE_STATUS_COMPLETED,
    PIPELINE_STATUS_FAILED,
    TRACE_STATUS_ERROR,
    TRACE_STATUS_OK,
    TRACE_STATUS_SKIPPED,
    AnalyzeLogsInput,
    Day19PipelineRequest,
    Day19PipelineResponse,
    SaveReportInput,
    SearchLogsPipelineInput,
)
from app.services.day19.artifacts import ArtifactError, ArtifactStore, generate_run_id
from app.services.day19.masking import DataMasker
from app.services.day19.tools import Day19Toolset
from app.services.deepseek import DeepSeekError
from app.services.mcp.pipeline.client import PipelineMCPClient

logger = logging.getLogger("app.services.day19.runner")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Day19PipelineService:
    """Runs the Day 19 pipeline: user request -> search -> analyze -> save."""

    def __init__(
        self,
        settings: Settings,
        *,
        toolset: Day19Toolset | None = None,
        mcp_client: PipelineMCPClient | None = None,
        deepseek: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        artifact_store: ArtifactStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._deepseek = deepseek
        self._toolset = toolset or Day19Toolset(
            settings,
            deepseek=deepseek,
            client_factory=client_factory,
            artifact_store=artifact_store,
        )
        self._mcp = mcp_client or PipelineMCPClient(self._toolset)
        self._now = clock or _utc_now

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def toolset(self) -> Day19Toolset:
        return self._toolset

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._toolset.artifact_store

    def read_artifact(self, run_id: str, filename: str) -> str:
        """Read a known Day 19 artifact (safe: root + filename enforced)."""
        return self._toolset.artifact_store.read(run_id, filename)

    def _provider(self) -> Any:
        if self._deepseek is None:
            from app.services.deepseek import DeepSeekService

            self._deepseek = DeepSeekService(self._settings)
        return self._deepseek

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    async def run(self, request: Day19PipelineRequest) -> Day19PipelineResponse:
        """Execute the full pipeline and return a traceable result."""
        trace: list[Any] = []
        started_at = self._now()
        pipeline_id = generate_run_id(started_at)
        # Opt-in masking layer. It starts with the request's service name and is
        # extended with service/container names discovered in the searched
        # rows before anything is sent to the LLM or written to disk.
        masker = DataMasker([request.service]) if request.mask_data else None

        def finalize(
            response: Day19PipelineResponse,
        ) -> Day19PipelineResponse:
            if masker is None:
                return response
            return Day19PipelineResponse(**masker.mask_value(response.model_dump()))

        def add(step: str, status: str, message: str, **details: Any) -> None:
            from app.schemas.day19 import Day19TraceStep

            trace.append(
                Day19TraceStep(
                    step=step, status=status, message=message, details=details
                )
            )

        def fail(message: str) -> Day19PipelineResponse:
            add("pipeline_failed", TRACE_STATUS_ERROR, message)
            return finalize(
                Day19PipelineResponse(
                    pipeline_id=pipeline_id,
                    status=PIPELINE_STATUS_FAILED,
                    answer="",
                    search=search_summary,
                    analysis=analysis_summary,
                    artifacts=artifacts,
                    trace=trace,
                    error=message,
                )
            )

        search_summary = None
        analysis_summary = None
        artifacts = None

        add(
            "pipeline_started",
            TRACE_STATUS_OK,
            f"Pipeline {pipeline_id} started",
            pipeline_id=pipeline_id,
            service=request.service,
        )

        # 0) Real MCP discovery of the three pipeline tools.
        status = await self._discover_tools(add)
        if status is None or not status.connected:
            message = "Pipeline MCP server is not available."
            if status is not None and status.error:
                message = status.error
            return fail(message)
        allowed = {tool.name for tool in status.tools}

        # 1) search_logs — gets the data.
        add(
            "search_logs",
            "started",
            "Calling MCP tool search_logs",
            arguments={
                "service": request.service,
                "since_minutes": request.since_minutes,
                "level": request.level,
                "limit": request.limit,
            },
        )
        search_result = await self._mcp.call_tool(
            "search_logs",
            {
                "service": request.service,
                "since_minutes": request.since_minutes,
                "level": request.level,
                "text_contains": request.text_contains,
                "limit": request.limit,
            },
        )
        search_payload = search_result.as_payload()
        search_error = self._tool_error(search_result, search_payload)
        if search_error:
            add("search_logs", TRACE_STATUS_ERROR, search_error)
            add(
                "analyze_logs",
                TRACE_STATUS_SKIPPED,
                "analyze_logs skipped because search_logs failed",
            )
            add(
                "save_report",
                TRACE_STATUS_SKIPPED,
                "save_report skipped because search_logs failed",
            )
            return fail(search_error)

        search_summary = self._toolset.search_summary(search_payload)
        logs = list(search_payload.get("logs") or [])

        # Build the masked view of the data if the user opted in. The masked
        # rows are what actually flow to analyze_logs / save_report, so the
        # LLM and the saved artifacts never see the original identifiers.
        if masker is not None:
            masker.add_log_literals(logs)
            safe_logs = masker.mask_logs(logs)
            safe_service = masker.mask_text(request.service)
            safe_question = masker.mask_text(request.question)
            safe_text_contains = masker.mask_text(request.text_contains)
        else:
            safe_logs = logs
            safe_service = request.service
            safe_question = request.question
            safe_text_contains = request.text_contains
        add(
            "search_logs",
            TRACE_STATUS_OK,
            f"search_logs returned {search_summary.count} logs",
            logs_received=search_summary.count,
            truncated=search_summary.truncated,
        )

        # Handoff #1: the ACTUAL sanitized (and optionally masked) rows go to
        # analyze_logs.
        add(
            "handoff_search_logs_to_analyze_logs",
            TRACE_STATUS_OK,
            f"{len(safe_logs)} logs passed to analyze_logs",
            logs_passed=len(safe_logs),
            masked=request.mask_data,
        )

        # 2) analyze_logs — processes the data (single LLM step).
        add(
            "analyze_logs",
            "started",
            "Calling MCP tool analyze_logs",
            logs_passed=len(safe_logs),
        )
        analyze_result = await self._mcp.call_tool(
            "analyze_logs",
            {
                "service": safe_service,
                "question": safe_question,
                "logs": safe_logs,
                "since_minutes": request.since_minutes,
                "level": search_summary.level,
                "start": search_summary.start,
                "end": search_summary.end,
                "logs_received": search_summary.count,
            },
        )
        analyze_payload = analyze_result.as_payload()
        analyze_error = self._tool_error(analyze_result, analyze_payload)
        if analyze_error:
            add("analyze_logs", TRACE_STATUS_ERROR, analyze_error)
            add(
                "save_report",
                TRACE_STATUS_SKIPPED,
                "save_report skipped because analyze_logs failed",
            )
            return fail(analyze_error)

        analysis_summary = self._toolset.analysis_summary(analyze_payload)
        analysis_model = self._toolset.analysis_model(analyze_payload)
        add(
            "analyze_logs",
            TRACE_STATUS_OK,
            f"analyze_logs found {analysis_summary.error_groups_count} error groups",
            logs_analyzed=analysis_summary.logs_analyzed,
            error_groups=analysis_summary.error_groups_count,
            analysis_truncated=analysis_summary.analysis_truncated,
        )

        # Handoff #2: the ACTUAL structured analysis goes to save_report.
        add(
            "handoff_analyze_logs_to_save_report",
            TRACE_STATUS_OK,
            (
                f"{analysis_summary.error_groups_count} error groups + "
                "summary passed to save_report"
            ),
            error_groups=analysis_summary.error_groups_count,
            has_summary=bool(analysis_summary.summary),
        )

        # 3) save_report — deterministic persistence (no LLM).
        run_id = self._toolset.new_run_id(self._now())
        duration_ms = max(
            0, int((self._now() - started_at).total_seconds() * 1000)
        )
        add("save_report", "started", "Calling MCP tool save_report", run_id=run_id)
        save_result = await self._mcp.call_tool(
            "save_report",
            {
                "run_id": run_id,
                "pipeline_id": pipeline_id,
                "service": safe_service,
                "since_minutes": request.since_minutes,
                "level": search_summary.level,
                "text_contains": safe_text_contains,
                "start": search_summary.start,
                "end": search_summary.end,
                "logs_received": search_summary.count,
                "logs_analyzed": analysis_summary.logs_analyzed,
                "malformed_lines": search_summary.malformed_lines,
                "truncated": search_summary.truncated,
                "analysis_truncated": analysis_summary.analysis_truncated,
                "model": self._settings.deepseek_model,
                "duration_ms": duration_ms,
                "logs": safe_logs,
                "analysis": analysis_model.model_dump(),
            },
        )
        save_payload = save_result.as_payload()
        save_error = self._tool_error(save_result, save_payload)
        if save_error:
            add("save_report", TRACE_STATUS_ERROR, save_error)
            return fail(save_error)

        artifacts = self._toolset.artifacts_from_payload(save_payload)
        add(
            "save_report",
            TRACE_STATUS_OK,
            "save_report created 3 artifacts",
            artifacts=[
                artifacts.raw,
                artifacts.analysis,
                artifacts.metadata,
            ],
            run_id=artifacts.run_id,
        )

        # 4) Final natural-language answer (compact input, deterministic fallback).
        answer = await self._final_answer(
            request=request.model_copy(
                update={
                    "service": safe_service,
                    "question": safe_question,
                    "text_contains": safe_text_contains,
                }
            ),
            search=search_summary,
            analysis=analysis_summary,
            artifacts=artifacts,
        )
        if masker is not None:
            answer = masker.mask_text(answer)
        add(
            "pipeline_completed",
            TRACE_STATUS_OK,
            "Pipeline completed successfully",
            error_groups=analysis_summary.error_groups_count,
        )
        return finalize(
            Day19PipelineResponse(
                pipeline_id=pipeline_id,
                status=PIPELINE_STATUS_COMPLETED,
                answer=answer,
                search=search_summary,
                analysis=analysis_summary,
                artifacts=artifacts,
                trace=trace,
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _discover_tools(self, add) -> Any:
        try:
            status = await self._mcp.discover_tools()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Day 19 MCP discovery failed")
            add("discover_tools", TRACE_STATUS_ERROR, str(exc) or "MCP discovery failed")
            return None
        for step in status.trace:
            if step.step in ("initialize", "list_tools"):
                add(step.step, step.status, step.message)
        if status.connected:
            add(
                "discover_tools",
                TRACE_STATUS_OK,
                f"Discovered {status.tools_count} pipeline MCP tools",
            )
        return status

    @staticmethod
    def _tool_error(result: Any, payload: dict[str, Any]) -> str | None:
        payload_error = payload.get("error") if isinstance(payload, dict) else None
        if getattr(result, "is_error", False) or payload_error:
            return str(payload_error or getattr(result, "error", None) or "tool error")
        return None

    async def _final_answer(
        self,
        *,
        request: Day19PipelineRequest,
        search: Any,
        analysis: Any,
        artifacts: Any,
    ) -> str:
        """Ask the LLM for a short answer; fall back deterministically."""
        fallback = self._fallback_answer(
            request=request, search=search, analysis=analysis, artifacts=artifacts
        )
        # No logs found: do not ask the model to "explain" an empty result.
        if search is not None and search.count == 0:
            return fallback
        analysis_path = (
            f"{artifacts.root}/{artifacts.run_id}/{artifacts.analysis}"
            if artifacts
            else "—"
        )
        user_content = (
            "Ты — SRE-ассистент. Ниже — результат уже выполненного анализа логов.\n"
            "Не добавляй факты, которых нет в анализе. Ответь пользователю "
            "кратко на его языке: сколько записей получено, сколько групп "
            "ошибок найдено, краткий вывод и путь к отчёту.\n\n"
            f"Вопрос пользователя: {request.question}\n"
            f"Сервис: {request.service}\n"
            f"Уровень: {request.level or 'ANY'}\n"
            f"Период: последние {request.since_minutes} минут\n"
            f"Получено записей: {search.count if search else 0}\n"
            f"Проанализировано записей: {analysis.logs_analyzed if analysis else 0}\n"
            f"Групп ошибок: {analysis.error_groups_count if analysis else 0}\n"
            f"Краткий вывод: {analysis.summary if analysis else ''}\n"
            f"Отчёт: {analysis_path}\n"
        )
        try:
            content, _finish, _usage = await self._provider().generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "You write concise SRE summaries based ONLY on the "
                            "provided structured analysis."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                model=self._settings.deepseek_model,
                temperature=0.3,
                max_tokens=600,
                thinking=False,
            )
        except DeepSeekError as exc:
            logger.warning("Day 19 final answer LLM call failed: %s", exc.message)
            return fallback
        except Exception as exc:  # noqa: BLE001 - deterministic fallback
            logger.warning(
                "Day 19 final answer LLM call failed (%s)", exc.__class__.__name__
            )
            return fallback
        text = (content or "").strip()
        return text or fallback

    @staticmethod
    def _fallback_answer(
        *,
        request: Day19PipelineRequest,
        search: Any,
        analysis: Any,
        artifacts: Any,
    ) -> str:
        count = search.count if search else 0
        groups = analysis.error_groups_count if analysis else 0
        summary = analysis.summary if analysis and analysis.summary else "—"
        path = (
            f"{artifacts.root}/{artifacts.run_id}/{artifacts.analysis}"
            if artifacts
            else "—"
        )
        level = request.level or "ANY"
        if count == 0:
            return (
                "Анализ завершён: за выбранный период логи не найдены.\n\n"
                f"Получено 0 {level}-записей за последние "
                f"{request.since_minutes} минут.\n"
                "Попробуйте увеличить период или изменить фильтры.\n\n"
                f"Отчёт сохранён:\n{path}"
            )
        return (
            "Анализ завершён.\n\n"
            f"Получено {count} {level}-записей.\n"
            f"Выделено {groups} основных групп ошибок.\n\n"
            f"Краткий вывод:\n{summary}\n\n"
            f"Отчёт сохранён:\n{path}"
        )


__all__ = ["Day19PipelineService"]
