"""Day 19 — the three pipeline tool implementations.

The MCP server in ``app/services/mcp/pipeline/server.py`` is a thin protocol
wrapper around this toolset. Keeping the real work here makes the MCP layer
trivially testable and keeps the VictoriaLogs/LLM/filesystem dependencies in
one place.

The responsibilities are deliberately split exactly as the assignment
requires:

* ``search_logs``  — gets data (and ONLY gets data);
* ``analyze_logs`` — processes data (the single LLM step);
* ``save_report``  — saves data + analysis (deterministic, no LLM).

There is no single "search-analyze-and-save" mega-tool.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from pydantic import ValidationError

from app.config import Settings
from app.schemas.day19 import (
    AnalyzeLogsInput,
    Day19AnalysisSummary,
    Day19Artifacts,
    Day19SearchSummary,
    LogAnalysis,
    SaveReportInput,
    SearchLogsPipelineInput,
)
from app.services.day19.analysis import AnalysisError, build_analysis_messages, run_structured_analysis
from app.services.day19.artifacts import ArtifactError, ArtifactStore, generate_run_id, validate_run_id
from app.services.mcp.victorialogs.client import (
    VictoriaLogsClient,
    VictoriaLogsError,
    format_time,
    normalize_logs,
)
from app.services.mcp.victorialogs.query_builder import QueryBuildError, build_logsql
from app.services.mcp.victorialogs.sanitizer import sanitize_value
from app.services.mcp.victorialogs.schemas import SearchLogsInput

logger = logging.getLogger("app.services.day19.tools")


class Day19Toolset:
    """Implements the three Day 19 tools against the existing components."""

    def __init__(
        self,
        settings: Settings,
        *,
        deepseek: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._settings = settings
        self._deepseek = deepseek
        self._client_factory = client_factory or self._default_client
        self._store = artifact_store or ArtifactStore(settings.day19_artifact_root)

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------
    def _default_client(self) -> VictoriaLogsClient:
        """Reuse the Day 17 client (no duplicated HTTP implementation)."""
        return VictoriaLogsClient(
            base_url=self._settings.victoria_logs_base_url,
            service_field=self._settings.victoria_logs_service_field,
            verify_ssl=self._settings.victoria_logs_verify_ssl,
            ca_bundle=self._settings.victoria_logs_ca_bundle,
            http_timeout_seconds=self._settings.victoria_logs_timeout_seconds,
        )

    def _provider(self) -> Any:
        if self._deepseek is None:
            from app.services.deepseek import DeepSeekService

            self._deepseek = DeepSeekService(self._settings)
        return self._deepseek

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._store

    # ------------------------------------------------------------------
    # Tool 1 — search_logs
    # ------------------------------------------------------------------
    async def search_logs(self, params: SearchLogsPipelineInput) -> dict[str, Any]:
        """Query VictoriaLogs through the existing Day 17 stack and normalize."""
        try:
            validated = SearchLogsInput(
                service=params.service,
                since_minutes=params.since_minutes,
                level=params.level,
                text_contains=params.text_contains,
                limit=params.limit,
            )
        except ValidationError as exc:
            return {"error": str(exc), "error_kind": "query"}

        try:
            query = build_logsql(
                service=validated.service,
                service_field=self._settings.victoria_logs_service_field,
                level=validated.level,
                text_contains=validated.text_contains,
            )
        except QueryBuildError as exc:
            return {"error": str(exc), "error_kind": "query"}

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=validated.since_minutes)

        try:
            rows, malformed = await self._client_factory().search(
                query=query,
                start=start_dt,
                end=end_dt,
                limit=validated.limit,
            )
        except VictoriaLogsError as exc:
            return {"error": exc.message, "error_kind": exc.kind}
        except Exception as exc:  # noqa: BLE001 - controlled tool error
            logger.exception("Day 19 search_logs failed")
            return {
                "error": f"log search failed ({exc.__class__.__name__})",
                "error_kind": "internal",
            }

        logs = normalize_logs(rows, sanitize=sanitize_value)
        return {
            "service": validated.service,
            "since_minutes": validated.since_minutes,
            "start": format_time(start_dt),
            "end": format_time(end_dt),
            "level": validated.level,
            "text_contains": validated.text_contains,
            "query": query,
            "count": len(logs),
            "limit": validated.limit,
            "truncated": len(logs) >= validated.limit,
            "malformed_lines": malformed,
            "logs": logs,
        }

    # ------------------------------------------------------------------
    # Tool 2 — analyze_logs
    # ------------------------------------------------------------------
    async def analyze_logs(self, params: AnalyzeLogsInput) -> dict[str, Any]:
        """Structured LLM analysis over the rows passed by the search step."""
        # An empty search result is NOT sent to the LLM: there is nothing to
        # analyse and asking the model would invite hallucinated findings.
        # Return a deterministic, explicit "no data" analysis instead.
        if not params.logs:
            received = (
                params.logs_received if params.logs_received is not None else 0
            )
            return {
                "service": params.service,
                "summary": (
                    f"За выбранный период логи не найдены (получено {received} "
                    "записей). Анализ не выполнялся."
                ),
                "error_groups": [],
                "error_groups_count": 0,
                "possible_causes": [],
                "notable_patterns": [],
                "recommended_checks": [
                    "Увеличить период поиска.",
                    "Проверить фильтры service/level/text_contains.",
                ],
                "logs_analyzed": 0,
                "logs_received": received,
                "analysis_truncated": False,
            }

        messages, analyzed, truncated = build_analysis_messages(
            service=params.service,
            question=params.question,
            logs=params.logs,
            since_minutes=params.since_minutes,
            level=params.level,
            start=params.start,
            end=params.end,
        )
        try:
            analysis = await run_structured_analysis(
                self._provider(),
                messages=messages,
                model=self._settings.deepseek_model,
            )
        except AnalysisError as exc:
            return {"error": exc.message, "error_kind": exc.kind}

        return {
            "service": params.service,
            "summary": analysis.summary,
            "error_groups": [group.model_dump() for group in analysis.error_groups],
            "error_groups_count": len(analysis.error_groups),
            "possible_causes": list(analysis.possible_causes),
            "notable_patterns": list(analysis.notable_patterns),
            "recommended_checks": list(analysis.recommended_checks),
            "logs_analyzed": analyzed,
            "logs_received": params.logs_received if params.logs_received is not None else len(params.logs),
            "analysis_truncated": truncated,
        }

    # ------------------------------------------------------------------
    # Tool 3 — save_report
    # ------------------------------------------------------------------
    async def save_report(self, params: SaveReportInput) -> dict[str, Any]:
        """Persist the sanitized rows + structured analysis to disk."""
        try:
            validate_run_id(params.run_id)
        except ArtifactError as exc:
            return {"error": exc.message, "error_kind": exc.kind}

        metadata = {
            "pipeline_id": params.pipeline_id,
            "service": params.service,
            "since_minutes": params.since_minutes,
            "level": params.level,
            "text_contains": params.text_contains,
            "start": params.start,
            "end": params.end,
            "logs_received": params.logs_received,
            "logs_analyzed": params.logs_analyzed,
            "malformed_lines": params.malformed_lines,
            "truncated": params.truncated,
            "analysis_truncated": params.analysis_truncated,
            "model": params.model,
            "duration_ms": params.duration_ms,
            "status": "completed",
        }
        try:
            artifacts = self._store.save(
                run_id=params.run_id,
                metadata=metadata,
                logs=params.logs,
                analysis=params.analysis,
            )
        except ArtifactError as exc:
            return {"error": exc.message, "error_kind": exc.kind}
        except OSError as exc:
            logger.exception("Day 19 save_report failed")
            return {
                "error": f"could not write artifacts ({exc.__class__.__name__})",
                "error_kind": "artifact",
            }
        return artifacts

    # ------------------------------------------------------------------
    # Helpers used by the orchestrator
    # ------------------------------------------------------------------
    @staticmethod
    def new_run_id(now: datetime | None = None) -> str:
        return generate_run_id(now)

    @staticmethod
    def search_summary(payload: dict[str, Any]) -> Day19SearchSummary:
        return Day19SearchSummary(
            service=payload.get("service", ""),
            since_minutes=int(payload.get("since_minutes") or 0),
            level=payload.get("level"),
            count=int(payload.get("count") or 0),
            limit=int(payload.get("limit") or 0),
            truncated=bool(payload.get("truncated")),
            malformed_lines=int(payload.get("malformed_lines") or 0),
            start=payload.get("start"),
            end=payload.get("end"),
        )

    @staticmethod
    def analysis_summary(payload: dict[str, Any]) -> Day19AnalysisSummary:
        return Day19AnalysisSummary(
            summary=str(payload.get("summary") or ""),
            error_groups_count=int(payload.get("error_groups_count") or 0),
            error_groups=payload.get("error_groups") or [],
            possible_causes=payload.get("possible_causes") or [],
            notable_patterns=payload.get("notable_patterns") or [],
            recommended_checks=payload.get("recommended_checks") or [],
            logs_analyzed=int(payload.get("logs_analyzed") or 0),
            analysis_truncated=bool(payload.get("analysis_truncated")),
        )

    @staticmethod
    def artifacts_from_payload(payload: dict[str, Any]) -> Day19Artifacts:
        return Day19Artifacts(
            run_id=payload["run_id"],
            root=payload["root"],
            raw=payload.get("raw", "raw.jsonl"),
            analysis=payload.get("analysis", "analysis.md"),
            metadata=payload.get("metadata", "metadata.json"),
        )

    @staticmethod
    def analysis_model(payload: dict[str, Any]) -> LogAnalysis:
        return LogAnalysis(**payload)


__all__ = ["Day19Toolset"]
