"""FastAPI application factory and entry point."""
from contextlib import asynccontextmanager
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agents.manager import AgentManager
from app.api.routes import router
from app.api.xpander import router as xpander_router
from app.config import Settings, get_settings
from app.services.day8 import Day8TokenService
from app.services.day9 import Day9CompressionService
from app.services.day10 import Day10ContextService
from app.services.day11 import Day11MemoryService
from app.services.day12 import Day12ProfileService
from app.services.day13 import Day13TaskService
from app.services.day14 import Day14InvariantService
from app.services.day15 import Day15LifecycleService
from app.services.day17 import Day17LogsService
from app.services.day18 import Day18MonitoringAgentService, MonitoringService
from app.services.day19 import Day19PipelineService
from app.services.day20 import Day20Service
from app.services.rag.answer_service import RagAnswerService
from app.services.rag.day23_evaluation import Day23EvaluationService
from app.services.rag.day24_evaluation import Day24EvaluationService
from app.services.rag.evaluation import EvaluationService
from app.services.rag.grounding.service import GroundedRagService
from app.services.day25 import Day25ChatService, Day25EvaluationService
from app.services.mcp import MCPClient

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

logger = logging.getLogger("app")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Keep logs at 5 MiB each, 3 rotated backups (simple rotation policy).
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def configure_logging(log_file: str | None = None) -> logging.Logger:
    """Configure the root logger once.

    * a StreamHandler -> stdout (so ``docker compose logs app`` works);
    * a RotatingFileHandler -> the persistent ``log_file`` when it can be
      created/written (falls back to stdout only otherwise — a logging
      failure must never break the application).

    Duplicate handlers are avoided (idempotent when create_app runs tests).
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(stream)

    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            # open the file once to prove it is writable
            with open(path, "a", encoding="utf-8"):
                pass
            file_handler = RotatingFileHandler(
                str(path),
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            root.addHandler(file_handler)
            root.info("Application logging configured: file=%s", os.path.abspath(log_file))
        except Exception as exc:  # pragma: no cover - path/env dependent
            root.warning("Persistent log file unavailable (%s); stdout only", exc)
    return root


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application (used by uvicorn and by tests)."""
    resolved = settings or get_settings()
    configure_logging(resolved.app_log_file)

    # Day 18 services are constructed here (cheap; SQLite is opened lazily).
    # The scheduler itself is started/stopped by the application lifespan so
    # it always matches the FastAPI process lifecycle.
    monitoring_service = MonitoringService(resolved)
    day18_service = Day18MonitoringAgentService(resolved, monitoring_service)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Start the monitoring scheduler and restore active jobs."""
        service: MonitoringService = application.state.monitoring_service
        service.start_scheduler()
        try:
            service.restore_active_jobs()
        except Exception:  # noqa: BLE001 - recovery must not block startup
            logger.exception("Day 18: failed to restore active monitoring jobs")
        try:
            yield
        finally:
            service.shutdown_scheduler()

    app = FastAPI(
        title=resolved.app_name,
        description=(
            "Educational web application demonstrating REST API integration "
            "with cloud LLM APIs and persistent, multi-agent dialog context."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.agent_manager = AgentManager.for_settings(resolved)
    app.state.day8_service = Day8TokenService(resolved)
    app.state.day9_service = Day9CompressionService(resolved)
    app.state.day10_service = Day10ContextService(resolved)
    app.state.day11_service = Day11MemoryService(resolved)
    app.state.day12_service = Day12ProfileService(
        resolved, memory_service=app.state.day11_service
    )
    app.state.day13_service = Day13TaskService(resolved)
    app.state.day14_service = Day14InvariantService(resolved)
    app.state.day15_service = Day15LifecycleService(resolved)
    # Day 16 — local MCP client (stdio). Stateless: every request/refresh
    # spawns a fresh Demo MCP server subprocess and lists its tools.
    app.state.mcp_client = MCPClient()
    # Day 17 — VictoriaLogs MCP tool-calling service. Constructing it does
    # NOT contact VictoriaLogs: the base URL is only required when a chat
    # request is actually made, so a missing VICTORIA_LOGS_BASE_URL never
    # breaks startup or Day 16.
    app.state.day17_service = Day17LogsService(resolved)
    # Day 18 — scheduled monitoring. The service owns the SQLite repository
    # and the scheduler; the agent service exposes the monitoring MCP tools
    # to the LLM only when the Day 18 tab is used.
    app.state.monitoring_service = monitoring_service
    app.state.day18_service = day18_service
    # Day 19 — MCP tool composition pipeline. Constructing the service does
    # NOT contact VictoriaLogs or the LLM: both are reached lazily when a
    # pipeline request is actually made.
    app.state.day19_service = Day19PipelineService(resolved)
    # Day 20 — multi-server MCP orchestration (VictoriaLogs + Gitea +
    # Reports). Constructing the service does NOT contact any server: each
    # registry is built and discovered lazily per request.
    app.state.day20_service = Day20Service(resolved)
    # Day 22 — first RAG request. Constructing these services does NOT open
    # the index or contact Ollama/DeepSeek: the generation provider and the
    # SQLite index are both reached lazily on the first request.
    app.state.day22_service = RagAnswerService(resolved)
    app.state.day22_evaluation_service = EvaluationService(
        resolved, answer_service=app.state.day22_service
    )
    # Day 23 — query rewrite + relevance filtering. Shares the Day 22 answer
    # service (same retriever, same generation model) and reuses the Day 22
    # evaluation dataset; only the manual grades live in a separate file.
    app.state.day23_service = app.state.day22_service
    app.state.day23_evaluation_service = Day23EvaluationService(
        resolved, answer_service=app.state.day22_service
    )
    # Day 24 — grounded RAG (citations + anti-hallucination). Shares the Day 22
    # answer service so it reuses the Day 23 improved retrieval unchanged.
    app.state.day24_service = GroundedRagService(
        resolved, answer_service=app.state.day22_service
    )
    app.state.day24_evaluation_service = Day24EvaluationService(
        resolved,
        answer_service=app.state.day22_service,
        grounded_service=app.state.day24_service,
    )
    # Day 25 — stateful mini-chat with RAG + task memory. It reuses the SAME
    # Day 22 answer service (hence the Day 23 improved retrieval) and the Day
    # 24 grounded pipeline; it only adds sessions, history and task state.
    app.state.day25_service = Day25ChatService(
        resolved, grounded_service=app.state.day24_service
    )
    app.state.day25_evaluation_service = Day25EvaluationService(
        resolved, chat_service=app.state.day25_service
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)
    # Standalone product page + its API facade. Reuses the Day 25 chat
    # service; adds no RAG logic.
    app.include_router(xpander_router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Last-resort handler: log internals, return a safe generic body."""
        logger.exception(
            "Unhandled exception on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again later."},
        )

    return app


app = create_app()
