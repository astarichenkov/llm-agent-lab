"""FastAPI application factory and entry point."""
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
from app.config import Settings, get_settings

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

    app = FastAPI(
        title=resolved.app_name,
        description=(
            "Educational web application demonstrating REST API integration "
            "with cloud LLM APIs and persistent, multi-agent dialog context."
        ),
        version="1.0.0",
    )
    app.state.settings = resolved
    app.state.agent_manager = AgentManager.for_settings(resolved)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)

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
