"""Public ``/xpander`` product page and its thin API facade.

This module does NOT add RAG/business logic. It is a presentation shell around
the existing Day 25 chat service (sessions + task memory + grounded answers).
The API aliases below delegate to the SAME ``Day25ChatService`` instance
already stored on ``app.state`` and reuse the existing dependency override
``get_day25_service`` so tests keep working unchanged.

Why aliases: the user-facing product must not expose internal laboratory
paths (``/api/week5/day25/...``). The network contract is renamed, the logic
is untouched.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.routes import get_day25_service
from app.config import Settings, get_settings
from app.schemas.day25 import (
    ChatSession,
    CreateSessionRequest,
    Day25StatusResponse,
    SendMessageRequest,
    SendMessageResponse,
    SessionDetailResponse,
    SessionListResponse,
    TaskStateResponse,
)
from app.services.day25 import ChatServiceError, Day25ChatService
from app.services.rag.answer_service import RagRetrievalError
from app.services.rag.generation import GenerationError

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "app" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _xpander_http_error(exc: Exception) -> HTTPException:
    """Translate a service error into a browser-safe HTTPException."""
    return HTTPException(
        status_code=getattr(exc, "status_code", 502),
        detail=getattr(exc, "message", str(exc)),
    )


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
@router.get("/xpander", response_class=HTMLResponse, include_in_schema=False)
async def xpander_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the standalone Mitsubishi Xpander Assistant product page."""
    return templates.TemplateResponse(
        request=request,
        name="xpander.html",
        context={
            "product_name": "Mitsubishi Xpander Assistant",
            "settings": settings,
        },
    )


# ----------------------------------------------------------------------
# API facade (reuses the Day 25 chat service unchanged)
# ----------------------------------------------------------------------
@router.get("/api/xpander/status", response_model=Day25StatusResponse)
async def xpander_status(
    service: Day25ChatService = Depends(get_day25_service),
) -> Day25StatusResponse:
    return Day25StatusResponse(**service.status())


@router.get("/api/xpander/sessions", response_model=SessionListResponse)
async def xpander_list_sessions(
    service: Day25ChatService = Depends(get_day25_service),
) -> SessionListResponse:
    return SessionListResponse(sessions=service.list_sessions())


@router.post(
    "/api/xpander/sessions", response_model=ChatSession, status_code=201
)
async def xpander_create_session(
    payload: CreateSessionRequest | None = None,
    service: Day25ChatService = Depends(get_day25_service),
) -> ChatSession:
    return service.create_session(payload)


@router.get(
    "/api/xpander/sessions/{session_id}", response_model=SessionDetailResponse
)
async def xpander_get_session(
    session_id: str,
    service: Day25ChatService = Depends(get_day25_service),
) -> SessionDetailResponse:
    try:
        return service.get_detail(session_id)
    except ChatServiceError as exc:
        raise _xpander_http_error(exc) from exc


@router.delete("/api/xpander/sessions/{session_id}", status_code=204)
async def xpander_delete_session(
    session_id: str,
    service: Day25ChatService = Depends(get_day25_service),
) -> None:
    try:
        service.delete_session(session_id)
    except ChatServiceError as exc:
        raise _xpander_http_error(exc) from exc


@router.get(
    "/api/xpander/sessions/{session_id}/state", response_model=TaskStateResponse
)
async def xpander_get_state(
    session_id: str,
    service: Day25ChatService = Depends(get_day25_service),
) -> TaskStateResponse:
    try:
        state = service.get_state(session_id)
    except ChatServiceError as exc:
        raise _xpander_http_error(exc) from exc
    return TaskStateResponse(
        session_id=session_id,
        state=state,
        updated_at=service.repository.get_state_updated_at(session_id),
    )


@router.post(
    "/api/xpander/sessions/{session_id}/messages",
    response_model=SendMessageResponse,
)
async def xpander_send_message(
    session_id: str,
    payload: SendMessageRequest,
    service: Day25ChatService = Depends(get_day25_service),
) -> SendMessageResponse:
    try:
        return await service.send_message(session_id, payload.content)
    except ChatServiceError as exc:
        raise _xpander_http_error(exc) from exc
    except (GenerationError, RagRetrievalError) as exc:
        raise _xpander_http_error(exc) from exc
