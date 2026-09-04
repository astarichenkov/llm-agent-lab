"""HTTP routes: homepage, health check, chat API and comparison API."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import Settings, get_settings
from app.schemas.chat import ChatRequest, ChatResponse, ErrorResponse
from app.schemas.compare import CompareRequest, CompareResponse
from app.schemas.reasoning import ReasoningRequest, ReasoningResponse
from app.schemas.temperature import TemperatureRequest, TemperatureResponse
from app.schemas.day5 import OpenRouterRunRequest, OpenRouterRunResponse
from app.services.openrouter_service import OpenRouterError, OpenRouterService
from app.or_models import models_for_ui, DEFAULT_MODELS
from app.services.deepseek import DeepSeekError, DeepSeekService

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "app" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Documented error responses shared by the provider-backed endpoints.
_PROVIDER_ERROR_RESPONSES = {
    401: {"model": ErrorResponse, "description": "DeepSeek authentication failure"},
    422: {"model": ErrorResponse, "description": "Invalid input"},
    429: {"model": ErrorResponse, "description": "DeepSeek rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Unexpected server error"},
    502: {"model": ErrorResponse, "description": "DeepSeek network / API error"},
    504: {"model": ErrorResponse, "description": "DeepSeek timeout"},
}


def get_deepseek_service(
    settings: Settings = Depends(get_settings),
) -> DeepSeekService:
    """Dependency factory. Tests override this to inject a mock service."""
    return DeepSeekService(settings)


def get_openrouter_service(
    settings: Settings = Depends(get_settings),
) -> OpenRouterService:
    """Day 5 only. Tests override this to inject a mock service."""
    return OpenRouterService(settings)


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the single-page frontend."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "max_message_length": settings.max_message_length,
        },
    )


@router.get("/health")
async def health() -> dict:
    """Liveness probe used by Docker and by load balancers."""
    return {"status": "ok", "application": "deepseek-study-app"}


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def chat(
    payload: ChatRequest,
    service: DeepSeekService = Depends(get_deepseek_service),
) -> ChatResponse:
    """Send a validated question to DeepSeek and return its answer."""
    try:
        return await service.chat(payload.message)
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/api/compare",
    response_model=CompareResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def compare(
    payload: CompareRequest,
    service: DeepSeekService = Depends(get_deepseek_service),
) -> CompareResponse:
    """Send the SAME prompt twice (unrestricted vs controlled) and compare.

    One request to this endpoint triggers exactly two DeepSeek provider
    calls (see ``DeepSeekService.compare``).
    """
    try:
        return await service.compare(payload)
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/api/reasoning",
    response_model=ReasoningResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def reasoning(
    payload: ReasoningRequest,
    service: DeepSeekService = Depends(get_deepseek_service),
) -> ReasoningResponse:
    """Run ONE Day-3 reasoning strategy (no JSON mode). One request = one
    provider call (except Method 3, whose two stages are separate requests).
    """
    try:
        return await service.reasoning(payload)
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

@router.post(
    "/api/temperature",
    response_model=TemperatureResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def temperature_run(
    payload: TemperatureRequest,
    service: DeepSeekService = Depends(get_deepseek_service),
) -> TemperatureResponse:
    """Day 4: send the SAME message with the given temperature (real API
    parameter). One request = one provider call. No JSON mode.
    """
    try:
        return await service.complete_with_temperature(payload)
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/api/openrouter/models")
async def openrouter_models() -> dict:
    """Safe, curated OpenRouter model metadata (no secrets, no client key)."""
    return {"defaults": DEFAULT_MODELS, "models": models_for_ui()}


@router.post(
    "/api/openrouter/run",
    response_model=OpenRouterRunResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def openrouter_run(
    payload: OpenRouterRunRequest,
    service: OpenRouterService = Depends(get_openrouter_service),
) -> OpenRouterRunResponse:
    """Day 5: run ONE model via OpenRouter. One request = one provider call."""
    try:
        return await service.run(payload)
    except OpenRouterError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
