"""HTTP routes: homepage, health check, chat API and comparison API."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import Settings, get_settings
from app.agents.agent import Agent
from app.agents.config import AgentConfig
from app.agents.manager import DEFAULT_AGENT_ID, AgentManager
from app.schemas.agent import (
    AgentChatResponse,
    AgentHistoryResponse,
    AgentInfo,
    AgentListResponse,
    Day6AgentMeta,
    Day6ChatRequest,
    Day6ChatResponse,
)
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


def get_agent_manager(request: Request) -> AgentManager:
    """Return the process-wide AgentManager stored on the FastAPI app.

    Tests override this dependency to inject a manager backed by a temporary
    SQLite database and a fake LLM client.
    """
    return request.app.state.agent_manager


def _agent_info(agent: Agent, settings: Settings) -> AgentInfo:
    """Build the public, secret-free description of an agent."""
    return AgentInfo(
        agent_id=agent.agent_id,
        name=agent.config.name,
        provider=agent.config.provider,
        model=agent.config.resolved_model(settings),
        system_prompt=agent.config.system_prompt,
        temperature=agent.config.temperature,
        max_tokens=agent.config.max_tokens,
        thinking=agent.config.thinking,
    )


# Day 6 — the transient agent. Its config is ALWAYS resolved by the helper
# below, so the metadata endpoint and the chat endpoint can never diverge.
DAY6_AGENT_ID = "day6-agent"
DAY6_AGENT_NAME = "Simple Agent (transient)"


def _day6_config(
    manager: AgentManager,
    provider: str | None = None,
    model: str | None = None,
) -> AgentConfig:
    """Resolve the Day 6 agent configuration with optional overrides.

    The default comes from the same ``AgentConfig.default(settings)`` used by
    the persistent agent layer; only the display name is Day 6-specific.
    """
    config = AgentConfig.default(manager.settings).model_copy(
        update={"name": DAY6_AGENT_NAME}
    )
    if provider is not None:
        # Reset the model so the provider's own default is used unless the
        # caller explicitly supplied one.
        config = config.model_copy(update={"provider": provider, "model": None})
    if model is not None:
        config = config.model_copy(update={"model": model})
    return config


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
    return {"status": "ok", "application": "llm-agent-lab"}


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def chat(
    payload: ChatRequest,
    manager: AgentManager = Depends(get_agent_manager),
) -> ChatResponse:
    """Day 7: the existing chat now runs through the persistent ``default``
    Agent, which restores the previous dialog from SQLite before answering."""
    agent = await manager.get_or_create(DEFAULT_AGENT_ID)
    try:
        result = await agent.chat(payload.message)
    except (DeepSeekError, OpenRouterError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return ChatResponse(answer=result.answer)


@router.get("/api/chat/history", response_model=AgentHistoryResponse)
async def chat_history(
    manager: AgentManager = Depends(get_agent_manager),
) -> AgentHistoryResponse:
    """Return the persisted dialog of the default agent (used by the UI)."""
    agent = await manager.get_or_create(DEFAULT_AGENT_ID)
    messages = await agent.get_history()
    return AgentHistoryResponse(
        agent=_agent_info(agent, manager.settings),
        messages=messages,
        count=len(messages),
    )


@router.delete("/api/chat/history")
async def clear_chat_history(
    manager: AgentManager = Depends(get_agent_manager),
) -> dict:
    """Clear the default agent's dialog (the agent itself is kept)."""
    agent = await manager.get_or_create(DEFAULT_AGENT_ID)
    await agent.clear_history()
    return {"status": "cleared", "agent_id": agent.agent_id}


@router.get("/api/agents", response_model=AgentListResponse)
async def list_agents(
    manager: AgentManager = Depends(get_agent_manager),
) -> AgentListResponse:
    """List every agent known to storage or to the current process."""
    agent_ids = await manager.list_agent_ids()
    infos = [
        _agent_info(await manager.get_or_create(agent_id), manager.settings)
        for agent_id in agent_ids
    ]
    return AgentListResponse(agents=infos)


@router.get("/api/agents/{agent_id}/history", response_model=AgentHistoryResponse)
async def agent_history(
    agent_id: str,
    manager: AgentManager = Depends(get_agent_manager),
) -> AgentHistoryResponse:
    """Return one agent's persisted dialog (multi-agent support)."""
    agent = await manager.get_or_create(agent_id)
    messages = await agent.get_history()
    return AgentHistoryResponse(
        agent=_agent_info(agent, manager.settings),
        messages=messages,
        count=len(messages),
    )


@router.delete("/api/agents/{agent_id}/history")
async def clear_agent_history(
    agent_id: str,
    manager: AgentManager = Depends(get_agent_manager),
) -> dict:
    """Clear one agent's dialog without touching other agents."""
    agent = await manager.get_or_create(agent_id)
    await agent.clear_history()
    return {"status": "cleared", "agent_id": agent.agent_id}


@router.post(
    "/api/agents/{agent_id}/chat",
    response_model=AgentChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def agent_chat(
    agent_id: str,
    payload: ChatRequest,
    manager: AgentManager = Depends(get_agent_manager),
) -> AgentChatResponse:
    """Chat with an explicitly named agent (the generic, multi-agent API)."""
    agent = await manager.get_or_create(agent_id)
    try:
        result = await agent.chat(payload.message)
    except (DeepSeekError, OpenRouterError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return AgentChatResponse(**result.model_dump())


@router.get("/api/day6/agent", response_model=Day6AgentMeta)
async def day6_agent_meta(
    manager: AgentManager = Depends(get_agent_manager),
) -> Day6AgentMeta:
    """Day 6: metadata for the default transient agent.

    Lets the UI show Agent/Provider/Model immediately on page load, before the
    user sends anything. Values come from the SAME ``_day6_config`` helper used
    by ``POST /api/day6/agent/chat``.
    """
    config = _day6_config(manager)
    return Day6AgentMeta(
        agent_id=DAY6_AGENT_ID,
        name=config.name,
        provider=config.provider,
        model=config.resolved_model(manager.settings),
        max_tokens=config.max_tokens,
        thinking=config.thinking,
        stateless=True,
    )


@router.post(
    "/api/day6/agent/chat",
    response_model=Day6ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day6_agent_chat(
    payload: Day6ChatRequest,
    manager: AgentManager = Depends(get_agent_manager),
) -> Day6ChatResponse:
    """Day 6: one stateless interaction with the Agent abstraction.

    A FRESH transient Agent is created for every request (in-memory context),
    so the previous conversation is never part of the LLM payload. The route
    stays thin: it only assembles the config and delegates to ``agent.chat``.
    """
    config = _day6_config(manager, payload.provider, payload.model)

    agent = manager.create_transient(config=config, agent_id=DAY6_AGENT_ID)
    try:
        result = await agent.chat(payload.message)
    except (DeepSeekError, OpenRouterError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return Day6ChatResponse(
        agent_id=result.agent_id,
        answer=result.answer,
        provider=result.provider,
        model=result.model,
        finish_reason=result.finish_reason,
        usage=result.usage,
        stateless=True,
    )


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
