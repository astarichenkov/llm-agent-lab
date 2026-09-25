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
from app.schemas.day8 import (
    Day8ChatRequest,
    Day8ChatResponse,
    Day8EstimateRequest,
    Day8EstimateResponse,
    Day8OverflowRequest,
    Day8OverflowResponse,
)
from app.services.day8 import Day8TokenService
from app.services.day9 import Day9CompressionService
from app.services.day10 import Day10ContextService
from app.services.day11 import Day11MemoryService
from app.services.day12 import Day12ProfileService
from app.services.day13 import Day13TaskService, TaskStateError
from app.services.day14 import Day14InvariantService
from app.services.day15 import Day15Error, Day15LifecycleService
from app.services.mcp import MCPClient
from app.schemas.day13 import (
    Day13ChatRequest,
    Day13ChatResponse,
    Day13StateResponse,
    Day13TaskCreateRequest,
    Day13TransitionRequest,
)
from app.schemas.day14 import (
    Day14ChatRequest,
    Day14ChatResponse,
    Day14StateResponse,
)
from app.schemas.day15 import (
    Day15ChatRequest,
    Day15ChatResponse,
    Day15PlanRequest,
    Day15StateResponse,
    Day15TaskCreateRequest,
    Day15TransitionRequest,
    Day15TransitionResponse,
    Day15ValidationRequest,
)
from app.schemas.day16 import MCPStatusResponse
from app.schemas.day10 import (
    ContextStrategyName,
    Day10BranchActivateRequest,
    Day10BranchActivateResponse,
    Day10BranchCreateRequest,
    Day10BranchCreateResponse,
    Day10ChatRequest,
    Day10ChatResponse,
    Day10DemoRunResponse,
    Day10EvaluateRequest,
    Day10EvaluateResponse,
    Day10ScenarioResponse,
    Day10StateResponse,
)
from app.schemas.day9 import (
    Day9ChatRequest,
    Day9ChatResponse,
    Day9DiagnosticsResponse,
    Day9SeedRequest,
    Day9SeedResponse,
)
from app.schemas.day11 import (
    Day11ChatRequest,
    Day11ChatResponse,
    Day11EvaluateRequest,
    Day11EvaluateResponse,
    Day11ScenarioResponse,
    Day11StateResponse,
)
from app.schemas.day12 import (
    Day12ChatRequest,
    Day12ChatResponse,
    Day12CompareRequest,
    Day12CompareResponse,
    Day12ProfileResponse,
    UserProfile,
)
from app.services.openrouter_service import OpenRouterError, OpenRouterService
from app.or_models import models_for_ui, DEFAULT_MODELS
from app.services.deepseek import DeepSeekError, DeepSeekService

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "app" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Documented error responses shared by the provider-backed endpoints.
_PROVIDER_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Provider rejected the request (e.g. context window exceeded)"},
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


def get_day8_service(request: Request) -> Day8TokenService:
    """Return the process-wide Day 8 token service (isolated in-memory dialog).

    Stored on the app so its synthetic experiment dialog survives between
    requests without ever touching the persistent Day 7 storage. Tests
    override this dependency.
    """
    return request.app.state.day8_service


def get_day9_service(request: Request) -> Day9CompressionService:
    """Return the process-wide Day 9 compression service.

    Owns one dialog whose full history, summary and compression metadata are
    kept separately and never touch the Day 7 SQLite storage. Tests override
    this dependency to inject a fake provider.
    """
    return request.app.state.day9_service


def get_day10_service(request: Request) -> Day10ContextService:
    """Return the process-wide Day 10 context-strategy service.

    Owns one isolated in-memory dialog per strategy (sliding window, sticky
    facts, branching). Never touches the Day 7 SQLite storage. Tests override
    this dependency to inject a fake provider.
    """
    return request.app.state.day10_service


def get_day11_service(request: Request) -> Day11MemoryService:
    """Return the process-wide Day 11 memory-layers service.

    Owns the three memory layers (short-term / working / long-term) plus the
    MemoryClassifier and the Context Builder. Short-term/working are session
    scoped; long-term is persisted to a small JSON file. Tests override this
    dependency to inject a fake provider and a temporary memory file.
    """
    return request.app.state.day11_service


def get_day12_service(request: Request) -> Day12ProfileService:
    """Return the process-wide Day 12 personalization service.

    Owns the stored user profile and applies it to every request by delegating
    to the EXISTING Day 11 memory service (so personalization sits on top of
    the memory model). Tests override this dependency to inject a fake provider
    and a temporary profile file.
    """
    return request.app.state.day12_service


def get_day13_service(request: Request) -> Day13TaskService:
    """Return the process-wide Day 13 Task State Machine service.

    Owns the current structured Task State (stage / current_step /
    expected_action / plan / paused) and drives the code-enforced FSM around
    the existing DeepSeek client. State is persisted to a small JSON file so
    it survives separate requests. Tests override this dependency to inject a
    fake provider and a temporary task-state file.
    """
    return request.app.state.day13_service


def get_day14_service(request: Request) -> Day14InvariantService:
    """Return the process-wide Day 14 invariants service.

    Owns the active invariants (persisted separately) and the independent
    conversation history. The service injects the ACTIVE INVARIANTS block into
    every model request and refuses requests that conflict with an invariant.
    Tests override this dependency to inject a fake provider and a temporary
    invariants file.
    """
    return request.app.state.day14_service


def get_day15_service(request: Request) -> Day15LifecycleService:
    """Return the process-wide Day 15 controlled-lifecycle service.

    Owns the current lifecycle state and the transition history. EVERY state
    change goes through the pure LifecycleMachine in code (allowed transitions
    table + guards). Tests override this dependency to inject a fake provider
    and a temporary lifecycle file.
    """
    return request.app.state.day15_service


def get_mcp_client(request: Request) -> MCPClient:
    """Return the process-wide Day 16 MCP client (stdio transport).

    The client owns ALL MCP transport concerns: it spawns the local Demo MCP
    server as a subprocess, initializes the session and calls ``tools/list``.
    Tests override this dependency to inject a client pointed at a healthy or
    deliberately broken server.
    """
    return request.app.state.mcp_client


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


# ----------------------------------------------------------------------
# Day 8 — token accounting (isolated from the persistent Day 7 dialog)
# ----------------------------------------------------------------------
@router.post(
    "/api/day8/estimate",
    response_model=Day8EstimateResponse,
    responses={422: {"model": ErrorResponse, "description": "Invalid input"}},
)
async def day8_estimate(
    payload: Day8EstimateRequest,
    service: Day8TokenService = Depends(get_day8_service),
) -> Day8EstimateResponse:
    """Day 8: LOCAL-only token estimate of a synthetic prompt.

    Performs NO provider call (zero API spend). Used by the UI to preview the
    estimated input tokens and to show "LIMIT EXCEEDED" before an experiment.
    """
    return service.estimate(
        message=payload.message,
        history=payload.history,
        model=payload.model,
        system_prompt=payload.system_prompt,
    )


@router.post(
    "/api/day8/chat",
    response_model=Day8ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day8_chat(
    payload: Day8ChatRequest,
    service: Day8TokenService = Depends(get_day8_service),
) -> Day8ChatResponse:
    """Day 8: one real turn with full token accounting.

    The context is NEVER trimmed here: if the synthetic history exceeds the
    model context window, the provider's real error is surfaced (HTTP 400)
    instead of being hidden.
    """
    try:
        return await service.chat(
            message=payload.message,
            history=payload.history,
            model=payload.model,
            system_prompt=payload.system_prompt,
        )
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/api/day8/overflow",
    response_model=Day8OverflowResponse,
    responses={422: {"model": ErrorResponse, "description": "Invalid input"}},
)
async def day8_overflow(
    payload: Day8OverflowRequest,
    service: Day8TokenService = Depends(get_day8_service),
) -> Day8OverflowResponse:
    """Day 8: build a deterministic oversized context SERVER-SIDE.

    The synthetic history is generated on the backend so the browser never
    uploads megabytes of filler (nginx caps the request body). Makes NO
    provider call - it only estimates. The returned ``request`` is then sent
    to ``/api/day8/chat`` by the UI to execute the real request.
    """
    return service.build_overflow(
        model=payload.model,
        system_prompt=payload.system_prompt,
        overshoot_factor=payload.overshoot_factor,
    )


@router.delete("/api/day8/history")
async def day8_clear(
    service: Day8TokenService = Depends(get_day8_service),
) -> dict:
    """Day 8: clear the isolated in-memory token-experiment dialog."""
    service.reset()
    return {"status": "cleared"}


@router.get("/api/day8/history")
async def day8_history(
    service: Day8TokenService = Depends(get_day8_service),
) -> dict:
    """Day 8: the isolated in-memory dialog (never the Day 7 database)."""
    return {"messages": service.history, "count": len(service.history)}


# ----------------------------------------------------------------------
# Day 9 — context compression (history summarization), isolated dialog
# ----------------------------------------------------------------------
@router.post(
    "/api/day9/chat",
    response_model=Day9ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day9_chat(
    payload: Day9ChatRequest,
    service: Day9CompressionService = Depends(get_day9_service),
) -> Day9ChatResponse:
    """Day 9: one turn, either with the full history or with compression.

    Compression (when a full batch of old messages accumulated) runs BEFORE
    the answer, on the backend. A failed summary build never corrupts the
    dialog and is reported in ``compression_error``.
    """
    try:
        return await service.chat(
            message=payload.message,
            mode=payload.mode,
            recent_messages_limit=payload.recent_messages_limit,
            compression_batch_size=payload.compression_batch_size,
            model=payload.model,
            system_prompt=payload.system_prompt,
        )
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/api/day9/seed",
    response_model=Day9SeedResponse,
    responses={422: {"model": ErrorResponse, "description": "Invalid input"}},
)
async def day9_seed(
    payload: Day9SeedRequest,
    service: Day9CompressionService = Depends(get_day9_service),
) -> Day9SeedResponse:
    """Day 9: install the deterministic demo dialog (no provider call).

    Facts are stated FIRST so they leave the recent window and can only
    survive if compression kept them in the summary.
    """
    return service.seed_demo(
        recent_messages_limit=payload.recent_messages_limit,
        compression_batch_size=payload.compression_batch_size,
    )


@router.get("/api/day9/diagnostics", response_model=Day9DiagnosticsResponse)
async def day9_diagnostics(
    service: Day9CompressionService = Depends(get_day9_service),
) -> Day9DiagnosticsResponse:
    """Day 9: current compression state (full history, summary, savings)."""
    return service.diagnostics()


@router.delete("/api/day9/history")
async def day9_clear(
    service: Day9CompressionService = Depends(get_day9_service),
) -> dict:
    """Day 9: clear history, summary and compression metadata together."""
    service.reset()
    return {"status": "cleared"}


# ----------------------------------------------------------------------
# Day 10 — context-management strategies (sliding window / sticky facts /
# branching). NO history summarisation: strategies only choose what context
# is sent to the model.
# ----------------------------------------------------------------------
@router.get("/api/day10/scenario", response_model=Day10ScenarioResponse)
async def day10_scenario() -> Day10ScenarioResponse:
    """Return the shared deterministic demo scenario (no provider call)."""
    from app.services.day10.service import scenario_definition

    return scenario_definition()


@router.get("/api/day10/state", response_model=Day10StateResponse)
async def day10_state(
    strategy: ContextStrategyName = "sliding_window",
    window_size: int | None = None,
    recent_messages_limit: int | None = None,
    service: Day10ContextService = Depends(get_day10_service),
) -> Day10StateResponse:
    """Current state of one strategy (facts / branches / window)."""
    return service.state(
        strategy=strategy,
        window_size=window_size,
        recent_messages_limit=recent_messages_limit,
    )


@router.delete("/api/day10/state")
async def day10_reset(
    strategy: ContextStrategyName = "sliding_window",
    service: Day10ContextService = Depends(get_day10_service),
) -> dict:
    """Reset ONE strategy's dialog (other strategies are untouched)."""
    service.reset(strategy)
    return {"status": "cleared", "strategy": strategy}


@router.post(
    "/api/day10/chat",
    response_model=Day10ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day10_chat(
    payload: Day10ChatRequest,
    service: Day10ContextService = Depends(get_day10_service),
) -> Day10ChatResponse:
    """One Day 10 turn under the selected context strategy."""
    try:
        return await service.chat(
            strategy=payload.strategy,
            message=payload.message,
            window_size=payload.window_size,
            recent_messages_limit=payload.recent_messages_limit,
            update_facts=payload.update_facts,
            branch_id=payload.branch_id,
            model=payload.model,
            system_prompt=payload.system_prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/api/day10/branches",
    response_model=Day10BranchCreateResponse,
    responses={400: {"model": ErrorResponse, "description": "Invalid branch"}},
)
async def day10_create_branch(
    payload: Day10BranchCreateRequest,
    service: Day10ContextService = Depends(get_day10_service),
) -> Day10BranchCreateResponse:
    """Create an independent branch from a checkpoint of an existing branch."""
    try:
        return service.create_branch(
            name=payload.name,
            parent_branch_id=payload.parent_branch_id,
            checkpoint_message_id=payload.checkpoint_message_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/api/day10/branches/activate",
    response_model=Day10BranchActivateResponse,
    responses={400: {"model": ErrorResponse, "description": "Unknown branch"}},
)
async def day10_activate_branch(
    payload: Day10BranchActivateRequest,
    service: Day10ContextService = Depends(get_day10_service),
) -> Day10BranchActivateResponse:
    """Switch the active branch (does not touch any branch's history)."""
    try:
        return service.activate_branch(payload.branch_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/day10/branches/seed", response_model=Day10StateResponse)
async def day10_seed_branches(
    service: Day10ContextService = Depends(get_day10_service),
) -> Day10StateResponse:
    """Install the deterministic branching demo (no provider call)."""
    return service.seed_branching_demo()


@router.post("/api/day10/evaluate", response_model=Day10EvaluateResponse)
async def day10_evaluate(
    payload: Day10EvaluateRequest,
    service: Day10ContextService = Depends(get_day10_service),
) -> Day10EvaluateResponse:
    """Demo-only keyword metric: which expected facts survived in the answer."""
    return service.evaluate(payload.answer, payload.expected)


@router.post(
    "/api/day10/demo/run",
    response_model=Day10DemoRunResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day10_demo_run(
    strategy: ContextStrategyName = "sliding_window",
    window_size: int | None = None,
    recent_messages_limit: int | None = None,
    update_facts: bool = True,
    service: Day10ContextService = Depends(get_day10_service),
) -> Day10DemoRunResponse:
    """Run the whole 15-message scenario server-side for one strategy.

    The web UI normally drives this scenario message-by-message to show
    progress; this batch endpoint is used by tests and by automation.
    """
    try:
        return await service.run_scenario(
            strategy=strategy,
            window_size=window_size,
            recent_messages_limit=recent_messages_limit,
            update_facts=update_facts,
        )
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


# ----------------------------------------------------------------------
# Day 11 — agent memory layers (short-term / working / long-term).
# Memory layers decide WHAT to store; a context strategy decides what part
# of the stored context reaches the model. The chat endpoint runs BOTH:
# MemoryClassifier -> storage -> Context Builder -> Sliding Window -> DeepSeek.
# ----------------------------------------------------------------------
@router.get("/api/day11/state", response_model=Day11StateResponse)
async def day11_state(
    service: Day11MemoryService = Depends(get_day11_service),
) -> Day11StateResponse:
    """Current memory state of the active session (no provider call)."""
    return Day11StateResponse(model=service.model, memory=service.state())


@router.post("/api/day11/session", response_model=Day11StateResponse)
async def day11_new_session(
    service: Day11MemoryService = Depends(get_day11_service),
) -> Day11StateResponse:
    """Start a NEW session: short-term and working are empty.

    Long-term memory is shared and survives the new session.
    """
    return Day11StateResponse(model=service.model, memory=service.new_session())


@router.delete("/api/day11/memory/short-term", response_model=Day11StateResponse)
async def day11_clear_short_term(
    service: Day11MemoryService = Depends(get_day11_service),
) -> Day11StateResponse:
    """Clear ONLY short-term memory (working + long-term untouched)."""
    return Day11StateResponse(model=service.model, memory=service.reset_short_term())


@router.delete("/api/day11/memory/working", response_model=Day11StateResponse)
async def day11_clear_working(
    service: Day11MemoryService = Depends(get_day11_service),
) -> Day11StateResponse:
    """Clear ONLY working memory (short-term + long-term untouched)."""
    return Day11StateResponse(model=service.model, memory=service.reset_working())


@router.delete("/api/day11/memory/long-term", response_model=Day11StateResponse)
async def day11_clear_long_term(
    service: Day11MemoryService = Depends(get_day11_service),
) -> Day11StateResponse:
    """Clear ONLY long-term memory (short-term + working untouched)."""
    return Day11StateResponse(model=service.model, memory=service.reset_long_term())


@router.post("/api/day11/memory/seed", response_model=Day11StateResponse)
async def day11_seed_memory(
    service: Day11MemoryService = Depends(get_day11_service),
) -> Day11StateResponse:
    """Deterministically seed the demo memory (no provider call).

    Short-term gets the scenario history; working + long-term get the durable
    demo facts. Used to demonstrate sliding-window loss without spending API
    calls on the opening messages.
    """
    return Day11StateResponse(model=service.model, memory=service.seed_demo())


@router.post(
    "/api/day11/chat",
    response_model=Day11ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day11_chat(
    payload: Day11ChatRequest,
    service: Day11MemoryService = Depends(get_day11_service),
) -> Day11ChatResponse:
    """One Day 11 turn: classify -> store -> build context -> answer."""
    try:
        return await service.chat(
            message=payload.message,
            session_id=payload.session_id,
            strategy=payload.strategy,
            window_size=payload.window_size,
            use_memory=payload.use_memory,
            classify=payload.classify,
            include_long_term=payload.include_long_term,
            include_working=payload.include_working,
            model=payload.model,
            system_prompt=payload.system_prompt,
        )
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/api/day11/scenario", response_model=Day11ScenarioResponse)
async def day11_scenario(
    service: Day11MemoryService = Depends(get_day11_service),
) -> Day11ScenarioResponse:
    """The deterministic demo scenario definition (no provider call)."""
    return service.scenario()


@router.post("/api/day11/evaluate", response_model=Day11EvaluateResponse)
async def day11_evaluate(
    payload: Day11EvaluateRequest,
    service: Day11MemoryService = Depends(get_day11_service),
) -> Day11EvaluateResponse:
    """Demo-only keyword metric for the scenario answer."""
    return service.evaluate(payload.answer, payload.expected)


# ----------------------------------------------------------------------
# Day 12 — user profile / personalization.
# The profile is stored separately from the dialog and from memory. The
# backend builds the USER PROFILE instructions and injects them into every
# request, so the user never repeats preferences in the message.
# ----------------------------------------------------------------------
@router.get("/api/day12/profile", response_model=Day12ProfileResponse)
async def day12_get_profile(
    service: Day12ProfileService = Depends(get_day12_service),
) -> Day12ProfileResponse:
    """Current active profile, its instructions and the built-in presets."""
    return service.response()


@router.put(
    "/api/day12/profile",
    response_model=Day12ProfileResponse,
    responses={422: {"model": ErrorResponse, "description": "Invalid profile"}},
)
async def day12_put_profile(
    payload: UserProfile,
    service: Day12ProfileService = Depends(get_day12_service),
) -> Day12ProfileResponse:
    """Validate, persist and activate a structured user profile."""
    return service.update_profile(payload)


@router.post(
    "/api/day12/profile/preset/{preset_id}",
    response_model=Day12ProfileResponse,
    responses={400: {"model": ErrorResponse, "description": "Unknown preset"}},
)
async def day12_apply_preset(
    preset_id: str,
    service: Day12ProfileService = Depends(get_day12_service),
) -> Day12ProfileResponse:
    """Apply one built-in demo profile (concise / detailed / technical)."""
    try:
        return service.apply_preset(preset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/api/day12/chat",
    response_model=Day12ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day12_chat(
    payload: Day12ChatRequest,
    service: Day12ProfileService = Depends(get_day12_service),
) -> Day12ChatResponse:
    """One personalized turn: profile + memory + answer.

    The message is forwarded unchanged; the saved profile is applied on the
    backend automatically.
    """
    try:
        return await service.chat(
            message=payload.message,
            session_id=payload.session_id,
            apply_profile=payload.apply_profile,
            use_memory=payload.use_memory,
            classify=payload.classify,
            strategy=payload.strategy,
            window_size=payload.window_size,
            model=payload.model,
            system_prompt=payload.system_prompt,
        )
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/api/day12/compare",
    response_model=Day12CompareResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day12_compare(
    payload: Day12CompareRequest,
    service: Day12ProfileService = Depends(get_day12_service),
) -> Day12CompareResponse:
    """Answer the SAME question under several profiles (demo comparison).

    Each profile runs in an isolated temporary session with memory disabled,
    so only the USER PROFILE block differs. The user's active session and
    memory are left untouched. One request triggers one provider call per
    selected preset.
    """
    try:
        return await service.compare(
            message=payload.message,
            preset_ids=payload.preset_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


# ----------------------------------------------------------------------
# Day 13 — Task State Machine (stage / current_step / expected_action).
# The structured Task State lives on the backend, is persisted separately from
# the chat history and is injected into every model request. Stage changes go
# through the code-enforced ALLOWED_TRANSITIONS table, never through the LLM.
# ----------------------------------------------------------------------
def _raise_task_state_error(exc: TaskStateError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/api/day13/state", response_model=Day13StateResponse)
async def day13_state(
    service: Day13TaskService = Depends(get_day13_service),
) -> Day13StateResponse:
    """Current Task State plus the transitions the FSM allows right now."""
    return service.state_response()


@router.post(
    "/api/day13/task",
    response_model=Day13StateResponse,
    responses={422: {"model": ErrorResponse, "description": "Invalid goal"}},
)
async def day13_create_task(
    payload: Day13TaskCreateRequest,
    service: Day13TaskService = Depends(get_day13_service),
) -> Day13StateResponse:
    """Start a NEW task in the ``planning`` stage (no provider call)."""
    try:
        service.create_task(payload.goal)
    except TaskStateError as exc:
        _raise_task_state_error(exc)
    return service.state_response()


@router.post(
    "/api/day13/chat",
    response_model=Day13ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day13_chat(
    payload: Day13ChatRequest,
    service: Day13TaskService = Depends(get_day13_service),
) -> Day13ChatResponse:
    """One turn on the current task, driven by the formal Task State.

    In ``planning`` the model produces the plan and CODE moves to
    ``execution``. In ``execution`` the current step is performed and the code
    advances ``current_step`` (or moves to ``validation``). In ``validation``
    the code reads the model verdict and applies ``done`` / ``execution``.
    """
    try:
        return await service.chat(
            message=payload.message,
            model=payload.model,
        )
    except TaskStateError as exc:
        _raise_task_state_error(exc)
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/api/day13/transition",
    response_model=Day13StateResponse,
    responses={400: {"model": ErrorResponse, "description": "Forbidden transition"}},
)
async def day13_transition(
    payload: Day13TransitionRequest,
    service: Day13TaskService = Depends(get_day13_service),
) -> Day13StateResponse:
    """Apply an explicit transition — rejected by code when not allowed."""
    try:
        service.transition(payload.to_stage)
    except TaskStateError as exc:
        _raise_task_state_error(exc)
    return service.state_response()


@router.post("/api/day13/pause", response_model=Day13StateResponse)
async def day13_pause(
    service: Day13TaskService = Depends(get_day13_service),
) -> Day13StateResponse:
    """Pause the task on any unfinished stage (stage/step/action preserved)."""
    try:
        service.pause()
    except TaskStateError as exc:
        _raise_task_state_error(exc)
    return service.state_response()


@router.post("/api/day13/resume", response_model=Day13StateResponse)
async def day13_resume(
    service: Day13TaskService = Depends(get_day13_service),
) -> Day13StateResponse:
    """Resume from the SAVED stage/current_step/expected_action."""
    try:
        service.resume()
    except TaskStateError as exc:
        _raise_task_state_error(exc)
    return service.state_response()


@router.delete("/api/day13/task")
async def day13_reset(
    service: Day13TaskService = Depends(get_day13_service),
) -> dict:
    """Forget the current task (demo/tests convenience)."""
    service.reset()
    return {"status": "cleared"}


# ----------------------------------------------------------------------
# Day 14 — invariants and state constraints.
# Invariants are stored separately from the conversation, injected into every
# model request as an ACTIVE INVARIANTS block, and enforced by a deterministic
# conflict check in code. Day 13's task state is left untouched.
# ----------------------------------------------------------------------
@router.get("/api/day14/state", response_model=Day14StateResponse)
async def day14_state(
    service: Day14InvariantService = Depends(get_day14_service),
) -> Day14StateResponse:
    """Active invariants and the separate conversation history (no provider call)."""
    return service.state_response()


@router.post(
    "/api/day14/chat",
    response_model=Day14ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day14_chat(
    payload: Day14ChatRequest,
    service: Day14InvariantService = Depends(get_day14_service),
) -> Day14ChatResponse:
    """Validate one request against the invariants and answer when allowed.

    A deterministic code check reports EVERY conflicting invariant. On a
    conflict the assistant refuses in code and names the invariant(s);
    otherwise the ACTIVE INVARIANTS block is sent to the model with the request.
    """
    try:
        return await service.chat(message=payload.message, model=payload.model)
    except DeepSeekError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.delete("/api/day14/history")
async def day14_clear_history(
    service: Day14InvariantService = Depends(get_day14_service),
) -> dict:
    """Clear ONLY the conversation; the active invariants are kept."""
    service.reset_conversation()
    return {"status": "cleared"}


# ----------------------------------------------------------------------
# Day 15 — controlled state transitions.
# The lifecycle (states + ALLOWED_TRANSITIONS + guards) is enforced in code.
# Every endpoint returns the structured decision; blocked transitions are
# persisted in the history but never change the state.
# ----------------------------------------------------------------------
def _raise_day15_error(exc: Day15Error) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/api/day15/state", response_model=Day15StateResponse)
async def day15_state(
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15StateResponse:
    """Current lifecycle state + allowed transitions + full history."""
    return service.state_response()


@router.post(
    "/api/day15/task",
    response_model=Day15StateResponse,
    responses={422: {"model": ErrorResponse, "description": "Invalid goal"}},
)
async def day15_create_task(
    payload: Day15TaskCreateRequest,
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15StateResponse:
    """Start a NEW task in ``planning`` (no provider call)."""
    try:
        service.create_task(payload.goal)
    except Day15Error as exc:
        _raise_day15_error(exc)
    return service.state_response()


@router.post(
    "/api/day15/start",
    response_model=Day15ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day15_start_task(
    payload: Day15TaskCreateRequest,
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15ChatResponse:
    """Create a task AND run the PLANNING stage automatically.

    A task is created in PLANNING, so the agent immediately generates a real
    plan, shows it in the chat and the state machine moves
    ``planning -> plan_approval``. The user never has to ask for a plan after
    creating the task.
    """
    try:
        return await service.start_task(payload.goal)
    except Day15Error as exc:
        _raise_day15_error(exc)


@router.post(
    "/api/day15/plan",
    response_model=Day15TransitionResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day15_prepare_plan(
    payload: Day15PlanRequest,
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15TransitionResponse:
    """Generate a plan and move ``planning -> plan_approval`` (guarded)."""
    try:
        return await service.prepare_plan(message=payload.message)
    except Day15Error as exc:
        _raise_day15_error(exc)


@router.post("/api/day15/approve", response_model=Day15TransitionResponse)
async def day15_approve_plan(
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15TransitionResponse:
    """Approve the plan (unlocks ``plan_approval -> execution``)."""
    try:
        return service.approve_plan()
    except Day15Error as exc:
        _raise_day15_error(exc)


@router.post("/api/day15/validation", response_model=Day15TransitionResponse)
async def day15_set_validation(
    payload: Day15ValidationRequest,
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15TransitionResponse:
    """Record the validation outcome (unlocks/locks ``validation -> done``)."""
    try:
        return service.set_validation(payload.passed, payload.summary)
    except Day15Error as exc:
        _raise_day15_error(exc)


@router.post(
    "/api/day15/transition",
    response_model=Day15TransitionResponse,
)
async def day15_transition(
    payload: Day15TransitionRequest,
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15TransitionResponse:
    """Request a transition — the backend state machine allows or blocks it."""
    try:
        return service.transition(payload.to_state)
    except Day15Error as exc:
        _raise_day15_error(exc)


@router.post("/api/day15/pause", response_model=Day15TransitionResponse)
async def day15_pause(
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15TransitionResponse:
    """Pause and remember the active state in ``previous_state``."""
    try:
        return service.pause()
    except Day15Error as exc:
        _raise_day15_error(exc)


@router.post("/api/day15/resume", response_model=Day15TransitionResponse)
async def day15_resume(
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15TransitionResponse:
    """Resume into EXACTLY the state the task was paused at."""
    try:
        return service.resume()
    except Day15Error as exc:
        _raise_day15_error(exc)


@router.get("/api/day15/history")
async def day15_history(
    service: Day15LifecycleService = Depends(get_day15_service),
) -> dict:
    """Transition history including blocked attempts (oldest first)."""
    if service.state() is None:
        return {"task_id": None, "history": []}
    return {
        "task_id": service.state().task_id,
        "history": [r.model_dump() for r in service.history()],
    }


@router.post(
    "/api/day15/chat",
    response_model=Day15ChatResponse,
    responses=_PROVIDER_ERROR_RESPONSES,
)
async def day15_chat(
    payload: Day15ChatRequest,
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15ChatResponse:
    """Natural-language assistant turn.

    The LLM only recognises the user's semantic intent; the application code
    maps it to an action and the state machine decides whether it is allowed.
    ``proposed_transition`` stays as a low-level debug escape hatch.
    """
    try:
        if payload.proposed_transition is not None:
            return service.chat(
                message=payload.message,
                proposed_transition=payload.proposed_transition,
                model=payload.model,
            )
        return await service.converse(
            message=payload.message,
            forced_intent=payload.forced_intent,
            model=payload.model,
        )
    except Day15Error as exc:
        _raise_day15_error(exc)


@router.post("/api/day15/scenario/{name}", response_model=Day15StateResponse)
async def day15_scenario(
    name: str,
    service: Day15LifecycleService = Depends(get_day15_service),
) -> Day15StateResponse:
    """Seed a deterministic demo scenario (skip/validation/pause/etc.)."""
    try:
        service.setup_scenario(name)
    except Day15Error as exc:
        _raise_day15_error(exc)
    return service.state_response()


@router.delete("/api/day15/task")
async def day15_reset(
    service: Day15LifecycleService = Depends(get_day15_service),
) -> dict:
    """Forget the current lifecycle state (demo/tests convenience)."""
    service.reset()
    return {"status": "cleared"}


# ----------------------------------------------------------------------
# Day 16 — MCP connection & tool discovery
# ----------------------------------------------------------------------
@router.get("/api/week4/day16/mcp/status", response_model=MCPStatusResponse)
async def day16_mcp_status(
    client: MCPClient = Depends(get_mcp_client),
) -> MCPStatusResponse:
    """Connect to the local Demo MCP server and list its tools.

    This really performs the MCP handshake over ``stdio`` (spawn subprocess ->
    initialize -> ``tools/list``) on every call. A connection/protocol failure
    is returned as a controlled ``connected=false`` payload; no stack trace is
    exposed to the client.
    """
    return await client.discover_tools()
