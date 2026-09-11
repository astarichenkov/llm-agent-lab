"""Thin LLM abstraction consumed by :class:`~app.agents.agent.Agent`.

The agent layer must not know about DeepSeek or OpenRouter specifics. These
adapters wrap the *existing* ``DeepSeekService`` / ``OpenRouterService`` (no
provider logic is reimplemented) and expose one uniform coroutine::

    await client.complete(messages, model=..., temperature=..., max_tokens=...)

Because ``Agent`` only depends on the small :class:`LLMClient` protocol, tests
can inject a deterministic fake without ever touching a network.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.services.deepseek import DeepSeekService
from app.services.openrouter_service import OpenRouterService


class LLMResult(BaseModel):
    """Provider-agnostic completion result."""

    content: str
    finish_reason: str | None = None
    usage: dict | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every provider adapter implements."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: str | None = None,
        thinking: bool = False,
    ) -> LLMResult: ...


class DeepSeekLLMClient:
    """Adapter over the existing :class:`DeepSeekService`."""

    def __init__(self, service: DeepSeekService) -> None:
        self._service = service

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: str | None = None,
        thinking: bool = False,
    ) -> LLMResult:
        content, finish_reason, usage = await self._service.generate(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            thinking=thinking,
        )
        return LLMResult(content=content, finish_reason=finish_reason, usage=usage)


class OpenRouterLLMClient:
    """Adapter over the existing :class:`OpenRouterService`.

    ``thinking`` is accepted for interface parity but ignored: OpenRouter
    exposes provider-specific reasoning switches, which the current Day 5
    model set does not use.
    """

    def __init__(self, service: OpenRouterService) -> None:
        self._service = service

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: str | None = None,
        thinking: bool = False,
    ) -> LLMResult:
        content, finish_reason, usage = await self._service.generate(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return LLMResult(content=content, finish_reason=finish_reason, usage=usage)
