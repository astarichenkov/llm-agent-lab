"""Provider-neutral generation (LLM) abstraction for the RAG pipeline.

Day 22 needs ONE generation model shared by the No-RAG and RAG modes so the
comparison is honest. The embedding model (``bge-m3``) must never be used for
generation: the two roles have different interfaces and different providers.

The protocol is deliberately tiny::

    await provider.generate(messages, temperature=..., max_tokens=...)

Implementations wrap an existing service (DeepSeek) or a local Ollama server.
Tests inject a deterministic fake, so no unit test ever contacts a network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class GenerationResult:
    """One completion plus its provider-side bookkeeping."""

    content: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class GenerationError(Exception):
    """Any generation failure.

    ``message`` is safe to show in the browser (no secrets / stack traces);
    ``status_code`` is the HTTP status the API layer should return.
    """

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@runtime_checkable
class GenerationProvider(Protocol):
    """Minimal interface every generation provider implements."""

    #: short identifier used in API responses / the UI (``deepseek``/``ollama``)
    name: str
    #: model name actually sent to the provider
    model: str

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> GenerationResult: ...
