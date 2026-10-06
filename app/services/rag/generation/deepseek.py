"""DeepSeek generation provider.

Reuses the project's EXISTING :class:`~app.services.deepseek.DeepSeekService`
instead of reimplementing an OpenAI-compatible client. The adapter only maps
the service result tuple to :class:`GenerationResult` and translates provider
errors into :class:`GenerationError` (with the same HTTP status codes the rest
of the application already uses).
"""
from __future__ import annotations

from app.services.deepseek import DeepSeekError, DeepSeekService
from app.services.rag.generation.base import GenerationError, GenerationResult


class DeepSeekGenerationProvider:
    """Adapter over the existing DeepSeek service."""

    name = "deepseek"

    def __init__(self, service: DeepSeekService, *, model: str) -> None:
        self._service = service
        self.model = model

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> GenerationResult:
        try:
            content, finish_reason, usage = await self._service.generate(
                messages,
                model=self.model,
                temperature=0.0 if temperature is None else float(temperature),
                max_tokens=1024 if max_tokens is None else int(max_tokens),
                response_format=response_format,
                # Day 22 compares two prompts with the SAME model. Reasoning is
                # disabled so temperature is meaningful, the output budget is
                # not silently consumed by hidden reasoning, and the comparison
                # is reproducible.
                thinking=False,
            )
        except DeepSeekError as exc:
            # Preserve the friendly message + status mapping used everywhere
            # else in the application.
            raise GenerationError(exc.message, status_code=exc.status_code) from exc
        return GenerationResult(
            content=content, finish_reason=finish_reason, usage=usage
        )
