"""Ollama generation provider (local, configurable model).

This is the GENERATION counterpart of
:class:`~app.services.rag.embeddings.ollama.OllamaEmbeddingProvider`. The
model is always supplied by configuration -- never hard-coded -- because the
local server may expose a different chat model than ``bge-m3`` (which is an
embedding-only model and must NOT be used here).
"""
from __future__ import annotations

import logging

import httpx

from app.services.rag.generation.base import GenerationError, GenerationResult

logger = logging.getLogger("app.services.rag.generation.ollama")


class OllamaGenerationProvider:
    """Generate text through a local Ollama ``/api/chat`` endpoint."""

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model:
            raise GenerationError(
                "Ollama generation model is not configured "
                "(set RAG_GENERATION_MODEL).",
                status_code=500,
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client

    async def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        try:
            if self._client is not None:
                response = await self._client.post(
                    url, json=payload, timeout=self.timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                f"Ollama rejected the generation request "
                f"(HTTP {exc.response.status_code if exc.response else '?'}). "
                "Check RAG_GENERATION_MODEL.",
            ) from exc
        except httpx.TimeoutException as exc:
            raise GenerationError(
                "Ollama generation timed out. Please try again.", status_code=504
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is the Ollama server running?"
            ) from exc

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> GenerationResult:
        options: dict = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        # Day 24 — structured grounded output. Ollama uses ``format="json"``
        # for its JSON mode; whichever OpenAI-style schema the caller passes
        # is intentionally reduced to that supported value.
        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"

        data = await self._post("/api/chat", payload)
        message = data.get("message") or {}
        content = str(message.get("content") or "")
        finish_reason = data.get("done_reason") or ("stop" if data.get("done") else None)
        return GenerationResult(
            content=content,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            },
        )
