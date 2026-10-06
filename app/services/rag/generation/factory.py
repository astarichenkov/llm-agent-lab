"""Factory for the configured generation provider.

The generation model is resolved from configuration in exactly ONE place:

* ``RAG_GENERATION_PROVIDER`` -- ``deepseek`` (default) or ``ollama``;
* ``RAG_GENERATION_MODEL``    -- explicit model name. When empty, the
  provider's own configured default is used (``DEEPSEEK_MODEL`` for DeepSeek,
  which is required for Ollama because no safe default exists).

The factory never contacts the network: a provider is only reached when a
generation request is actually made.
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.services.deepseek import DeepSeekService
from app.services.rag.generation.base import GenerationError, GenerationProvider
from app.services.rag.generation.deepseek import DeepSeekGenerationProvider
from app.services.rag.generation.ollama import OllamaGenerationProvider

logger = logging.getLogger("app.services.rag.generation")


def build_generation_provider(settings: Settings) -> GenerationProvider:
    """Build the generation provider selected by configuration."""
    provider = (settings.rag_generation_provider or "deepseek").strip().lower()
    model = (settings.rag_generation_model or "").strip()

    if provider == "ollama":
        return OllamaGenerationProvider(
            settings.rag_ollama_base_url,
            model=model,
            timeout=settings.rag_generation_timeout_seconds,
        )
    if provider in ("deepseek", "default"):
        resolved_model = model or settings.deepseek_model
        return DeepSeekGenerationProvider(
            DeepSeekService(settings), model=resolved_model
        )
    raise GenerationError(
        f"Unknown generation provider '{provider}'. "
        "Use 'deepseek' or 'ollama'.",
        status_code=500,
    )
