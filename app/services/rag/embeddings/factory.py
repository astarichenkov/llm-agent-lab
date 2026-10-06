"""Embedding provider factory.

Resolution order for ``provider = auto``:

1. **Ollama** — preferred, but only if the server is reachable AND the
   configured model is already pulled;
2. **sentence-transformers** — if the optional package is installed;
3. **hashing** — deterministic, dependency-free local fallback.

This guarantees the pipeline always works offline while using a real local
neural embedding model whenever one is actually available.
"""
from __future__ import annotations

import importlib.util
import logging

from app.config import Settings
from app.services.rag.embeddings.base import EmbeddingError, EmbeddingProvider
from app.services.rag.embeddings.hashing import HashingEmbeddingProvider
from app.services.rag.embeddings.ollama import (
    OllamaEmbeddingProvider,
    ollama_model_available,
)

logger = logging.getLogger("app.services.rag.embeddings.factory")

AUTO = "auto"
OLLAMA = "ollama"
SENTENCE_TRANSFORMERS = "sentence_transformers"
HASHING = "hashing"
SUPPORTED_PROVIDERS = (AUTO, OLLAMA, SENTENCE_TRANSFORMERS, HASHING)


def _sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the embedding provider for the current settings."""
    requested = (settings.rag_embedding_provider or AUTO).strip().lower()
    model = settings.rag_embedding_model

    if requested == HASHING:
        return HashingEmbeddingProvider(dimension=settings.rag_embedding_dimension)

    if requested == OLLAMA:
        return OllamaEmbeddingProvider(
            settings.rag_ollama_base_url,
            model=model,
            timeout=settings.rag_ollama_timeout_seconds,
        )

    if requested == SENTENCE_TRANSFORMERS:
        from app.services.rag.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingProvider,
        )

        return SentenceTransformerEmbeddingProvider(model=model)

    if requested != AUTO:
        raise EmbeddingError(
            f"Unknown embedding provider '{requested}'. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    # auto
    if ollama_model_available(settings.rag_ollama_base_url, model):
        logger.info("RAG embeddings: using Ollama model %s", model)
        return OllamaEmbeddingProvider(
            settings.rag_ollama_base_url,
            model=model,
            timeout=settings.rag_ollama_timeout_seconds,
        )
    if _sentence_transformers_available():
        try:
            from app.services.rag.embeddings.sentence_transformer import (
                SentenceTransformerEmbeddingProvider,
            )

            logger.info("RAG embeddings: using sentence-transformers %s", model)
            return SentenceTransformerEmbeddingProvider(model=model)
        except Exception as exc:  # noqa: BLE001 - fall through to hashing
            logger.warning("sentence-transformers unavailable (%s); using hashing", exc)
    logger.info("RAG embeddings: using offline hashing fallback embedder")
    return HashingEmbeddingProvider(dimension=settings.rag_embedding_dimension)


__all__ = [
    "AUTO",
    "HASHING",
    "OLLAMA",
    "SENTENCE_TRANSFORMERS",
    "SUPPORTED_PROVIDERS",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "build_embedding_provider",
]
