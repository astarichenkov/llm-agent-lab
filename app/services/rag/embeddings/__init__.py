"""Embedding providers and factory."""
from app.services.rag.embeddings.base import EmbeddingError, EmbeddingProvider
from app.services.rag.embeddings.factory import (
    AUTO,
    HASHING,
    OLLAMA,
    SENTENCE_TRANSFORMERS,
    SUPPORTED_PROVIDERS,
    build_embedding_provider,
)
from app.services.rag.embeddings.hashing import HashingEmbeddingProvider
from app.services.rag.embeddings.ollama import OllamaEmbeddingProvider

__all__ = [
    "AUTO",
    "HASHING",
    "OLLAMA",
    "SENTENCE_TRANSFORMERS",
    "SUPPORTED_PROVIDERS",
    "EmbeddingError",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "build_embedding_provider",
]
