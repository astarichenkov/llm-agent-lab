"""Embedding provider abstraction.

RAG code depends only on this protocol, so the concrete model can be swapped
(local Ollama today, a hosted API later) without touching the pipeline.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


class EmbeddingError(Exception):
    """Raised when an embedding provider cannot produce vectors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol every embedding backend implements."""

    name: str
    model: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
