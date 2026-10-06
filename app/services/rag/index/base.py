"""Local persistent vector index interface."""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from app.services.rag.models import Chunk, SearchHit


class VectorIndexError(Exception):
    """Raised for index storage problems."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for a persistent chunk + embedding store."""

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Persist chunks and their vectors; return the number written."""

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[SearchHit]:
        """Return the ``top_k`` most similar chunks."""

    def iter_chunks(
        self, *, source_type: str | None = None, limit: int | None = None, offset: int = 0
    ) -> Iterable[Chunk]:
        """Iterate stored chunks (for ``rag chunks``)."""

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Return one chunk by id, or None."""

    def stats(self) -> dict:
        """Return index statistics."""

    def count(self) -> int:
        """Return the number of stored chunks."""

    def reset(self) -> None:
        """Drop all stored data and metadata."""

    def close(self) -> None:
        """Release resources."""
