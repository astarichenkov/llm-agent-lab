"""High-level RAG service used by the CLI.

It ties together source scanning, index building, statistics, chunk listing
and a debug similarity search. Day 21 deliberately stops here — no LLM
generation, no reranking, no citations in an answer. Those belong to Day 22+.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.services.rag.embeddings.base import EmbeddingProvider
from app.services.rag.embeddings.factory import build_embedding_provider
from app.services.rag.embeddings.hashing import HashingEmbeddingProvider
from app.services.rag.embeddings.ollama import OllamaEmbeddingProvider
from app.services.rag.index.sqlite_index import SqliteVectorIndex
from app.services.rag.ingestion.service import IngestionService
from app.services.rag.models import BuildReport, Chunk, LoaderStats, SearchHit

logger = logging.getLogger("app.services.rag.service")


def build_query_provider(
    settings: Settings,
    meta: dict,
    provider: EmbeddingProvider | None = None,
) -> EmbeddingProvider:
    """Match a query embedder to the provider that built an index.

    Reused both by :meth:`RagService.search` and by the Day 21 chunking
    comparison so a query embedding is always computed with the exact same
    provider/model that produced the persisted vectors.
    """
    if provider is not None:
        return provider
    name = str(meta.get("embedding_provider", ""))
    model = str(meta.get("embedding_model", ""))
    dimension = int(meta.get("embedding_dimension", 0) or 0)
    if name == "hashing":
        return HashingEmbeddingProvider(dimension=dimension or 384, model=model or "hashing-ngrams")
    if name == "ollama":
        return OllamaEmbeddingProvider(
            settings.rag_ollama_base_url,
            model=model or settings.rag_embedding_model,
            timeout=settings.rag_ollama_timeout_seconds,
        )
    if name == "sentence_transformers":
        from app.services.rag.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingProvider,
        )

        return SentenceTransformerEmbeddingProvider(
            model=model or settings.rag_embedding_model
        )
    return build_embedding_provider(settings)


class RagService:
    """Read/write facade over the ingestion pipeline and the vector index."""

    def __init__(
        self,
        settings: Settings,
        *,
        index_path: str | None = None,
        manual_path: str | None = None,
        telegram_path: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings
        self.index_path = Path(index_path or settings.rag_index_path)
        self.ingestion = IngestionService(
            settings,
            manual_path=manual_path,
            telegram_path=telegram_path,
            index_path=str(self.index_path),
            embedding_provider=embedding_provider,
        )
        self._provider = embedding_provider

    # ------------------------------------------------------------------
    # sources
    # ------------------------------------------------------------------
    def sources(self) -> list[LoaderStats]:
        return self.ingestion.scan_sources()

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build_index(self, chunking: str = "fixed", *, rebuild: bool = True) -> BuildReport:
        return self.ingestion.build(chunking, rebuild=rebuild)

    # ------------------------------------------------------------------
    # index access
    # ------------------------------------------------------------------
    def index_exists(self) -> bool:
        return self.index_path.exists()

    def _open_index(self) -> SqliteVectorIndex:
        return SqliteVectorIndex(self.index_path)

    def stats(self) -> dict:
        if not self.index_exists():
            return {
                "chunks": 0,
                "index_path": str(self.index_path),
                "exists": False,
            }
        index = self._open_index()
        try:
            stats = index.stats()
            stats["exists"] = True
            return stats
        finally:
            index.close()

    def neighbors(
        self, chunk_id: str, *, radius: int = 1, limit: int = 8
    ) -> list[Chunk]:
        """Adjacent same-document chunks (Day 25 controlled expansion)."""
        if not self.index_exists():
            return []
        index = self._open_index()
        try:
            return list(index.neighbors(chunk_id, radius=radius, limit=limit))
        finally:
            index.close()

    def chunks(
        self,
        source_type: str | None = None,
        *,
        limit: int = 5,
        offset: int = 0,
    ) -> list[Chunk]:
        if not self.index_exists():
            return []
        index = self._open_index()
        try:
            return list(
                index.iter_chunks(source_type=source_type, limit=limit, offset=offset)
            )
        finally:
            index.close()

    # ------------------------------------------------------------------
    # debug similarity search (no LLM)
    # ------------------------------------------------------------------
    def _query_provider(self, meta: dict) -> EmbeddingProvider:
        """Match the query embedder to the provider used to build the index."""
        return build_query_provider(self.settings, meta, self._provider)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[SearchHit]:
        if not self.index_exists():
            return []
        index = self._open_index()
        try:
            meta = index.get_meta()
            provider = self._query_provider(meta)
            vector = provider.embed_query(query)
            return index.search(vector, top_k=top_k, source_type=source_type)
        finally:
            index.close()
