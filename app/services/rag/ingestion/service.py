"""Full index-build orchestration.

    documents -> normalization -> chunking -> embeddings -> vector index

The build is atomic: everything is written to a temporary SQLite file and
only ``os.replace``d into place after it succeeds. An obvious failure never
leaves a half-built index in the configured path.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.services.rag.chunking import FIXED, build_chunker
from app.services.rag.embeddings import build_embedding_provider
from app.services.rag.embeddings.base import EmbeddingProvider
from app.services.rag.index.sqlite_index import SqliteVectorIndex
from app.services.rag.loaders import PdfLoader, TelegramLoader, TextLoader
from app.services.rag.loaders.base import DocumentLoader
from app.services.rag.models import BuildReport, Chunk, Document, LoaderStats

logger = logging.getLogger("app.services.rag.ingestion")

EMBEDDING_BATCH_SIZE = 64


class IngestionService:
    """Build a persistent RAG index from the configured knowledge sources."""

    def __init__(
        self,
        settings: Settings,
        *,
        manual_path: str | None = None,
        telegram_path: str | None = None,
        index_path: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings
        self.manual_path = manual_path or settings.rag_manual_path
        self.telegram_path = telegram_path or settings.rag_telegram_export_path
        self.index_path = Path(index_path or settings.rag_index_path)
        self._embedding_provider = embedding_provider

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def build_loaders(self) -> list[DocumentLoader]:
        """Create the configured loaders (manual txt/md/pdf + Telegram)."""
        return [
            TextLoader(self.manual_path, source_type="manual"),
            PdfLoader(self.manual_path, source_type="manual"),
            TelegramLoader(
                self.telegram_path,
                gap_minutes=self.settings.rag_telegram_gap_minutes,
                max_group_messages=self.settings.rag_telegram_max_group_messages,
                max_group_chars=self.settings.rag_structural_max_chars,
            ),
        ]

    def scan_sources(self) -> list[LoaderStats]:
        """Cheap source statistics for the ``rag sources`` command."""
        stats: list[LoaderStats] = []
        for loader in self.build_loaders():
            try:
                stats.append(loader.scan())
            except Exception as exc:  # noqa: BLE001 - one bad source must not hide others
                logger.warning("Scan failed for %s: %s", loader.source_type, exc)
                stats.append(LoaderStats(source_type=loader.source_type))
        return stats

    def load_documents(self) -> tuple[list[Document], list[LoaderStats]]:
        """Load and normalize every configured source."""
        documents: list[Document] = []
        stats: list[LoaderStats] = []
        for loader in self.build_loaders():
            try:
                loaded = list(loader.load())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Load failed for %s: %s", loader.source_type, exc)
                stats.append(loader.scan())
                continue
            documents.extend(loaded)
            stats.append(loader.stats())
        return documents, stats

    # ------------------------------------------------------------------
    # embedding provider
    # ------------------------------------------------------------------
    def embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = build_embedding_provider(self.settings)
        return self._embedding_provider

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self, chunking: str = FIXED, *, rebuild: bool = True) -> BuildReport:
        """Run the full pipeline and atomically replace the index file."""
        started = time.time()
        provider = self.embedding_provider()
        chunker = build_chunker(chunking, settings=self.settings)

        documents, loader_stats = self.load_documents()
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(chunker.chunk(document))

        target = self.index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".building")
        if temp.exists():
            temp.unlink()

        dimension = 0
        try:
            index = SqliteVectorIndex(temp)
            try:
                for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                    batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
                    embeddings = provider.embed([chunk.text for chunk in batch])
                    if embeddings and not dimension:
                        dimension = len(embeddings[0])
                    index.add_chunks(batch, embeddings)
                index.set_meta(
                    {
                        "embedding_provider": provider.name,
                        "embedding_model": provider.model,
                        "embedding_dimension": dimension or provider.dimension,
                        "chunking": chunker.name,
                        "chunk_size": self.settings.rag_chunk_size,
                        "chunk_overlap": self.settings.rag_chunk_overlap,
                        "structural_max_chars": self.settings.rag_structural_max_chars,
                        "structural_min_chars": self.settings.rag_structural_min_chars,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                index.checkpoint()
            finally:
                index.close()
            os.replace(temp, target)
        except BaseException:
            # Never leave a partial build behind.
            temp.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                Path(str(temp) + suffix).unlink(missing_ok=True)
            raise

        sizes = [len(chunk.text) for chunk in chunks]
        by_source: dict[str, int] = {}
        for chunk in chunks:
            source_type = str(chunk.metadata.get("source_type", "unknown"))
            by_source[source_type] = by_source.get(source_type, 0) + 1

        messages_loaded = 0
        messages_skipped = 0
        for stat in loader_stats:
            if stat.source_type == "telegram":
                messages_loaded += stat.usable_messages
                messages_skipped += stat.skipped_messages

        report = BuildReport(
            chunking=chunker.name,
            documents_loaded=len(documents),
            messages_loaded=messages_loaded,
            messages_skipped=messages_skipped,
            chunks_created=len(chunks),
            avg_chunk_size=round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
            min_chunk_size=min(sizes) if sizes else 0,
            max_chunk_size=max(sizes) if sizes else 0,
            chunks_by_source_type=by_source,
            embedding_provider=provider.name,
            embedding_model=provider.model,
            embedding_dimension=dimension or provider.dimension,
            index_path=str(target),
            elapsed_seconds=round(time.time() - started, 2),
            loader_stats=loader_stats,
        )
        return report
