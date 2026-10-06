"""Core data models for the reusable document-ingestion / RAG pipeline.

The pipeline deliberately keeps a single, small internal contract:

    source (PDF / TXT / MD / Telegram JSON)
        -> Document(text, metadata, segments)
        -> Chunk(chunk_id, text, metadata, doc_id)
        -> embeddings -> persistent vector index

``Document`` is the *normalized* form every loader produces. It carries the
full text plus optional ``Segment`` units (a PDF paragraph, a Telegram
message, an MD section, ...). Segments are what lets the structural chunker
respect the source structure instead of blindly cutting on character counts.

IDs are deterministic: the same source + chunking strategy produce the same
``doc_id`` / ``chunk_id`` across runs, which is the foundation for future
incremental indexing (Day 22+).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# ----------------------------------------------------------------------
# Deterministic identifiers
# ----------------------------------------------------------------------
_ID_SEPARATOR = "\x1f"


def stable_id(*parts: Any, length: int = 20) -> str:
    """Return a stable, collision-resistant id derived from *parts*.

    Uses BLAKE2b (stdlib) rather than the built-in ``hash`` because the
    latter is randomized per process and would break determinism / the
    persisted index across restarts.
    """
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(_ID_SEPARATOR.encode("utf-8"))
    return digest.hexdigest()[:length]


# ----------------------------------------------------------------------
# Segment — one structural unit inside a document
# ----------------------------------------------------------------------
@dataclass
class Segment:
    """A single coherent unit of a document.

    ``metadata`` is loader-specific, e.g. ``{"page": 12, "section": "..."}``
    for a PDF or ``{"message_id": 42, "author": "...", "date": "..."}`` for
    Telegram. Chunkers merge the metadata of the segments they cover.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()


# ----------------------------------------------------------------------
# Document — normalized loader output
# ----------------------------------------------------------------------
@dataclass
class Document:
    """Normalized document shared by every loader.

    ``metadata`` always contains at least ``source_type`` and ``source``.
    ``segments`` defaults to a single segment covering the whole text so that
    simple loaders (plain TXT) do not need to build units explicitly.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.segments:
            # Keep the whole document as one unit when a loader provides no
            # finer structure. An empty document is still legal (skipped by
            # chunkers).
            self.segments = [Segment(text=self.text, metadata={})] if self.text.strip() else []
        if not self.doc_id:
            self.doc_id = stable_id(
                self.metadata.get("source_type", ""),
                self.metadata.get("source", ""),
                self.metadata.get("locator", ""),
            )


# ----------------------------------------------------------------------
# Chunk — chunker output
# ----------------------------------------------------------------------
@dataclass
class Chunk:
    """A retrievable unit stored in the vector index alongside its metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def to_record(self, embedding: Iterable[float]) -> dict[str, Any]:
        """Flatten to the row shape used by the SQLite index."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": list(embedding),
        }


def make_chunk_id(
    doc_id: str,
    strategy: str,
    ordinal: int,
    text: str,
    length: int = 24,
) -> str:
    """Deterministic chunk id.

    It depends only on the parent document id, the chunking strategy, the
    chunk ordinal and the chunk text. Re-running the same pipeline therefore
    yields identical ids; changing the source text or strategy yields new
    ones (so a future incremental indexer can detect work to redo).
    """
    return stable_id(doc_id, strategy, ordinal, text, length=length)


# ----------------------------------------------------------------------
# Search hit + build/scan reports
# ----------------------------------------------------------------------
@dataclass
class SearchHit:
    """One similarity-search result returned by the vector index."""

    score: float
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    doc_id: str = ""


@dataclass
class LoaderStats:
    """Per-loader scan/load statistics exposed by the ``sources`` command."""

    source_type: str
    files: int = 0
    pages: int = 0
    pages_with_text: int = 0
    messages: int = 0
    usable_messages: int = 0
    skipped_messages: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "files": self.files,
            "pages": self.pages,
            "pages_with_text": self.pages_with_text,
            "messages": self.messages,
            "usable_messages": self.usable_messages,
            "skipped_messages": self.skipped_messages,
            "skip_reasons": dict(self.skip_reasons),
        }


@dataclass
class BuildReport:
    """Result of one full index build (shown by ``rag index``/``rag stats``)."""

    chunking: str
    documents_loaded: int = 0
    messages_loaded: int = 0
    messages_skipped: int = 0
    chunks_created: int = 0
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    chunks_by_source_type: dict[str, int] = field(default_factory=dict)
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    index_path: str = ""
    elapsed_seconds: float = 0.0
    loader_stats: list[LoaderStats] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunking": self.chunking,
            "documents_loaded": self.documents_loaded,
            "messages_loaded": self.messages_loaded,
            "messages_skipped": self.messages_skipped,
            "chunks_created": self.chunks_created,
            "avg_chunk_size": self.avg_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "chunks_by_source_type": dict(self.chunks_by_source_type),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "index_path": self.index_path,
            "elapsed_seconds": self.elapsed_seconds,
            "loader_stats": [s.as_dict() for s in self.loader_stats],
        }
