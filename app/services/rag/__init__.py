"""Day 21 — universal document ingestion + local RAG index."""
from app.services.rag.models import (
    BuildReport,
    Chunk,
    Document,
    LoaderStats,
    SearchHit,
    Segment,
    make_chunk_id,
    stable_id,
)
from app.services.rag.service import RagService

__all__ = [
    "BuildReport",
    "Chunk",
    "Document",
    "LoaderStats",
    "RagService",
    "SearchHit",
    "Segment",
    "make_chunk_id",
    "stable_id",
]
