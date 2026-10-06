"""Chunking strategy registry / factory."""
from __future__ import annotations

from app.config import Settings
from app.services.rag.chunking.base import ChunkingStrategy
from app.services.rag.chunking.fixed import FixedChunker
from app.services.rag.chunking.structural import StructuralChunker

FIXED = "fixed"
STRUCTURAL = "structural"
SUPPORTED_STRATEGIES = (FIXED, STRUCTURAL)


def build_chunker(
    name: str,
    *,
    settings: Settings | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    structural_max_chars: int | None = None,
    structural_min_chars: int | None = None,
) -> ChunkingStrategy:
    """Build a chunker by name, pulling unset parameters from settings."""
    normalized = (name or "").strip().lower()
    if settings is not None:
        chunk_size = chunk_size if chunk_size is not None else settings.rag_chunk_size
        overlap = overlap if overlap is not None else settings.rag_chunk_overlap
        structural_max_chars = (
            structural_max_chars
            if structural_max_chars is not None
            else settings.rag_structural_max_chars
        )
        structural_min_chars = (
            structural_min_chars
            if structural_min_chars is not None
            else settings.rag_structural_min_chars
        )
    chunk_size = chunk_size if chunk_size is not None else 1200
    overlap = overlap if overlap is not None else 200
    structural_max_chars = structural_max_chars if structural_max_chars is not None else 1800
    structural_min_chars = structural_min_chars if structural_min_chars is not None else 400

    if normalized == FIXED:
        return FixedChunker(chunk_size=chunk_size, overlap=overlap)
    if normalized == STRUCTURAL:
        return StructuralChunker(max_chars=structural_max_chars, min_chars=structural_min_chars)
    raise ValueError(
        f"Unknown chunking strategy '{name}'. "
        f"Supported: {', '.join(SUPPORTED_STRATEGIES)}"
    )


__all__ = [
    "FIXED",
    "STRUCTURAL",
    "SUPPORTED_STRATEGIES",
    "ChunkingStrategy",
    "FixedChunker",
    "StructuralChunker",
    "build_chunker",
]
