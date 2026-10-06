"""Fixed-size chunking with overlap."""
from __future__ import annotations

from dataclasses import dataclass

from app.services.rag.chunking.base import (
    ChunkingStrategy,
    _segmented_text,
    merge_segment_metadata,
)
from app.services.rag.models import Chunk, Document, make_chunk_id


@dataclass
class FixedChunker:
    """Sliding-window chunker.

    The document's segments are concatenated (separated by a blank line) and
    then sliced into windows of ``chunk_size`` characters that advance by
    ``chunk_size - overlap``. A chunk's metadata is the merge of every segment
    it overlaps, so a window spanning pages 12-13 keeps both page numbers.
    """

    chunk_size: int = 1200
    overlap: int = 200
    name: str = "fixed"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must not be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    def chunk(self, document: Document) -> list[Chunk]:
        text, ranges = _segmented_text(document)
        if not text.strip():
            return []

        step = self.chunk_size - self.overlap
        total = len(text)
        chunks: list[Chunk] = []
        ordinal = 0
        start = 0
        while start < total:
            end = min(start + self.chunk_size, total)
            piece = text[start:end]
            cleaned = piece.strip()
            if cleaned:
                covered = [seg for (a, b, seg) in ranges if a < end and b > start]
                metadata = merge_segment_metadata(document.metadata, covered)
                metadata.update(
                    {
                        "chunking": self.name,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.overlap,
                        "start_char": start,
                        "end_char": end,
                    }
                )
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(document.doc_id, self.name, ordinal, cleaned),
                        text=cleaned,
                        metadata=metadata,
                        doc_id=document.doc_id,
                    )
                )
                ordinal += 1
            if end >= total:
                break
            start += step
        return chunks
