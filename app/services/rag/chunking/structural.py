"""Structure-aware chunking.

This is NOT fixed chunking with a different size. It walks the document's
segments (PDF paragraphs + detected sections, Markdown sections, Telegram
messages) and groups *whole* segments, starting a new chunk whenever:

* a new section/heading begins;
* adding the next segment would exceed ``max_chars``.

A segment is never cut in the middle unless it alone is longer than
``max_chars``; only then is it split on sentence boundaries. The result keeps
a heading together with the paragraphs it introduces and keeps a Telegram
conversation together with its replies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.rag.chunking.base import (
    ChunkingStrategy,
    SEGMENT_SEPARATOR,
    merge_segment_metadata,
)
from app.services.rag.models import Chunk, Document, Segment, make_chunk_id

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split an oversized single segment on sentence/word boundaries."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer)
        if len(sentence) <= max_chars:
            buffer = sentence
        else:
            # Very long "sentence": hard-split on whitespace near the limit.
            remaining = sentence
            while len(remaining) > max_chars:
                cut = remaining.rfind(" ", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                pieces.append(remaining[:cut].strip())
                remaining = remaining[cut:].strip()
            buffer = remaining
    if buffer.strip():
        pieces.append(buffer.strip())
    return [p for p in pieces if p.strip()]


@dataclass
class StructuralChunker:
    """Group segments respecting section / conversation boundaries."""

    max_chars: int = 1800
    min_chars: int = 400
    name: str = "structural"

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if self.min_chars < 0:
            raise ValueError("min_chars must not be negative")

    def chunk(self, document: Document) -> list[Chunk]:
        groups: list[list[Segment]] = []
        current: list[Segment] = []
        current_chars = 0
        current_section: str | None = None

        def flush() -> None:
            nonlocal current, current_chars, current_section
            if current:
                groups.append(current)
            current = []
            current_chars = 0
            current_section = None

        for segment in document.segments:
            text = segment.text.strip()
            if not text:
                continue
            section = segment.metadata.get("section")
            is_heading = bool(segment.metadata.get("is_heading"))

            if len(text) > self.max_chars:
                flush()
                for piece in _split_long_text(text, self.max_chars):
                    groups.append([Segment(text=piece, metadata=dict(segment.metadata))])
                continue

            if current:
                section_changed = bool(
                    section and current_section and section != current_section
                )
                would_overflow = current_chars + len(text) + len(SEGMENT_SEPARATOR) > self.max_chars
                # Boundaries only apply once the current chunk has meaningful
                # content; otherwise many short (mis)detected headings would
                # produce tiny chunks.
                at_boundary = (is_heading or section_changed) and current_chars >= self.min_chars
                if at_boundary or would_overflow:
                    flush()
            current.append(segment)
            current_chars += len(text) + len(SEGMENT_SEPARATOR)
            if not current_section and section:
                current_section = section
        flush()

        chunks: list[Chunk] = []
        ordinal = 0
        for group in groups:
            text = SEGMENT_SEPARATOR.join(seg.text for seg in group).strip()
            if not text:
                continue
            metadata = merge_segment_metadata(document.metadata, group)
            metadata.update(
                {
                    "chunking": self.name,
                    "segment_count": len(group),
                    "max_chars": self.max_chars,
                    "min_chars": self.min_chars,
                }
            )
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, self.name, ordinal, text),
                    text=text,
                    metadata=metadata,
                    doc_id=document.doc_id,
                )
            )
            ordinal += 1
        return chunks
