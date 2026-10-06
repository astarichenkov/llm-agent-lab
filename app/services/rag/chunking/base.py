"""Chunking strategies.

Two strategies operate on the same normalized :class:`Document` and are
intentionally different in spirit:

* ``FixedChunker`` ignores structure and produces sliding windows of a fixed
  character size with overlap;
* ``StructuralChunker`` keeps whole segments together and starts a new chunk
  at section/conversation boundaries.

Both never drop the source metadata: every produced chunk carries the merged
metadata of the segments it covers (pages, message ids, authors, sections).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.services.rag.models import Chunk, Document, Segment, make_chunk_id

SEGMENT_SEPARATOR = "\n\n"


class ChunkingStrategy(Protocol):
    """Protocol implemented by every chunker."""

    name: str

    def chunk(self, document: Document) -> list[Chunk]:
        """Split one document into retrievable chunks."""


# ----------------------------------------------------------------------
# metadata merging
# ----------------------------------------------------------------------
def _ordered_unique(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        key = value if isinstance(value, (str, int, float, bool)) else repr(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def merge_segment_metadata(base: dict[str, Any], segments: list[Segment]) -> dict[str, Any]:
    """Merge document metadata with the metadata of the covered segments.

    The result always keeps ``source_type``/``source`` from the document and
    adds page ranges, message ids, authors and sections when present.
    """
    metadata = dict(base)

    pages = [s.metadata.get("page") for s in segments if s.metadata.get("page") is not None]
    if pages:
        metadata["page_from"] = min(pages)
        metadata["page_to"] = max(pages)
        if min(pages) == max(pages):
            metadata["page"] = min(pages)

    message_ids = [s.metadata.get("message_id") for s in segments if s.metadata.get("message_id") is not None]
    if message_ids:
        metadata["message_ids"] = message_ids
        metadata["message_id"] = message_ids[0] if len(message_ids) == 1 else message_ids[0]
        metadata["message_id_to"] = message_ids[-1]

    authors = _ordered_unique([s.metadata.get("author") for s in segments if s.metadata.get("author")])
    if authors:
        metadata["authors"] = authors
        metadata["author"] = authors[0] if len(authors) == 1 else ", ".join(str(a) for a in authors)

    reply_ids = [
        s.metadata.get("reply_to_message_id")
        for s in segments
        if s.metadata.get("reply_to_message_id") is not None
    ]
    if reply_ids:
        metadata["reply_to_message_ids"] = _ordered_unique(reply_ids)

    sections = _ordered_unique(
        [s.metadata.get("section") for s in segments if s.metadata.get("section")]
    )
    if sections:
        metadata["sections"] = sections
        if len(sections) == 1:
            metadata["section"] = sections[0]

    return metadata


def _segmented_text(document: Document) -> tuple[str, list[tuple[int, int, Segment]]]:
    """Concatenate segments and remember each segment's char range."""
    parts: list[str] = []
    ranges: list[tuple[int, int, Segment]] = []
    position = 0
    for segment in document.segments:
        text = segment.text
        if not text.strip():
            continue
        if position:
            parts.append(SEGMENT_SEPARATOR)
            position += len(SEGMENT_SEPARATOR)
        start = position
        parts.append(text)
        position += len(text)
        ranges.append((start, position, segment))
    return "".join(parts), ranges
