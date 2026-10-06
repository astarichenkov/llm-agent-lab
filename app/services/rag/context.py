"""Context builder for the RAG prompt.

Turns retrieved chunks into a compact, label-rich text block that the
generation model can consume. It is deliberately separate from retrieval and
from the LLM so it can be unit-tested in isolation and improved on Day 23+
(reranking / citations) without touching the index or the provider.

The block exposes only REAL metadata: a manual chunk has ``page``/``section``
and a Telegram chunk has ``message_ids``/``authors``. Missing fields are
simply omitted -- nothing is invented.
"""
from __future__ import annotations

from app.schemas.day22 import RetrievedChunk

# Human-readable labels explain the nature of each source type to the model.
SOURCE_TYPE_LABELS = {
    "manual": "Официальная техническая документация (manual)",
    "telegram": "Обсуждение владельцев (Telegram community)",
}


class ContextBuilder:
    """Assemble retrieved chunks into a labelled context block."""

    def __init__(self, *, max_chars: int = 12000) -> None:
        self.max_chars = max_chars

    # ------------------------------------------------------------------
    def _metadata_lines(self, chunk: RetrievedChunk) -> list[str]:
        meta = chunk.metadata or {}
        lines: list[str] = []
        lines.append(f"source_type: {chunk.source_type}")
        label = SOURCE_TYPE_LABELS.get(chunk.source_type)
        if label:
            lines.append(f"source_kind: {label}")

        if chunk.source:
            lines.append(f"source: {chunk.source}")

        # Manual metadata -------------------------------------------------
        if chunk.source_type == "manual":
            page = meta.get("page")
            if page is None:
                page_from, page_to = meta.get("page_from"), meta.get("page_to")
                if page_from is not None and page_to is not None and page_from != page_to:
                    page = f"{page_from}-{page_to}"
                elif page_from is not None:
                    page = page_from
            if page is not None:
                lines.append(f"page: {page}")
            if meta.get("section"):
                lines.append(f"section: {meta['section']}")

        # Telegram metadata ----------------------------------------------
        if chunk.source_type == "telegram":
            if meta.get("chat_name"):
                lines.append(f"chat: {meta['chat_name']}")
            if meta.get("message_ids"):
                ids = meta["message_ids"]
                if isinstance(ids, (list, tuple)):
                    lines.append("message_ids: " + ", ".join(str(i) for i in ids))
                else:
                    lines.append(f"message_ids: {ids}")
            if meta.get("authors"):
                authors = meta["authors"]
                if isinstance(authors, (list, tuple)):
                    lines.append("authors: " + ", ".join(str(a) for a in authors))
                else:
                    lines.append(f"authors: {authors}")
            if meta.get("date_from"):
                lines.append(f"date_from: {meta['date_from']}")
            if meta.get("date_to"):
                lines.append(f"date_to: {meta['date_to']}")

        lines.append(f"chunk_id: {chunk.chunk_id}")
        return lines

    def build(self, chunks: list[RetrievedChunk]) -> str:
        """Return the full context block (empty string when no chunks)."""
        if not chunks:
            return ""

        blocks: list[str] = []
        used = 0
        for chunk in chunks:
            header = "\n".join(self._metadata_lines(chunk))
            text = (chunk.text or "").strip()
            block = f"{header}\n\nTEXT:\n{text}"
            remaining = self.max_chars - used
            if len(block) + 1 > remaining:
                # Keep whole chunks where possible; only the current oversized
                # block is cut so the context never exceeds the cap.
                if remaining <= len(header) + 20:
                    break
                block = block[:remaining].rstrip() + "\n[...обрезано...]"
            blocks.append(block)
            used += len(block) + 1
            if used >= self.max_chars:
                break

        return "\n\n".join(blocks)
