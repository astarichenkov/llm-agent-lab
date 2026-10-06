"""PDF loader built on ``pypdf``.

Each PDF file becomes one :class:`Document`; every text block (paragraph) is a
:class:`Segment` carrying its page number and, when it can be detected
reliably, its section. The running section is carried across pages so a
heading on page *N* still labels the paragraphs on page *N+1*.

The original PDFs are only ever opened for reading — never rewritten.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

from app.services.rag.loaders.base import LoaderError
from app.services.rag.models import Document, LoaderStats, Segment

logger = logging.getLogger("app.services.rag.loaders.pdf")

PDF_SUFFIXES = (".pdf",)
_MAX_HEADING_CHARS = 70
_MAX_HEADING_WORDS = 9


def _relative(path: Path, root: Path | None) -> str:
    if root is not None and root.is_dir():
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.name


def _looks_like_heading(block: str) -> bool:
    """Conservative heading heuristic for extracted PDF text.

    Extraction mangles layout, so this is intentionally simple. It never
    guesses a section when the evidence is weak — a wrong guess is worse than
    an absent one.
    """
    text = " ".join(block.split())
    if not text or len(text) > _MAX_HEADING_CHARS:
        return False
    words = text.split()
    if len(words) > _MAX_HEADING_WORDS:
        return False
    if text.endswith((".", "。", ":", ";", ",", "!", "?")):
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    starts_numbered = text[0].isdigit()
    title_like = len(words) <= 5 or text.isupper()
    return starts_numbered or title_like


class PdfLoader:
    """Load PDF files from a directory (or a single file)."""

    source_type = "manual"

    def __init__(self, root: str | Path, *, source_type: str = "manual") -> None:
        self.root = Path(root)
        self.source_type = source_type
        self._stats = LoaderStats(source_type=source_type)

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    def _files(self) -> list[Path]:
        if not self.root.exists():
            return []
        if self.root.is_file():
            return [self.root] if self.root.suffix.lower() in PDF_SUFFIXES else []
        found = [p for p in self.root.rglob("*") if p.suffix.lower() in PDF_SUFFIXES]
        return sorted(p for p in found if p.is_file())

    # ------------------------------------------------------------------
    # scan (cheap: page counts only)
    # ------------------------------------------------------------------
    def scan(self) -> LoaderStats:
        from pypdf import PdfReader

        stats = LoaderStats(source_type=self.source_type)
        files = self._files()
        stats.files = len(files)
        for path in files:
            try:
                stats.pages += len(PdfReader(str(path)).pages)
            except Exception as exc:  # noqa: BLE001 - malformed PDF must not crash
                logger.warning("PDF scan failed for %s: %s", path, exc)
        return stats

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load(self) -> Iterable[Document]:
        from pypdf import PdfReader

        stats = LoaderStats(source_type=self.source_type)
        files = self._files()
        stats.files = len(files)
        for path in files:
            try:
                reader = PdfReader(str(path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("PDF loader: cannot open %s: %s", path, exc)
                continue
            document, file_stats = self._build_document(path, reader)
            stats.pages += file_stats.pages
            stats.pages_with_text += file_stats.pages_with_text
            if document is not None:
                yield document
        # Preserve the last file's page counters? No: pages/pages_with_text
        # are aggregate; rebuild them in the loop above.
        self._stats = stats

    def stats(self) -> LoaderStats:
        return self._stats

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_document(self, path: Path, reader) -> tuple[Document | None, LoaderStats]:
        file_stats = LoaderStats(source_type=self.source_type, files=1)
        source = _relative(path, self.root)
        segments: list[Segment] = []
        page_count = len(reader.pages)
        file_stats.pages = page_count
        current_section: str | None = None
        first_page_title: str | None = None

        for index, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop the file
                logger.warning("PDF page extraction failed %s p%d: %s", path.name, index, exc)
                page_text = ""
            if page_text.strip():
                file_stats.pages_with_text += 1
            blocks = self._blocks(page_text)
            for block in blocks:
                if _looks_like_heading(block):
                    current_section = " ".join(block.split())
                    if first_page_title is None:
                        first_page_title = current_section
                    metadata: dict[str, object] = {
                        "page": index,
                        "is_heading": True,
                    }
                    if current_section:
                        metadata["section"] = current_section
                    segments.append(Segment(text=block, metadata=metadata))
                else:
                    metadata = {"page": index}
                    if current_section:
                        metadata["section"] = current_section
                    segments.append(Segment(text=block, metadata=metadata))

        # Drop empty segments (scanned pages without a text layer).
        segments = [s for s in segments if s.text.strip()]
        if not segments:
            return None, file_stats

        full_text = "\n\n".join(seg.text for seg in segments)
        metadata = {
            "source_type": self.source_type,
            "source": source,
            "filename": path.name,
            "title": first_page_title or path.stem,
            "locator": source,
        }
        return Document(text=full_text, metadata=metadata, segments=segments), file_stats

    @staticmethod
    def _blocks(page_text: str) -> list[str]:
        """Split a page into paragraph-ish blocks."""
        text = (page_text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            return []
        # Primary split: blank lines.
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        if len(blocks) > 1:
            return blocks
        # Fallback: merge non-empty lines into blocks of a few lines so that a
        # page without blank lines is not one giant segment.
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        merged: list[str] = []
        buffer: list[str] = []
        for line in lines:
            buffer.append(line)
            if len(buffer) >= 3:
                merged.append("\n".join(buffer))
                buffer = []
        if buffer:
            merged.append("\n".join(buffer))
        return merged
