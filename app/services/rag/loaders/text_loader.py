"""Plain-text (TXT) and Markdown (MD) loader.

Each file becomes one :class:`Document`. Markdown headings are used to set the
``section`` metadata for the paragraphs they introduce; TXT files are split on
blank lines into paragraph segments. No source file is ever modified.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

from app.services.rag.loaders.base import LoaderError
from app.services.rag.models import Document, LoaderStats, Segment

logger = logging.getLogger("app.services.rag.loaders.text")

TEXT_SUFFIXES = (".txt", ".md", ".markdown")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _relative(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.name


class TextLoader:
    """Load ``.txt`` / ``.md`` files from a directory tree."""

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
            return [self.root] if self.root.suffix.lower() in TEXT_SUFFIXES else []
        found: list[Path] = []
        for suffix in TEXT_SUFFIXES:
            found.extend(self.root.rglob(f"*{suffix}"))
        return sorted({p for p in found if p.is_file()})

    # ------------------------------------------------------------------
    # scan (cheap)
    # ------------------------------------------------------------------
    def scan(self) -> LoaderStats:
        stats = LoaderStats(source_type=self.source_type)
        stats.files = len(self._files())
        return stats

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load(self) -> Iterable[Document]:
        stats = LoaderStats(source_type=self.source_type)
        files = self._files()
        stats.files = len(files)
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:  # pragma: no cover - filesystem dependent
                logger.warning("Text loader: cannot read %s: %s", path, exc)
                continue
            document = self._build_document(path, text)
            if document.text.strip():
                yield document
        self._stats = stats

    def stats(self) -> LoaderStats:
        return self._stats

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_document(self, path: Path, text: str) -> Document:
        source = _relative(path, self.root if self.root.is_dir() else None)
        title = path.stem
        is_markdown = path.suffix.lower() in (".md", ".markdown")
        if is_markdown:
            segments, first_heading = self._markdown_segments(text)
            if first_heading:
                title = first_heading
        else:
            segments = self._paragraph_segments(text)

        full_text = "\n\n".join(seg.text for seg in segments if seg.text.strip())
        metadata = {
            "source_type": self.source_type,
            "source": source,
            "filename": path.name,
            "title": title,
            "locator": source,
        }
        return Document(text=full_text, metadata=metadata, segments=segments)

    @staticmethod
    def _paragraph_segments(text: str) -> list[Segment]:
        blocks = re.split(r"\n\s*\n", text)
        segments: list[Segment] = []
        for block in blocks:
            cleaned = block.strip()
            if cleaned:
                segments.append(Segment(text=cleaned, metadata={}))
        return segments

    @staticmethod
    def _markdown_segments(text: str) -> tuple[list[Segment], str | None]:
        segments: list[Segment] = []
        current_section: str | None = None
        first_heading: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            block = "\n".join(buffer).strip()
            buffer.clear()
            if not block:
                return
            metadata: dict[str, object] = {}
            if current_section:
                metadata["section"] = current_section
            segments.append(Segment(text=block, metadata=metadata))

        for line in text.splitlines():
            match = _HEADING_RE.match(line)
            if match:
                flush()
                current_section = match.group(2).strip() or current_section
                if first_heading is None:
                    first_heading = current_section
                continue
            buffer.append(line)
        flush()
        return segments, first_heading
