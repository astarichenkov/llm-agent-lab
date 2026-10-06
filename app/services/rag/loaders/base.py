"""Loader interface shared by every knowledge source.

A loader turns a concrete source (a PDF file, a Markdown file, a Telegram
export) into the normalized :class:`~app.services.rag.models.Document` model
and reports scan/load statistics. Adding a new source type (HTML, DOCX, ...)
only requires implementing this protocol and registering the loader.
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from app.services.rag.models import Document, LoaderStats


@runtime_checkable
class DocumentLoader(Protocol):
    """Protocol every source loader implements.

    ``scan()`` must be cheap (no full text extraction) and is used by the
    ``sources`` command; ``load()`` performs the real parsing and returns the
    normalized documents.
    """

    source_type: str

    def scan(self) -> LoaderStats:
        """Return source statistics without heavy parsing."""

    def load(self) -> Iterable[Document]:
        """Yield normalized documents for this source."""

    def stats(self) -> LoaderStats:
        """Return statistics collected by the most recent ``load()`` call."""


class LoaderError(Exception):
    """Raised when a source cannot be read at all."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
