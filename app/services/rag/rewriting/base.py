"""Query-rewrite contract for the Day 23 improved retrieval pipeline.

The rewriter turns a natural-language user question into a *search query*
suitable for the Mitsubishi Xpander knowledge base. It is used ONLY for
retrieval; the generation LLM always answers the ORIGINAL question.

Every implementation must degrade gracefully: a rewrite failure is not an
error condition, it simply returns the original query with ``applied=False``
and an explanation in ``error``. This keeps the demo working when the
generation provider is temporarily unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RewriteResult:
    """Result of one rewrite attempt (successful or fallback)."""

    original_query: str
    rewritten_query: str
    applied: bool
    error: str | None = None
    provider: str = ""
    model: str = ""


@runtime_checkable
class QueryRewriter(Protocol):
    """Minimal interface every query rewriter implements."""

    name: str
    model: str

    async def rewrite(self, question: str) -> RewriteResult: ...
