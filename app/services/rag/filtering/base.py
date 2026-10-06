"""Relevance-filter contract for the Day 23 improved retrieval pipeline.

A filter receives the full candidate list (already sorted by similarity) and
decides, for every candidate, whether it is relevant enough to keep. The
decision is data only — no mutation of the candidate — so the caller can keep
the complete accepted/rejected trace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.schemas.day22 import RetrievedChunk


@dataclass
class FilteredCandidate:
    """One candidate plus the filter verdict."""

    candidate: RetrievedChunk
    accepted: bool
    reason: str | None = None


@runtime_checkable
class RelevanceFilter(Protocol):
    """Minimal interface every relevance filter implements."""

    threshold: float

    def filter(self, candidates: list[RetrievedChunk]) -> list[FilteredCandidate]: ...
