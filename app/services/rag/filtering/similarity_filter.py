"""Similarity-threshold relevance filter.

Rule (explicitly chosen and tested): a candidate is ACCEPTED when its cosine
similarity is ``>= threshold``. A candidate exactly at the threshold is kept.
Everything below is rejected with reason ``below_similarity_threshold``.

The filter never re-orders candidates: the index already returns them sorted
by descending similarity, and Day 23 deliberately does not add a reranker.
"""
from __future__ import annotations

from app.schemas.day22 import RetrievedChunk
from app.schemas.day23 import REASON_BELOW_THRESHOLD
from app.services.rag.filtering.base import FilteredCandidate


class SimilarityFilter:
    """Keep candidates whose similarity is at or above ``threshold``."""

    def __init__(self, threshold: float) -> None:
        threshold = float(threshold)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("similarity threshold must be within [0, 1]")
        self.threshold = threshold

    def filter(self, candidates: list[RetrievedChunk]) -> list[FilteredCandidate]:
        results: list[FilteredCandidate] = []
        for candidate in candidates:
            accepted = float(candidate.similarity) >= self.threshold
            results.append(
                FilteredCandidate(
                    candidate=candidate,
                    accepted=accepted,
                    reason=None if accepted else REASON_BELOW_THRESHOLD,
                )
            )
        return results
