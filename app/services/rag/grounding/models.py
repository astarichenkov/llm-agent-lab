"""Internal models for the Day 24 grounding components.

These small dataclasses describe the *decisions* the grounding layer makes.
They are deliberately separate from the Pydantic API/DB schemas so the gate
and the validator can be unit-tested without constructing HTTP responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroundingGateDecision:
    """Outcome of the "can we answer from the retrieved context?" check.

    The gate runs BEFORE the generation provider and, when it fails, the
    caller must return a deterministic refusal without calling the LLM.
    """

    passed: bool
    reason: str
    best_score: float | None
    required: float
    accepted_count: int
    context_count: int


@dataclass
class ValidatedCitations:
    """Result of validating the LLM-provided evidence list."""

    citations: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        return bool(self.citations) and not self.errors

    @property
    def has_quotes(self) -> bool:
        return any((c.quote or "").strip() for c in self.citations)

    @property
    def has_sources(self) -> bool:
        return any((c.chunk_id or "").strip() for c in self.citations)
