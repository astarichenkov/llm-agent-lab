"""Pydantic schemas for Week 5 / Day 23 — Query Rewrite + relevance filtering.

Day 23 keeps the Day 22 baseline pipeline untouched and adds an *improved*
pipeline that runs a wider candidate search before filtering:

    baseline:  question -> vector search -> Top-5 -> context -> LLM
    improved:  question -> rewrite -> vector search -> Top-N (retrieval_top_k)
                        -> similarity threshold -> Top-5 (final_top_k)
                        -> context -> LLM

The models here are the single contract shared by the application service,
the HTTP endpoints and the UI. Retrieval candidates keep their FULL trace
(accepted *and* rejected) so the UI can explain every filtering decision.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.day22 import EvaluationQuestion, RAGAnswer, RetrievedChunk

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MODE_BASELINE = "baseline"
MODE_IMPROVED = "improved"

DEFAULT_RETRIEVAL_TOP_K = 20
MIN_RETRIEVAL_TOP_K = 1
MAX_RETRIEVAL_TOP_K = 50

DEFAULT_FINAL_TOP_K = 5
MIN_FINAL_TOP_K = 1
MAX_FINAL_TOP_K = 20

# Data-driven default (see docs/week5/day23.md): on the 10 Day 22 questions the
# lowest best-expected-source score was 0.5102, so 0.50 keeps every expected
# source while dropping the long tail of weak candidates.
DEFAULT_SIMILARITY_THRESHOLD = 0.50
MIN_SIMILARITY_THRESHOLD = 0.0
MAX_SIMILARITY_THRESHOLD = 1.0

# Rejection reasons (kept stable — the UI and tests depend on them).
REASON_BELOW_THRESHOLD = "below_similarity_threshold"

# Per-candidate status shown in the UI.
STATUS_USED = "used"
STATUS_RELEVANT_NOT_USED = "relevant_not_used"
STATUS_BELOW_THRESHOLD = "below_threshold"

MAX_QUESTION_LENGTH = 2000


# ----------------------------------------------------------------------
# Query rewrite
# ----------------------------------------------------------------------
class RewriteInfo(BaseModel):
    """Outcome of the query-rewrite step (written only if it was attempted)."""

    original_query: str
    rewritten_query: str
    applied: bool
    error: str | None = None
    provider: str = ""
    model: str = ""


# ----------------------------------------------------------------------
# Retrieval candidates with their filtering outcome
# ----------------------------------------------------------------------
class ImprovedCandidate(RetrievedChunk):
    """A retrieved chunk plus the reason it was (not) sent to the LLM.

    ``status`` is one of:

    * ``used``               — passed the threshold and fits in ``final_top_k``;
    * ``relevant_not_used``  — passed the threshold but lost to ``final_top_k``;
    * ``below_threshold``    — rejected by the similarity filter.
    """

    accepted: bool
    used: bool = False
    reason: str | None = None
    status: Literal["used", "relevant_not_used", "below_threshold"]


class ImprovedRAGResult(BaseModel):
    """Full trace of one improved RAG request."""

    original_question: str
    rewritten_query: str
    rewrite: RewriteInfo

    retrieval_top_k: int
    similarity_threshold: float
    final_top_k: int

    retrieved_candidates: list[ImprovedCandidate] = Field(default_factory=list)
    accepted_candidates: list[ImprovedCandidate] = Field(default_factory=list)
    rejected_candidates: list[ImprovedCandidate] = Field(default_factory=list)
    context_chunks: list[ImprovedCandidate] = Field(default_factory=list)

    answer: str
    no_relevant_context: bool = False

    model: str
    generation_provider: str
    temperature: float
    embedding_model: str = ""
    index_path: str = ""
    chunking: str = ""
    finish_reason: str | None = None
    pipeline: list[str] = Field(default_factory=list)

    @property
    def retrieved_count(self) -> int:
        return len(self.retrieved_candidates)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_candidates)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_candidates)


class Day23Comparison(BaseModel):
    """Baseline (Day 22 RAG) vs Improved (Day 23) for the SAME question."""

    question: str
    model: str
    temperature: float
    baseline_top_k: int
    retrieval_top_k: int
    final_top_k: int
    similarity_threshold: float
    baseline: RAGAnswer
    improved: ImprovedRAGResult


# ----------------------------------------------------------------------
# API requests / status
# ----------------------------------------------------------------------
def _strip_question(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError("question must not be blank")
    return value


class Day23Settings(BaseModel):
    """The three tunable knobs of the improved pipeline.

    ``similarity_threshold`` is accepted from the client ONLY within its
    documented range; ``model_validator`` enforces ``final_top_k`` <=
    ``retrieval_top_k`` so the UI can never ask for a wider final context than
    the candidate set.
    """

    retrieval_top_k: int = Field(
        default=DEFAULT_RETRIEVAL_TOP_K,
        ge=MIN_RETRIEVAL_TOP_K,
        le=MAX_RETRIEVAL_TOP_K,
    )
    final_top_k: int = Field(
        default=DEFAULT_FINAL_TOP_K,
        ge=MIN_FINAL_TOP_K,
        le=MAX_FINAL_TOP_K,
    )
    similarity_threshold: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD,
        ge=MIN_SIMILARITY_THRESHOLD,
        le=MAX_SIMILARITY_THRESHOLD,
    )

    @model_validator(mode="after")
    def _check_final_le_retrieval(self) -> "Day23Settings":
        if self.final_top_k > self.retrieval_top_k:
            raise ValueError("final_top_k must be <= retrieval_top_k")
        return self


class Day23AskRequest(Day23Settings):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        return _strip_question(value)


class Day23CompareRequest(Day23AskRequest):
    """Compare baseline vs improved for one question (same settings)."""


class Day23StatusResponse(BaseModel):
    generation_provider: str
    generation_model: str
    temperature: float
    embedding_model: str
    index_path: str
    chunking: str
    default_top_k: int
    default_retrieval_top_k: int
    default_final_top_k: int
    default_similarity_threshold: float
    min_similarity_threshold: float
    max_similarity_threshold: float
    retrieval_top_k_max: int
    final_top_k_max: int
    rewrite_provider: str
    rewrite_model: str
    recommended_demo_question: str
    index_exists: bool
    chunks: int = 0
    chunks_by_source_type: dict[str, int] = Field(default_factory=dict)


# ----------------------------------------------------------------------
# Evaluation (same 10 questions, baseline vs improved)
# ----------------------------------------------------------------------
class Day23EvaluationGradeRecord(BaseModel):
    """Manual, independent grading for baseline and improved pipelines."""

    baseline: Literal["pass", "partial", "fail"] | None = None
    improved: Literal["pass", "partial", "fail"] | None = None
    baseline_expected_source_retrieved: bool | None = None
    improved_expected_source_retrieved: bool | None = None


class Day23EvaluationSummary(BaseModel):
    total_questions: int
    evaluated: int
    baseline: dict[str, int] = Field(default_factory=dict)
    improved: dict[str, int] = Field(default_factory=dict)
    baseline_expected_source_retrieved: int = 0
    improved_expected_source_retrieved: int = 0


class Day23EvaluationResultsResponse(BaseModel):
    results: dict[str, Day23EvaluationGradeRecord] = Field(default_factory=dict)
    summary: Day23EvaluationSummary


class Day23EvaluationRunResponse(BaseModel):
    question: EvaluationQuestion
    comparison: Day23Comparison
    baseline_expected_source_retrieved: bool
    improved_expected_source_retrieved: bool


class Day23EvaluationRunAllResponse(BaseModel):
    runs: list[Day23EvaluationRunResponse] = Field(default_factory=list)


class Day23ScoreRow(BaseModel):
    """Per-question similarity statistics used to justify the threshold."""

    id: str
    question: str
    category: str
    top1: float | None = None
    top5_min: float | None = None
    top_retrieval_min: float | None = None
    best_expected_score: float | None = None
    best_expected_rank: int | None = None
    expected_chunk_count: int = 0
    matched_at_threshold: int = 0
    noise_count: int = 0


class Day23ScoreReport(BaseModel):
    retrieval_top_k: int
    similarity_threshold: float
    rows: list[Day23ScoreRow] = Field(default_factory=list)
    best_expected_min: float | None = None
    best_expected_avg: float | None = None
    expected_kept: int = 0
    expected_total: int = 0
    noise_removed: int = 0
    question_best_expected_kept: int = 0
    question_count: int = 0
