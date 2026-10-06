"""Pydantic schemas for Week 5 / Day 22 — first RAG request.

These models are the single contract shared by the application service
(``RagAnswerService``), the HTTP endpoints and the UI. The service never
returns a bare string: callers always receive a structured result so Day 23+
can build citations, evaluation and agent workflows on top.

Two clearly separated roles appear throughout:

* ``embedding_model`` -- ``bge-m3`` (Ollama), used ONLY for query/doc vectors;
* ``model``           -- the generation LLM (DeepSeek/Ollama chat), used for
  both No-RAG and RAG. The comparison is ONLY the presence/absence of context.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MODE_NO_RAG = "no_rag"
MODE_RAG = "rag"

# Manual grading vocabulary for the evaluation tab (Day 22 is manual only).
GRADE_PASS = "pass"
GRADE_PARTIAL = "partial"
GRADE_FAIL = "fail"
GRADE_NONE = "none"
GRADES = (GRADE_PASS, GRADE_PARTIAL, GRADE_FAIL)

DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 20
MAX_QUESTION_LENGTH = 2000


# ----------------------------------------------------------------------
# Retrieval + answer
# ----------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """One retrieved chunk exposed to the UI.

    ``metadata`` is passed through verbatim from the Day 21 index; the UI
    renders only fields that are actually present (a manual chunk has no
    ``message_ids`` and a Telegram chunk has no ``page``).
    """

    rank: int
    similarity: float
    chunk_id: str
    doc_id: str = ""
    source_type: str
    source: str = ""
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGAnswer(BaseModel):
    """A single answer produced in one mode (No-RAG or RAG)."""

    question: str
    mode: Literal["no_rag", "rag"]
    answer: str
    model: str
    generation_provider: str
    temperature: float
    top_k: int
    retrieval_performed: bool
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    embedding_model: str = ""
    index_path: str = ""
    chunking: str = ""
    finish_reason: str | None = None
    pipeline: list[str] = Field(default_factory=list)


class RAGComparison(BaseModel):
    """Both answers for the SAME question and the SAME generation model."""

    question: str
    top_k: int
    model: str
    temperature: float
    no_rag: RAGAnswer
    rag: RAGAnswer


# ----------------------------------------------------------------------
# API requests
# ----------------------------------------------------------------------
class Day22AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    mode: Literal["no_rag", "rag"] = MODE_RAG
    top_k: int = Field(default=DEFAULT_TOP_K, ge=MIN_TOP_K, le=MAX_TOP_K)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class Day22CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=MIN_TOP_K, le=MAX_TOP_K)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


# ----------------------------------------------------------------------
# Status (technical block in the UI)
# ----------------------------------------------------------------------
class Day22StatusResponse(BaseModel):
    generation_provider: str
    generation_model: str
    temperature: float
    embedding_model: str
    index_path: str
    chunking: str
    default_top_k: int
    top_k_options: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    index_exists: bool
    chunks: int = 0
    chunks_by_source_type: dict[str, int] = Field(default_factory=dict)


# ----------------------------------------------------------------------
# Evaluation dataset
# ----------------------------------------------------------------------
class ExpectedSource(BaseModel):
    """A source the question is expected to retrieve.

    Telegram chunks cover a conversation region, so ``message_ids`` is a
    superset of the exact message; the UI only requires an overlap.
    """

    source_type: str
    source: str | None = None
    page: int | None = None
    message_ids: list[int] | None = None


class EvaluationQuestion(BaseModel):
    id: str
    question: str
    category: str
    expected_facts: list[str] = Field(default_factory=list)
    expected_sources: list[ExpectedSource] = Field(default_factory=list)


class EvaluationGradeRecord(BaseModel):
    """Manual PASS/PARTIAL/FAIL grading for one control question."""

    no_rag: Literal["pass", "partial", "fail"] | None = None
    rag: Literal["pass", "partial", "fail"] | None = None
    expected_source_retrieved: bool | None = None


class EvaluationSummary(BaseModel):
    """Counts over ONLY the grades that were actually assigned."""

    total_questions: int
    evaluated: int
    no_rag: dict[str, int] = Field(default_factory=dict)
    rag: dict[str, int] = Field(default_factory=dict)
    expected_source_retrieved: int = 0


class EvaluationResultsResponse(BaseModel):
    results: dict[str, EvaluationGradeRecord] = Field(default_factory=dict)
    summary: EvaluationSummary


class EvaluationRunRequest(BaseModel):
    top_k: int = Field(default=DEFAULT_TOP_K, ge=MIN_TOP_K, le=MAX_TOP_K)


class EvaluationRunResponse(BaseModel):
    question: EvaluationQuestion
    comparison: RAGComparison


class EvaluationRunAllResponse(BaseModel):
    runs: list[EvaluationRunResponse] = Field(default_factory=list)
