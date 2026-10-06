"""Day 24 — Grounded RAG, grounding gate and citation validation tests.

No test contacts Ollama, DeepSeek or a real SQLite index:

* retrieval is the Day 22 ``FakeDay22RagService`` (in-memory hits);
* generation is ``FakeGroundedGeneration`` (returns structured JSON);
* rewrite is ``FakeQueryRewriter`` (passthrough for Day 24);
* the HTTP layer uses the shared ``client`` fixture.

The test names follow the Day 24 checklist (gate -> evidence -> validator).
"""
from __future__ import annotations

import asyncio

import pytest

from conftest import FakeGroundedGeneration, FakeQueryRewriter
from app.config import Settings
from app.schemas.day22 import RetrievedChunk
from app.schemas.day24 import (
    ERR_ANSWER_WITHOUT_EVIDENCE,
    ERR_CHUNK_ID_NOT_FOUND,
    ERR_CITATION_NOT_IN_CONTEXT,
    ERR_EMPTY_QUOTE,
    ERR_INVALID_MODEL_OUTPUT,
    ERR_QUOTE_NOT_IN_CHUNK,
    GATE_NO_ACCEPTED_CONTEXT,
    GATE_PASSED,
    STATUS_ANSWERED,
    STATUS_GROUNDING_FAILED,
    STATUS_INSUFFICIENT_CONTEXT,
    Day24NegativeQuestion,
)
from app.services.rag.answer_service import RagAnswerService
from app.services.rag.grounding.service import (
    GroundedRagService,
    extract_json_object,
)
from app.services.rag.grounding.validator import CitationValidator, normalize_whitespace


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _chunk(chunk_id, text, source_type="manual", **meta):
    metadata = {"source_type": source_type}
    metadata.update(meta)
    return RetrievedChunk(
        rank=1,
        similarity=0.9,
        chunk_id=chunk_id,
        source_type=source_type,
        source=str(meta.get("source", "doc.pdf")),
        text=text,
        metadata=metadata,
    )


MANUAL_TEXT = "Давление в шинах 205/55R16: 2,1 бар (210 кПа)."
TELEGRAM_TEXT = "Владелец связал вспучивание пластика с пеной для мойки."


def _manual_chunk():
    return _chunk(
        "manual-chunk-1",
        MANUAL_TEXT,
        source_type="manual",
        source="20_XPANDER_RU1.pdf",
        page=238,
        section="Шины",
    )


def _telegram_chunk():
    return _chunk(
        "telegram-chunk-1",
        TELEGRAM_TEXT,
        source_type="telegram",
        source="result.json",
        chat_name="Xpander Club",
        message_ids=[83159, 83165],
        date_from="2026-09-22T12:08:00",
        date_to="2026-09-22T14:20:00",
    )


# ----------------------------------------------------------------------
# 1. Grounded RAG reuses the Day 23 improved retrieval
# ----------------------------------------------------------------------
async def test_grounded_reuses_improved_retrieval(
    day24_service, day22_rag_service
) -> None:
    result = await day24_service.ask("Проверка", retrieval_top_k=7, final_top_k=3)
    assert result.status == STATUS_ANSWERED
    # The retrieval used the rewritten (passthrough) query and retrieval_top_k.
    assert day22_rag_service.search_calls
    assert day22_rag_service.search_calls[0][1] == 7
    assert result.retrieval.retrieval_top_k == 7
    assert result.retrieval.final_top_k == 3
    assert result.retrieval.context_count >= 1


# ----------------------------------------------------------------------
# 2-4. Grounding gate
# ----------------------------------------------------------------------
async def test_grounded_merges_original_and_rewritten_retrieval(
    settings: Settings,
) -> None:
    """A lossy rewrite must not hide a chunk the original question finds."""
    from app.services.rag.models import SearchHit

    def _hit(score, chunk_id, text):
        return SearchHit(
            score=score,
            chunk_id=chunk_id,
            text=text,
            metadata={"source_type": "telegram", "source": "result.json"},
            doc_id="doc-" + chunk_id,
        )

    class QueryAwareRag:
        def __init__(self):
            self.calls = []

        def index_exists(self):
            return True

        def stats(self):
            return {"chunks": 2, "index_path": "x", "exists": True,
                    "embedding_model": "bge-m3", "chunking": "structural",
                    "chunks_by_source_type": {"telegram": 2}}

        def search(self, query, *, top_k=5, source_type=None):
            self.calls.append(query)
            if query == "переформулированный запрос":
                return [_hit(0.90, "rewritten-only", "rewritten text")]
            return [_hit(0.95, "original-only", "original text")]

    rag = QueryAwareRag()
    service = RagAnswerService(
        settings,
        rag_service=rag,
        generation=FakeGroundedGeneration(),
        rewriter=FakeQueryRewriter("переформулированный запрос"),
    )
    merged = await service.prepare_improved_retrieval(
        "исходный вопрос", merge_original_query=True, similarity_threshold=0.0
    )
    assert {c.chunk_id for c in merged.retrieved_candidates} == {
        "rewritten-only",
        "original-only",
    }

    plain = await service.prepare_improved_retrieval(
        "исходный вопрос", merge_original_query=False, similarity_threshold=0.0
    )
    assert {c.chunk_id for c in plain.retrieved_candidates} == {"rewritten-only"}


async def test_gate_accepts_sufficient_context(day24_service) -> None:
    result = await day24_service.ask("Давление в шинах?")
    assert result.retrieval.gate_reason == GATE_PASSED
    assert result.retrieval.gate_passed is True
    assert result.status == STATUS_ANSWERED


async def test_gate_rejects_weak_context(day24_service) -> None:
    result = await day24_service.ask(
        "Как выполняется адаптация вариатора Xpander сканером?"
    )
    assert result.status == STATUS_INSUFFICIENT_CONTEXT
    assert result.retrieval.gate_passed is False
    assert result.retrieval.gate_reason == GATE_NO_ACCEPTED_CONTEXT
    assert result.sources == []
    assert result.citations == []


async def test_generation_not_called_when_gate_rejects(
    day24_service, day24_generation
) -> None:
    result = await day24_service.ask(
        "Как выполняется адаптация вариатора Xpander сканером?"
    )
    assert result.llm_called is False
    assert day24_generation.generate_call_count == 0
    assert result.message  # the deterministic refusal message is present
    assert result.clarification_request


# ----------------------------------------------------------------------
# 5. Answered result requires evidence
# ----------------------------------------------------------------------
async def test_answer_without_evidence_is_grounding_failure(
    day24_service, day24_generation
) -> None:
    day24_generation.payload = {
        "status": "answered",
        "answer": "Ответ без доказательств.",
        "evidence": [],
    }
    result = await day24_service.ask("Давление в шинах?")
    assert result.status == STATUS_GROUNDING_FAILED
    assert result.grounding_valid is False
    assert any(ERR_ANSWER_WITHOUT_EVIDENCE in e for e in result.validation_errors)


# ----------------------------------------------------------------------
# 6-12. Citation validator rules (pure unit tests)
# ----------------------------------------------------------------------
def test_valid_chunk_id_and_quote_passes() -> None:
    validator = CitationValidator([_manual_chunk()])
    outcome = validator.validate(
        [{"chunk_id": "manual-chunk-1", "quote": MANUAL_TEXT}]
    )
    assert outcome.all_valid
    assert outcome.citations[0].quote_valid is True


def test_unknown_chunk_id_rejected() -> None:
    validator = CitationValidator(
        [_manual_chunk()], known_chunk_ids={"manual-chunk-1", "other-chunk"}
    )
    outcome = validator.validate([{"chunk_id": "invented-id", "quote": MANUAL_TEXT}])
    assert not outcome.all_valid
    assert ERR_CHUNK_ID_NOT_FOUND in outcome.citations[0].validation_errors


def test_chunk_not_in_context_rejected() -> None:
    validator = CitationValidator(
        [_manual_chunk()], known_chunk_ids={"manual-chunk-1", "other-chunk"}
    )
    outcome = validator.validate([{"chunk_id": "other-chunk", "quote": MANUAL_TEXT}])
    assert not outcome.all_valid
    assert ERR_CITATION_NOT_IN_CONTEXT in outcome.citations[0].validation_errors


def test_whitespace_normalization_passes() -> None:
    validator = CitationValidator([_manual_chunk()])
    outcome = validator.validate(
        [
            {
                "chunk_id": "manual-chunk-1",
                "quote": "  Давление   в шинах\n205/55R16:   2,1 бар (210 кПа).  ",
            }
        ]
    )
    assert outcome.all_valid
    assert normalize_whitespace("a\r\n b") == "a b"


def test_invented_quote_rejected() -> None:
    validator = CitationValidator([_manual_chunk()])
    outcome = validator.validate(
        [{"chunk_id": "manual-chunk-1", "quote": "Давление 3,5 бар"}]
    )
    assert not outcome.all_valid
    assert ERR_QUOTE_NOT_IN_CHUNK in outcome.citations[0].validation_errors


def test_empty_quote_rejected() -> None:
    validator = CitationValidator([_manual_chunk()])
    outcome = validator.validate([{"chunk_id": "manual-chunk-1", "quote": "   "}])
    assert not outcome.all_valid
    assert ERR_EMPTY_QUOTE in outcome.citations[0].validation_errors


def test_multiple_valid_citations_pass() -> None:
    validator = CitationValidator([_manual_chunk(), _telegram_chunk()])
    outcome = validator.validate(
        [
            {"chunk_id": "manual-chunk-1", "quote": MANUAL_TEXT},
            {"chunk_id": "telegram-chunk-1", "quote": TELEGRAM_TEXT},
        ]
    )
    assert outcome.all_valid
    assert len(outcome.citations) == 2


# ----------------------------------------------------------------------
# 13-16. Metadata source of truth is the retrieved chunk
# ----------------------------------------------------------------------
def test_source_metadata_comes_from_chunk() -> None:
    validator = CitationValidator([_manual_chunk()])
    outcome = validator.validate(
        [
            {
                "chunk_id": "manual-chunk-1",
                "quote": MANUAL_TEXT,
                # Fake metadata the LLM might have invented — must be ignored.
                "source": "fake.pdf",
                "page": 999,
                "section": "Fake section",
            }
        ]
    )
    citation = outcome.citations[0]
    assert citation.source == "20_XPANDER_RU1.pdf"
    assert citation.page == 238
    assert citation.section == "Шины"


def test_manual_metadata_shapes_correctly() -> None:
    validator = CitationValidator([_manual_chunk()])
    citation = validator.validate(
        [{"chunk_id": "manual-chunk-1", "quote": MANUAL_TEXT}]
    ).citations[0]
    assert citation.source_type == "manual"
    assert citation.as_source().chunk_id == "manual-chunk-1"
    assert citation.as_source().page == 238


def test_telegram_metadata_shapes_correctly() -> None:
    validator = CitationValidator([_telegram_chunk()])
    citation = validator.validate(
        [{"chunk_id": "telegram-chunk-1", "quote": TELEGRAM_TEXT}]
    ).citations[0]
    assert citation.source_type == "telegram"
    assert citation.message_ids == [83159, 83165]
    assert citation.chat_name == "Xpander Club"
    assert citation.as_source().message_ids == [83159, 83165]


def test_telegram_is_not_marked_as_official_manual() -> None:
    validator = CitationValidator([_telegram_chunk()])
    citation = validator.validate(
        [{"chunk_id": "telegram-chunk-1", "quote": TELEGRAM_TEXT}]
    ).citations[0]
    assert citation.source_type != "manual"
    assert citation.as_source().source_type == "telegram"


# ----------------------------------------------------------------------
# 17-19. Service-level citation failure modes
# ----------------------------------------------------------------------
async def test_invented_quote_in_service_fails(
    day24_service, day24_generation
) -> None:
    day24_generation.payload = {
        "status": "answered",
        "answer": "Ответ с выдуманной цитатой.",
        "evidence": [{"chunk_id": "manual-chunk-1", "quote": "выдуманная цитата"}],
    }
    result = await day24_service.ask("Давление в шинах?")
    assert result.status == STATUS_GROUNDING_FAILED
    assert result.grounding_valid is False
    assert result.llm_called is True


async def test_unknown_chunk_in_service_fails(
    day24_service, day24_generation
) -> None:
    day24_generation.payload = {
        "status": "answered",
        "answer": "Ответ с неизвестным chunk_id.",
        "evidence": [{"chunk_id": "invented-id", "quote": MANUAL_TEXT}],
    }
    result = await day24_service.ask("Давление в шинах?")
    assert result.status == STATUS_GROUNDING_FAILED
    assert result.validation_errors


async def test_invalid_json_in_service_fails(
    day24_service, day24_generation
) -> None:
    day24_generation.payload = "not-json at all"
    result = await day24_service.ask("Давление в шинах?")
    assert result.status == STATUS_GROUNDING_FAILED
    assert ERR_INVALID_MODEL_OUTPUT in result.validation_errors
    assert result.llm_called is True


def test_extract_json_object_handles_fences() -> None:
    assert extract_json_object('```json\n{"status": "answered"}\n```') == {
        "status": "answered"
    }
    assert extract_json_object("prefix {\"a\": 1} suffix") == {"a": 1}
    assert extract_json_object("no json") is None


# ----------------------------------------------------------------------
# 20-22. HTTP API
# ----------------------------------------------------------------------
def test_api_grounded_ask(client) -> None:
    response = client.post(
        "/api/week5/day24/ask",
        json={"question": "Давление в шинах?", "similarity_threshold": 0.5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == STATUS_ANSWERED
    assert body["grounding_valid"] is True
    assert len(body["sources"]) >= 1
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["quote_valid"] is True


def test_api_insufficient_context(client) -> None:
    response = client.post(
        "/api/week5/day24/ask",
        json={
            "question": "Как выполняется адаптация вариатора Xpander сканером?",
            "similarity_threshold": 0.5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == STATUS_INSUFFICIENT_CONTEXT
    assert body["sources"] == []
    assert body["citations"] == []
    assert body["llm_called"] is False


def test_api_negative_evaluation(client) -> None:
    response = client.post(
        "/api/week5/day24/negative/run-all",
        json={"retrieval_top_k": 20, "final_top_k": 5, "similarity_threshold": 0.5},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["runs"]) == 2
    assert all(run["passed"] for run in body["runs"])
    assert all(run["actual_status"] == STATUS_INSUFFICIENT_CONTEXT for run in body["runs"])
    assert all(run["llm_called"] is False for run in body["runs"])


def test_api_status(client) -> None:
    response = client.get("/api/week5/day24/status")
    assert response.status_code == 200
    body = response.json()
    assert body["answer_threshold"] == 0.5
    assert body["demo_question_manual"]


# ----------------------------------------------------------------------
# 23-26. Evaluation
# ----------------------------------------------------------------------
async def test_evaluation_automatic_checks(day24_evaluation_service) -> None:
    response = await day24_evaluation_service.run_all()
    assert len(response.runs) == 2
    for run in response.runs:
        assert run.checks.has_sources is True
        assert run.checks.has_quotes is True
        assert run.checks.all_quotes_valid is True
        assert run.checks.grounding_valid is True
    summary = day24_evaluation_service.results().summary
    assert summary.sources_present == 2
    assert summary.quotes_present == 2
    assert summary.quotes_valid == 2


def test_evaluation_manual_grade_roundtrip(day24_evaluation_service) -> None:
    from app.schemas.day24 import Day24EvaluationGradeRecord

    day24_evaluation_service.update_result(
        "q01", Day24EvaluationGradeRecord(answer_supported_by_evidence="pass")
    )
    result = day24_evaluation_service.results()
    assert result.results["q01"].answer_supported_by_evidence == "pass"
    assert result.summary.supported["pass"] == 1


async def test_negative_evaluation_report(day24_evaluation_service) -> None:
    report = await day24_evaluation_service.negative_report()
    assert report.questions == 2
    assert report.correctly_refused == 2
    assert report.incorrectly_answered == 0
    assert report.llm_invoked == 0


# ----------------------------------------------------------------------
# 27. Day 22 / Day 23 modes are not broken
# ----------------------------------------------------------------------
async def test_baseline_and_improved_still_work(
    day22_service, day23_service
) -> None:
    baseline = await day22_service.answer_with_rag("вопрос", top_k=2)
    assert baseline.mode == "rag"
    improved = await day23_service.answer_with_improved_rag("вопрос")
    assert improved.original_question == "вопрос"


# ----------------------------------------------------------------------
# UI surface
# ----------------------------------------------------------------------
def test_homepage_contains_day24(client) -> None:
    html = client.get("/").text
    assert "Day 24 — Grounded RAG" in html
    assert 'id="panel-day24"' in html
    assert 'src="/static/js/day24.js"' in html
    for element_id in (
        "d24-question",
        "d24-retrieval-top-k",
        "d24-similarity-threshold",
        "d24-final-top-k",
        "d24-ask",
        "d24-results",
        "d24-eval-table-body",
        "d24-eval-summary",
        "d24-negative-summary",
        "d24-negative-table-body",
    ):
        assert f'id="{element_id}"' in html


# ----------------------------------------------------------------------
# Integration smoke test (real index + real generation)
# ----------------------------------------------------------------------
@pytest.mark.integration
def test_day24_integration_smoke() -> None:
    """Ground one manual question and refuse one unknown question.

    Run explicitly: ``pytest -m integration tests/test_day24.py``. Requires
    the Day 21 index, Ollama/bge-m3 and a reachable generation provider.
    """
    from app.config import get_settings

    settings = get_settings()
    service = GroundedRagService(settings)

    manual = asyncio.run(
        service.ask(
            "Какое давление должно быть в шинах Mitsubishi Xpander размера "
            "205/55R16 при нагрузке 1–5 человек + груз?"
        )
    )
    assert manual.status == STATUS_ANSWERED
    assert manual.sources
    assert all(citation.quote_valid for citation in manual.citations)

    unknown = asyncio.run(
        service.ask("Как выполняется адаптация вариатора Xpander сканером?")
    )
    assert unknown.status == STATUS_INSUFFICIENT_CONTEXT
