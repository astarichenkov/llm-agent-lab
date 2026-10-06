"""Day 23 — query rewrite + relevance filtering tests.

No test contacts Ollama, DeepSeek or a real SQLite index:

* retrieval is the Day 22 ``FakeDay22RagService`` (in-memory hits);
* generation is the Day 22 ``FakeDay22Generation``;
* rewrite is ``FakeQueryRewriter`` (records calls, no LLM);
* the HTTP layer uses the shared ``client`` fixture.

The only real-API test is the explicitly-marked integration smoke test at the
bottom.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from conftest import FakeDay22RagService, FakeQueryRewriter
from app.config import Settings
from app.schemas.day22 import ExpectedSource, RetrievedChunk
from app.schemas.day23 import (
    REASON_BELOW_THRESHOLD,
    STATUS_BELOW_THRESHOLD,
    STATUS_RELEVANT_NOT_USED,
    STATUS_USED,
    Day23AskRequest,
    Day23Settings,
)
from app.services.rag.filtering.similarity_filter import SimilarityFilter
from app.services.rag.generation import GenerationError
from app.services.rag.models import SearchHit
from app.services.rag.source_matching import (
    all_expected_sources_retrieved,
    matches_expected_source,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _hit(score, chunk_id, text=None, source_type="manual", **meta):
    text = text or f"text-{chunk_id}"
    metadata = {"source_type": source_type, "source": meta.pop("source", "doc.pdf")}
    metadata.update(meta)
    return SearchHit(score=score, chunk_id=chunk_id, text=text, metadata=metadata,
                     doc_id="doc-" + chunk_id)


def _chunks(*scores):
    return [RetrievedChunk(rank=i + 1, similarity=s, chunk_id=f"c{i}", source_type="manual",
                           source="doc.pdf", text=f"text {i}") for i, s in enumerate(scores)]


# ----------------------------------------------------------------------
# 1-3. rewrite is used for retrieval, original question for generation
# ----------------------------------------------------------------------
async def test_rewriter_receives_original_question(day23_service, day23_rewriter) -> None:
    await day23_service.answer_with_improved_rag("Почему машина вибрирует на D?")
    assert day23_rewriter.calls == ["Почему машина вибрирует на D?"]


async def test_rewritten_query_used_for_retrieval(
    day23_service, day22_rag_service, day23_rewriter
) -> None:
    day23_rewriter.rewritten = "Mitsubishi Xpander vibration Drive D engine mounts"
    result = await day23_service.answer_with_improved_rag("Почему машина вибрирует на D?")
    assert day22_rag_service.search_calls[0][0] == day23_rewriter.rewritten
    assert result.rewritten_query == day23_rewriter.rewritten


async def test_original_question_used_for_generation(
    day23_service, day22_generation, day23_rewriter
) -> None:
    question = "Почему машина вибрирует на D?"
    day23_rewriter.rewritten = "vibration at idle Drive D engine mount"
    await day23_service.answer_with_improved_rag(question)
    # The rewrite is a fake (no generation call); the only generation call is
    # the answer, and it must contain the ORIGINAL question.
    assert len(day22_generation.calls) == 1
    user_message = day22_generation.calls[0]["messages"][-1]["content"]
    assert question in user_message
    assert day23_rewriter.rewritten not in user_message.split("ВОПРОС:")[-1]


# ----------------------------------------------------------------------
# 4-5. baseline never rewrites / never thresholds
# ----------------------------------------------------------------------
async def test_baseline_does_not_rewrite(day23_service, day23_rewriter) -> None:
    await day23_service.answer_with_rag("вопрос")
    assert day23_rewriter.calls == []


async def test_baseline_does_not_apply_threshold(
    settings: Settings, day22_generation
) -> None:
    rag = FakeDay22RagService(hits=[_hit(0.10, "low-1"), _hit(0.05, "low-2")])
    from app.services.rag.answer_service import RagAnswerService

    service = RagAnswerService(settings, rag_service=rag, generation=day22_generation)
    answer = await service.answer_with_rag("вопрос", top_k=2)
    assert len(answer.chunks) == 2  # both kept despite very low similarity


# ----------------------------------------------------------------------
# 6. improved uses retrieval_top_k
# ----------------------------------------------------------------------
async def test_improved_uses_retrieval_top_k(day23_service, day22_rag_service) -> None:
    await day23_service.answer_with_improved_rag("вопрос", retrieval_top_k=3)
    assert day22_rag_service.search_calls[0][1] == 3


# ----------------------------------------------------------------------
# 7-9. similarity filter rules
# ----------------------------------------------------------------------
def test_filter_accepts_above_threshold() -> None:
    result = SimilarityFilter(0.5).filter(_chunks(0.9))
    assert result[0].accepted is True
    assert result[0].reason is None


def test_filter_rejects_below_threshold() -> None:
    result = SimilarityFilter(0.5).filter(_chunks(0.4))
    assert result[0].accepted is False
    assert result[0].reason == REASON_BELOW_THRESHOLD


def test_filter_accepts_exactly_at_threshold() -> None:
    result = SimilarityFilter(0.5).filter(_chunks(0.5))
    assert result[0].accepted is True


def test_filter_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError):
        SimilarityFilter(-0.01)
    with pytest.raises(ValueError):
        SimilarityFilter(1.01)


# ----------------------------------------------------------------------
# 10-14. final_top_k / edge cases / trace
# ----------------------------------------------------------------------
async def test_final_top_k_applied_after_threshold(
    settings: Settings, day22_generation
) -> None:
    from app.services.rag.answer_service import RagAnswerService

    rag = FakeDay22RagService(
        hits=[_hit(0.9, "a"), _hit(0.8, "b"), _hit(0.7, "c"), _hit(0.4, "d")]
    )
    service = RagAnswerService(
        settings, rag_service=rag, generation=day22_generation,
        rewriter=FakeQueryRewriter(),
    )
    result = await service.answer_with_improved_rag(
        "q", retrieval_top_k=20, final_top_k=2, similarity_threshold=0.5
    )
    assert len(result.accepted_candidates) == 3
    assert len(result.context_chunks) == 2
    assert len(result.rejected_candidates) == 1
    assert len(result.retrieved_candidates) == 4
    assert [c.chunk_id for c in result.context_chunks] == ["a", "b"]
    # accepted but not used must be labelled distinctly
    not_used = [c for c in result.accepted_candidates if not c.used]
    assert [c.status for c in not_used] == [STATUS_RELEVANT_NOT_USED]
    assert [c.status for c in result.context_chunks] == [STATUS_USED, STATUS_USED]


async def test_fewer_than_final_top_k_kept(settings: Settings, day22_generation) -> None:
    from app.services.rag.answer_service import RagAnswerService

    rag = FakeDay22RagService(hits=[_hit(0.9, "a"), _hit(0.6, "b")])
    service = RagAnswerService(
        settings, rag_service=rag, generation=day22_generation,
        rewriter=FakeQueryRewriter(),
    )
    result = await service.answer_with_improved_rag(
        "q", retrieval_top_k=20, final_top_k=5, similarity_threshold=0.5
    )
    assert len(result.context_chunks) == 2
    assert result.no_relevant_context is False


async def test_zero_passed_returns_no_context_and_keeps_rejected(
    settings: Settings, day22_generation
) -> None:
    from app.services.rag.answer_service import RagAnswerService
    from app.services.rag.prompts import CONTEXT_EMPTY_NOTE

    rag = FakeDay22RagService(hits=[_hit(0.2, "a"), _hit(0.1, "b")])
    service = RagAnswerService(
        settings, rag_service=rag, generation=day22_generation,
        rewriter=FakeQueryRewriter(),
    )
    result = await service.answer_with_improved_rag(
        "q", retrieval_top_k=20, final_top_k=5, similarity_threshold=0.6
    )
    assert result.context_chunks == []
    assert result.no_relevant_context is True
    assert len(result.rejected_candidates) == 2
    # rejected chunks are NOT smuggled back into the LLM context
    user_message = day22_generation.calls[0]["messages"][-1]["content"]
    assert CONTEXT_EMPTY_NOTE in user_message
    assert "text-a" not in user_message
    assert "text-b" not in user_message


async def test_rejected_reason_and_status_saved(settings: Settings, day22_generation) -> None:
    from app.services.rag.answer_service import RagAnswerService

    rag = FakeDay22RagService(hits=[_hit(0.9, "a"), _hit(0.2, "b")])
    service = RagAnswerService(
        settings, rag_service=rag, generation=day22_generation,
        rewriter=FakeQueryRewriter(),
    )
    result = await service.answer_with_improved_rag("q", similarity_threshold=0.5)
    rejected = result.rejected_candidates[0]
    assert rejected.reason == REASON_BELOW_THRESHOLD
    assert rejected.status == STATUS_BELOW_THRESHOLD
    assert rejected.accepted is False
    assert rejected.used is False


async def test_accepted_candidate_metadata_preserved(
    settings: Settings, day22_generation
) -> None:
    from app.services.rag.answer_service import RagAnswerService

    hit = _hit(0.9, "a", source_type="telegram", source="result.json",
               message_ids=[83159, 83165], chat_name="Xpander Club")
    rag = FakeDay22RagService(hits=[hit])
    service = RagAnswerService(
        settings, rag_service=rag, generation=day22_generation,
        rewriter=FakeQueryRewriter(),
    )
    result = await service.answer_with_improved_rag("q", similarity_threshold=0.5)
    accepted = result.accepted_candidates[0]
    assert accepted.metadata["message_ids"] == [83159, 83165]
    assert accepted.metadata["chat_name"] == "Xpander Club"
    assert accepted.source_type == "telegram"


# ----------------------------------------------------------------------
# 15. rewrite fallback
# ----------------------------------------------------------------------
async def test_rewrite_failure_falls_back_to_original(
    settings: Settings, day22_generation, day22_rag_service
) -> None:
    from app.services.rag.answer_service import RagAnswerService

    rewriter = FakeQueryRewriter(applied=False, error="provider down")
    service = RagAnswerService(
        settings, rag_service=day22_rag_service, generation=day22_generation,
        rewriter=rewriter,
    )
    result = await service.answer_with_improved_rag("исходный вопрос")
    assert result.rewrite.applied is False
    assert result.rewrite.error == "provider down"
    assert result.rewritten_query == "исходный вопрос"
    assert day22_rag_service.search_calls[0][0] == "исходный вопрос"


async def test_llm_rewriter_maps_generation_error(settings: Settings) -> None:
    from app.services.rag.generation.base import GenerationResult
    from app.services.rag.rewriting.llm_rewriter import LLMQueryRewriter

    class BoomGeneration:
        name = "boom"
        model = "m"

        async def generate(self, messages, **kwargs):
            raise GenerationError("down", status_code=502)

    rewrite = await LLMQueryRewriter(BoomGeneration()).rewrite("вопрос")
    assert rewrite.applied is False
    assert rewrite.rewritten_query == "вопрос"
    assert "down" in rewrite.error


async def test_llm_rewriter_cleans_output(settings: Settings) -> None:
    from app.services.rag.rewriting.llm_rewriter import LLMQueryRewriter

    class CleanGeneration:
        name = "fake"
        model = "m"

        async def generate(self, messages, **kwargs):
            from app.services.rag.generation.base import GenerationResult

            return GenerationResult(content='```\nQuery: vibration Drive D\n```')

    rewrite = await LLMQueryRewriter(CleanGeneration()).rewrite("вопрос")
    assert rewrite.applied is True
    assert rewrite.rewritten_query == "vibration Drive D"


# ----------------------------------------------------------------------
# 16-18. HTTP API
# ----------------------------------------------------------------------
def test_api_baseline(client, day23_rewriter) -> None:
    response = client.post(
        "/api/week5/day22/ask", json={"question": "вопрос", "mode": "rag", "top_k": 2}
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "rag"
    assert day23_rewriter.calls == []


def test_api_improved(client, day23_rewriter, day22_rag_service) -> None:
    day23_rewriter.rewritten = "rewritten retrieval query"
    response = client.post(
        "/api/week5/day23/ask",
        json={"question": "вопрос", "retrieval_top_k": 20,
              "final_top_k": 5, "similarity_threshold": 0.5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rewritten_query"] == "rewritten retrieval query"
    assert body["retrieval_top_k"] == 20
    assert body["similarity_threshold"] == 0.5
    assert len(body["retrieved_candidates"]) == 2
    assert day22_rag_service.search_calls[0][1] == 20


def test_api_compare_baseline_vs_improved(client) -> None:
    response = client.post(
        "/api/week5/day23/compare",
        json={"question": "вопрос", "retrieval_top_k": 10,
              "final_top_k": 2, "similarity_threshold": 0.5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["baseline"]["mode"] == "rag"
    assert body["improved"]["retrieval_top_k"] == 10
    assert body["model"] == body["baseline"]["model"]


def test_api_status(client) -> None:
    response = client.get("/api/week5/day23/status")
    assert response.status_code == 200
    body = response.json()
    assert body["default_retrieval_top_k"] == 20
    assert body["default_final_top_k"] == 5
    assert body["default_similarity_threshold"] == 0.5
    assert body["rewrite_model"] == "fake-generation-model"


# ----------------------------------------------------------------------
# 19-20. expected source detection
# ----------------------------------------------------------------------
def test_expected_source_manual_page() -> None:
    expected = ExpectedSource(source_type="manual", source="doc.pdf", page=238)
    assert matches_expected_source(
        {"source_type": "manual", "source": "doc.pdf", "page": 238}, expected
    )
    assert not matches_expected_source(
        {"source_type": "manual", "source": "doc.pdf", "page": 100}, expected
    )


def test_expected_source_telegram_message_overlap() -> None:
    expected = ExpectedSource(source_type="telegram", message_ids=[83159, 83165])
    assert matches_expected_source(
        {"source_type": "telegram", "message_ids": [83165, 83170]}, expected
    )
    assert not matches_expected_source(
        {"source_type": "telegram", "message_ids": [1, 2]}, expected
    )


def test_all_expected_sources_retrieved_requires_every_source() -> None:
    manual = RetrievedChunk(rank=1, similarity=0.9, chunk_id="m", source_type="manual",
                            source="doc.pdf", text="t", metadata={"page": 238})
    telegram = RetrievedChunk(rank=2, similarity=0.8, chunk_id="t", source_type="telegram",
                              source="result.json", text="t",
                              metadata={"message_ids": [83159]})
    expected = [
        ExpectedSource(source_type="manual", source="doc.pdf", page=238),
        ExpectedSource(source_type="telegram", message_ids=[83159]),
    ]
    assert all_expected_sources_retrieved([manual], expected) is False
    assert all_expected_sources_retrieved([manual, telegram], expected) is True


async def test_evaluation_run_reports_expected_sources(day23_evaluation_service) -> None:
    run = await day23_evaluation_service.run("q01")
    assert run.baseline_expected_source_retrieved is True
    assert run.improved_expected_source_retrieved is True
    assert run.comparison.improved.original_question == run.question.question


def test_score_distribution_report(day23_evaluation_service) -> None:
    report = day23_evaluation_service.score_distribution(
        retrieval_top_k=20, similarity_threshold=0.5
    )
    assert report.question_count == 2
    assert len(report.rows) == 2
    # The fake retriever always returns manual(0.91) + telegram(0.82); both
    # are expected sources for the two test questions and both pass 0.50.
    assert report.expected_kept == report.expected_total > 0
    assert report.best_expected_min is not None
    assert report.question_best_expected_kept == 2


def test_api_evaluation_scores(client) -> None:
    response = client.get(
        "/api/week5/day23/evaluation/scores"
        "?retrieval_top_k=20&similarity_threshold=0.5"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_top_k"] == 20
    assert body["similarity_threshold"] == 0.5
    assert len(body["rows"]) == 2


def test_api_evaluation_results_roundtrip(client) -> None:
    response = client.put(
        "/api/week5/day23/evaluation/result/q01",
        json={"baseline": "pass", "improved": "pass",
              "baseline_expected_source_retrieved": True,
              "improved_expected_source_retrieved": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"]["q01"]["improved"] == "pass"
    assert body["summary"]["baseline_expected_source_retrieved"] == 1
    assert body["summary"]["improved_expected_source_retrieved"] == 1


# ----------------------------------------------------------------------
# 21-22. configuration validation
# ----------------------------------------------------------------------
def test_threshold_config_validation() -> None:
    assert Day23Settings(similarity_threshold=0.0).similarity_threshold == 0.0
    assert Day23Settings(similarity_threshold=1.0).similarity_threshold == 1.0
    with pytest.raises(ValidationError):
        Day23AskRequest(question="q", similarity_threshold=1.5)
    with pytest.raises(ValidationError):
        Day23AskRequest(question="q", similarity_threshold=-0.1)


def test_top_k_config_validation() -> None:
    with pytest.raises(ValidationError):
        Day23AskRequest(question="q", retrieval_top_k=0)
    with pytest.raises(ValidationError):
        Day23AskRequest(question="q", retrieval_top_k=51)
    with pytest.raises(ValidationError):
        Day23AskRequest(question="q", retrieval_top_k=3, final_top_k=5)
    assert Day23AskRequest(question="q", retrieval_top_k=20, final_top_k=5).final_top_k == 5


def test_config_defaults(settings: Settings) -> None:
    assert settings.rag_retrieval_top_k == 20
    assert settings.rag_final_top_k == 5
    assert settings.rag_similarity_threshold == 0.50


# ----------------------------------------------------------------------
# UI surface
# ----------------------------------------------------------------------
def test_homepage_contains_day23(client) -> None:
    html = client.get("/").text
    assert "Day 23 — Query Rewrite + Relevance Filtering" in html
    assert 'id="panel-day23"' in html
    assert 'src="/static/js/day23.js"' in html
    for element_id in (
        "d23-question",
        "d23-mode-baseline",
        "d23-mode-improved",
        "d23-retrieval-top-k",
        "d23-similarity-threshold",
        "d23-final-top-k",
        "d23-ask",
        "d23-compare",
        "d23-results",
        "d23-eval-table-body",
        "d23-eval-summary",
        "d23-score-report",
    ):
        assert f'id="{element_id}"' in html


# ----------------------------------------------------------------------
# Integration smoke test (real index + real rewrite + real generation)
# ----------------------------------------------------------------------
@pytest.mark.integration
def test_day23_integration_smoke() -> None:
    """Exercise real retrieval, rewrite and generation.

    Run explicitly: ``pytest -m integration tests/test_day23.py``.
    Requires the Day 21 index, Ollama/bge-m3 and a reachable generation
    provider. Only asserts structural facts (no "RAG is better" claim).
    """
    import asyncio

    from app.config import get_settings
    from app.services.rag.answer_service import RagAnswerService

    settings = get_settings()
    service = RagAnswerService(settings)

    result = asyncio.run(
        service.answer_with_improved_rag(
            "Объём топливного бака Xpander в литрах по техническим характеристикам",
            retrieval_top_k=20,
            final_top_k=5,
            similarity_threshold=settings.rag_similarity_threshold,
        )
    )
    assert result.rewritten_query
    assert len(result.retrieved_candidates) <= 20
    assert result.answer
