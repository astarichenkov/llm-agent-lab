"""Day 22 — first RAG request tests.

No test contacts a real Ollama, DeepSeek or SQLite index:

* retrieval is replaced by ``FakeDay22RagService`` (in-memory chunks);
* generation is replaced by ``FakeDay22Generation`` (records prompts);
* the HTTP layer uses the shared ``client`` fixture with dependency overrides.

The only exception is the explicitly-marked integration smoke test.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from conftest import FakeDay22Generation, FakeDay22RagService
from app.config import Settings
from app.schemas.day22 import (
    EvaluationGradeRecord,
    RetrievedChunk,
)
from app.services.deepseek import DeepSeekRateLimitError
from app.services.rag.answer_service import RagAnswerService, RagRetrievalError
from app.services.rag.context import ContextBuilder
from app.services.rag.evaluation import (
    EvaluationError,
    EvaluationResultStore,
    EvaluationService,
    load_questions,
)
from app.services.rag.generation import GenerationError
from app.services.rag.generation.deepseek import DeepSeekGenerationProvider
from app.services.rag.generation.ollama import OllamaGenerationProvider


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _service(
    settings: Settings,
    *,
    hits=None,
    exists=True,
    answer="Fake generated answer.",
):
    rag = FakeDay22RagService(hits=hits, exists=exists)
    generation = FakeDay22Generation(answer=answer)
    service = RagAnswerService(settings, rag_service=rag, generation=generation)
    return service, rag, generation


def _telegram_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        rank=2,
        similarity=0.81,
        chunk_id="tg-1",
        source_type="telegram",
        source="result.json",
        text="Владелец связал это с пеной для мойки.",
        metadata={
            "source_type": "telegram",
            "source": "result.json",
            "chat_name": "Xpander Club",
            "message_ids": [83159, 83165],
            "authors": ["Владелец A"],
            "date_from": "2026-09-22T12:08:00",
        },
    )


def _manual_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        similarity=0.91,
        chunk_id="man-1",
        source_type="manual",
        source="20_XPANDER_RU1.pdf",
        text="Давление в шинах 2,1 бар.",
        metadata={
            "source_type": "manual",
            "source": "20_XPANDER_RU1.pdf",
            "page": 238,
        },
    )


# ----------------------------------------------------------------------
# 1-2. retrieval is / is not performed
# ----------------------------------------------------------------------
async def test_no_rag_never_retrieves(settings: Settings) -> None:
    service, rag, generation = _service(settings)
    answer = await service.answer_without_rag("вопрос")
    assert rag.search_calls == []
    assert answer.retrieval_performed is False
    assert answer.chunks == []
    assert answer.answer == "Fake generated answer."
    # exactly one generation call, messages = system + user
    assert len(generation.calls) == 1
    assert [m["role"] for m in generation.calls[0]["messages"]] == ["system", "user"]


async def test_rag_calls_retrieval(settings: Settings) -> None:
    service, rag, _ = _service(settings)
    answer = await service.answer_with_rag("вопрос", top_k=2)
    assert len(rag.search_calls) == 1
    assert rag.search_calls[0][0] == "вопрос"
    assert rag.search_calls[0][1] == 2
    assert answer.retrieval_performed is True
    assert len(answer.chunks) == 2


# ----------------------------------------------------------------------
# 3-4. context is built and passed to the LLM
# ----------------------------------------------------------------------
async def test_rag_passes_chunks_to_context_builder(settings: Settings) -> None:
    captured = {}

    class RecordingContextBuilder(ContextBuilder):
        def build(self, chunks):  # type: ignore[override]
            captured["chunks"] = list(chunks)
            return super().build(chunks)

    rag = FakeDay22RagService()
    generation = FakeDay22Generation()
    service = RagAnswerService(
        settings,
        rag_service=rag,
        generation=generation,
        context_builder=RecordingContextBuilder(),
    )
    await service.answer_with_rag("вопрос", top_k=2)
    assert len(captured["chunks"]) == 2
    assert captured["chunks"][0].chunk_id == rag.hits[0].chunk_id


async def test_rag_prompt_contains_context(settings: Settings) -> None:
    service, rag, generation = _service(settings)
    await service.answer_with_rag("вопрос", top_k=2)
    user_message = generation.calls[0]["messages"][-1]["content"]
    assert "КОНТЕКСТ:" in user_message
    assert "ВОПРОС:" in user_message
    assert rag.hits[0].text in user_message
    assert rag.hits[1].text in user_message


# ----------------------------------------------------------------------
# 5. same generation model/config in both modes
# ----------------------------------------------------------------------
async def test_compare_uses_one_generation_model(settings: Settings) -> None:
    service, _, generation = _service(settings)
    comparison = await service.compare("вопрос", top_k=2)
    assert comparison.no_rag.model == comparison.rag.model == generation.model
    assert comparison.no_rag.temperature == comparison.rag.temperature
    assert comparison.no_rag.generation_provider == comparison.rag.generation_provider
    # both generation calls used the configured temperature
    for call in generation.calls:
        assert call["temperature"] == settings.rag_generation_temperature


# ----------------------------------------------------------------------
# 6. top_k is applied
# ----------------------------------------------------------------------
async def test_top_k_applied(settings: Settings) -> None:
    service, rag, _ = _service(settings)
    answer = await service.answer_with_rag("вопрос", top_k=1)
    assert len(answer.chunks) == 1
    assert rag.search_calls[0][1] == 1
    assert answer.top_k == 1


# ----------------------------------------------------------------------
# 7. metadata is preserved in the result
# ----------------------------------------------------------------------
async def test_retrieved_metadata_preserved(settings: Settings) -> None:
    service, _, _ = _service(settings)
    answer = await service.answer_with_rag("вопрос", top_k=2)
    manual = [c for c in answer.chunks if c.source_type == "manual"][0]
    telegram = [c for c in answer.chunks if c.source_type == "telegram"][0]
    assert manual.metadata["page"] == 238
    assert manual.source == "20_XPANDER_RU1.pdf"
    assert telegram.metadata["message_ids"] == [83159, 83165, 83205]
    assert telegram.metadata["authors"] == ["Владелец A", "Владелец B"]


# ----------------------------------------------------------------------
# 8-9. context formatting & telegram is not manual
# ----------------------------------------------------------------------
def test_context_builder_formats_both_sources() -> None:
    context = ContextBuilder(max_chars=10000).build([_manual_chunk(), _telegram_chunk()])
    assert "source_type: manual" in context
    assert "page: 238" in context
    assert "source_type: telegram" in context
    assert "message_ids: 83159, 83165" in context
    assert "authors: Владелец A" in context
    # manual and telegram are clearly distinguished
    assert "Официальная техническая документация" in context
    assert "Обсуждение владельцев" in context


async def test_telegram_is_not_presented_as_manual(settings: Settings) -> None:
    service, _, generation = _service(settings)
    await service.answer_with_rag("вопрос", top_k=2)
    system = generation.calls[0]["messages"][0]["content"]
    user = generation.calls[0]["messages"][-1]["content"]
    # System prompt explicitly separates the two source types.
    assert "TELEGRAM" in system
    assert "НЕ как официальную рекомендацию" in system
    # The telegram chunk block labels itself telegram, not manual.
    telegram_start = user.index("source_type: telegram")
    telegram_block = user[telegram_start:telegram_start + 400]
    assert "source_kind: Обсуждение владельцев" in telegram_block
    assert "page:" not in telegram_block
    assert "message_ids: 83159, 83165" in telegram_block


def test_context_builder_truncates() -> None:
    chunk = _manual_chunk()
    chunk.text = "данные " * 400
    context = ContextBuilder(max_chars=300).build([chunk])
    assert len(context) <= 320  # bound plus the truncation marker
    assert "обрезано" in context


# ----------------------------------------------------------------------
# 10. compare returns both answers
# ----------------------------------------------------------------------
async def test_compare_returns_both(settings: Settings) -> None:
    service, _, _ = _service(settings)
    comparison = await service.compare("вопрос", top_k=2)
    assert comparison.no_rag.mode == "no_rag"
    assert comparison.rag.mode == "rag"
    assert comparison.rag.chunks
    assert comparison.question == "вопрос"


async def test_rag_empty_retrieval_still_answers(settings: Settings) -> None:
    service, _, generation = _service(settings, hits=[])
    answer = await service.answer_with_rag("вопрос", top_k=5)
    assert answer.retrieval_performed is True
    assert answer.chunks == []
    assert answer.answer == "Fake generated answer."
    assert "не найдено" in generation.calls[0]["messages"][-1]["content"]


# ----------------------------------------------------------------------
# 11. error handling
# ----------------------------------------------------------------------
async def test_missing_index_raises(settings: Settings) -> None:
    service, _, _ = _service(settings, exists=False)
    with pytest.raises(RagRetrievalError) as exc_info:
        await service.answer_with_rag("вопрос")
    assert exc_info.value.status_code == 503


async def test_generation_error_propagates(settings: Settings) -> None:
    service, _, generation = _service(settings)
    generation.error = GenerationError("Provider down", status_code=504)
    with pytest.raises(GenerationError) as exc_info:
        await service.answer_without_rag("вопрос")
    assert exc_info.value.status_code == 504


async def test_ollama_provider_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "model": "local-chat",
                "message": {"role": "assistant", "content": "локальный ответ"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 7,
                "eval_count": 3,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OllamaGenerationProvider(
            "http://localhost:11434", "local-chat", client=client
        )
        result = await provider.generate(
            [{"role": "user", "content": "привет"}], temperature=0.0
        )
    assert result.content == "локальный ответ"
    assert result.finish_reason == "stop"
    assert provider.model == "local-chat"


def test_ollama_provider_requires_model() -> None:
    with pytest.raises(GenerationError):
        OllamaGenerationProvider("http://localhost:11434", "")


async def test_ollama_provider_maps_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OllamaGenerationProvider(
            "http://localhost:11434", "local-chat", client=client
        )
        with pytest.raises(GenerationError):
            await provider.generate([{"role": "user", "content": "hi"}])


async def test_deepseek_provider_maps_error() -> None:
    class BoomService:
        async def generate(self, *args, **kwargs):
            raise DeepSeekRateLimitError()

    provider = DeepSeekGenerationProvider(BoomService(), model="m")  # type: ignore[arg-type]
    with pytest.raises(GenerationError) as exc_info:
        await provider.generate([{"role": "user", "content": "hi"}])
    assert exc_info.value.status_code == 429


# ----------------------------------------------------------------------
# 12-13. evaluation dataset + persistence
# ----------------------------------------------------------------------
def test_evaluation_loader(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q01",
                    "question": "Q?",
                    "category": "manual_fact",
                    "expected_facts": ["fact"],
                    "expected_sources": [{"source_type": "manual", "source": "m.pdf"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    questions = load_questions(path)
    assert len(questions) == 1
    assert questions[0].id == "q01"
    assert questions[0].expected_facts == ["fact"]


def test_evaluation_loader_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError) as exc_info:
        load_questions(tmp_path / "missing.json")
    assert exc_info.value.status_code == 404


def test_evaluation_result_persistence(tmp_path: Path) -> None:
    store = EvaluationResultStore(tmp_path / "results.json")
    assert store.load() == {}
    store.update("q01", EvaluationGradeRecord(no_rag="partial", rag="pass"))
    reloaded = EvaluationResultStore(tmp_path / "results.json").load()
    assert reloaded["q01"].no_rag == "partial"
    assert reloaded["q01"].rag == "pass"
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["q01"][
        "rag"
    ] == "pass"


def test_evaluation_summary_ignores_ungraded(settings: Settings, tmp_path: Path) -> None:
    dataset = tmp_path / "questions.json"
    dataset.write_text(
        json.dumps(
            [
                {"id": "q01", "question": "A", "category": "manual_fact"},
                {"id": "q02", "question": "B", "category": "manual_fact"},
            ]
        ),
        encoding="utf-8",
    )
    rag = FakeDay22RagService()
    service = EvaluationService(
        settings,
        answer_service=RagAnswerService(
            settings, rag_service=rag, generation=FakeDay22Generation()
        ),
        dataset_path=str(dataset),
        results_path=str(tmp_path / "results.json"),
    )
    service.update_result("q01", EvaluationGradeRecord(rag="pass", expected_source_retrieved=True))
    response = service.results()
    assert response.summary.total_questions == 2
    assert response.summary.evaluated == 1
    assert response.summary.rag["pass"] == 1
    assert response.summary.rag["fail"] == 0
    assert response.summary.expected_source_retrieved == 1


def test_evaluation_invalid_id(settings: Settings, day22_service) -> None:
    service = EvaluationService(
        settings, answer_service=day22_service, dataset_path="data/day22/evaluation_questions.json"
    )
    with pytest.raises(EvaluationError) as exc_info:
        service.get("does-not-exist")
    assert exc_info.value.status_code == 404


# ----------------------------------------------------------------------
# 14-17. HTTP API (fake retriever + fake LLM via conftest overrides)
# ----------------------------------------------------------------------
def test_api_status(client) -> None:
    response = client.get("/api/week5/day22/status")
    assert response.status_code == 200
    body = response.json()
    assert body["embedding_model"] == "bge-m3"
    assert body["generation_model"] == "fake-generation-model"
    assert body["default_top_k"] == 5


def test_api_no_rag(client, day22_rag_service) -> None:
    day22_rag_service.search_calls.clear()
    response = client.post(
        "/api/week5/day22/ask", json={"question": "вопрос", "mode": "no_rag", "top_k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "no_rag"
    assert body["retrieval_performed"] is False
    assert body["chunks"] == []
    assert day22_rag_service.search_calls == []


def test_api_rag(client, day22_rag_service) -> None:
    response = client.post(
        "/api/week5/day22/ask", json={"question": "вопрос", "mode": "rag", "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "rag"
    assert body["retrieval_performed"] is True
    assert len(body["chunks"]) == 2
    assert body["chunks"][0]["source_type"] == "manual"
    assert day22_rag_service.search_calls[-1][1] == 2


def test_api_compare(client) -> None:
    response = client.post(
        "/api/week5/day22/compare", json={"question": "вопрос", "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["no_rag"]["mode"] == "no_rag"
    assert body["rag"]["mode"] == "rag"
    assert body["model"] == body["no_rag"]["model"] == body["rag"]["model"]


def test_api_invalid_top_k(client) -> None:
    assert client.post(
        "/api/week5/day22/ask", json={"question": "вопрос", "mode": "rag", "top_k": 0}
    ).status_code == 422
    assert client.post(
        "/api/week5/day22/ask", json={"question": "вопрос", "mode": "rag", "top_k": 21}
    ).status_code == 422


def test_api_blank_question(client) -> None:
    assert client.post(
        "/api/week5/day22/ask", json={"question": "   ", "mode": "rag"}
    ).status_code == 422


def test_api_rag_index_missing(client, day22_rag_service) -> None:
    day22_rag_service.exists = False
    try:
        response = client.post(
            "/api/week5/day22/ask", json={"question": "вопрос", "mode": "rag"}
        )
        assert response.status_code == 503
        assert "индекс" in response.json()["detail"].lower()
    finally:
        day22_rag_service.exists = True


def test_api_generation_error_handled(client, day22_generation) -> None:
    day22_generation.error = GenerationError("Provider down", status_code=502)
    try:
        response = client.post(
            "/api/week5/day22/ask", json={"question": "вопрос", "mode": "no_rag"}
        )
        assert response.status_code == 502
        assert response.json()["detail"] == "Provider down"
    finally:
        day22_generation.error = None


def test_api_evaluation_questions(client) -> None:
    response = client.get("/api/week5/day22/evaluation/questions")
    assert response.status_code == 200
    body = response.json()
    assert [q["id"] for q in body] == ["q01", "q02"]
    assert body[0]["expected_facts"] == ["2,1 бар"]


def test_api_evaluation_run(client) -> None:
    response = client.post(
        "/api/week5/day22/evaluation/run/q01", json={"top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["question"]["id"] == "q01"
    assert body["comparison"]["rag"]["mode"] == "rag"


def test_api_evaluation_run_invalid_id(client) -> None:
    response = client.post(
        "/api/week5/day22/evaluation/run/nope", json={"top_k": 2}
    )
    assert response.status_code == 404


async def test_evaluation_run_all(day22_evaluation_service) -> None:
    response = await day22_evaluation_service.run_all(top_k=2)
    assert [run.question.id for run in response.runs] == ["q01", "q02"]
    assert all(run.comparison.rag.retrieval_performed for run in response.runs)


def test_api_evaluation_run_all(client) -> None:
    response = client.post("/api/week5/day22/evaluation/run-all", json={"top_k": 2})
    assert response.status_code == 200
    runs = response.json()["runs"]
    assert [run["question"]["id"] for run in runs] == ["q01", "q02"]


def test_api_evaluation_result_roundtrip(client) -> None:
    response = client.put(
        "/api/week5/day22/evaluation/result/q01",
        json={"no_rag": "fail", "rag": "pass", "expected_source_retrieved": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"]["q01"]["rag"] == "pass"
    assert body["summary"]["rag"]["pass"] == 1
    assert body["summary"]["expected_source_retrieved"] == 1

    fetched = client.get("/api/week5/day22/evaluation/results")
    assert fetched.status_code == 200
    assert fetched.json()["results"]["q01"]["no_rag"] == "fail"


def test_api_evaluation_result_invalid_id(client) -> None:
    response = client.put(
        "/api/week5/day22/evaluation/result/nope", json={"rag": "pass"}
    )
    assert response.status_code == 404


# ----------------------------------------------------------------------
# UI surface
# ----------------------------------------------------------------------
def test_homepage_contains_day22(client) -> None:
    html = client.get("/").text
    assert "Mitsubishi Xpander RAG Assistant" in html
    assert "Day 22 — No RAG vs RAG" in html
    assert 'id="panel-day22"' in html
    assert 'src="/static/js/day22.js"' in html
    for element_id in (
        "d22-question",
        "d22-mode-no-rag",
        "d22-mode-rag",
        "d22-top-k",
        "d22-ask",
        "d22-compare",
        "d22-results",
        "d22-eval-table-body",
        "d22-eval-summary",
    ):
        assert f'id="{element_id}"' in html


# ----------------------------------------------------------------------
# Integration smoke test (real Day 21 index + real generation model)
# ----------------------------------------------------------------------
@pytest.mark.integration
def test_day22_integration_smoke() -> None:
    """Exercise the real index and the configured generation model.

    Run explicitly: ``pytest -m integration tests/test_day22.py``.
    Requires the Day 21 index, Ollama/bge-m3 and a reachable generation
    provider. No assertion is made that RAG is "better" — only that both
    modes return answers and that retrieval returns real sources.
    """
    from app.config import get_settings

    settings = get_settings()
    service = RagAnswerService(settings)

    manual = service.rag.search("давление в шинах 205/55R16", top_k=3)
    telegram = service.rag.search(
        "Что владельцы называют причиной того, что пластик решётка радиатора и зеркала поплыл?",
        top_k=3,
    )
    assert any(h.metadata.get("source_type") == "manual" for h in manual)
    assert any(h.metadata.get("source_type") == "telegram" for h in telegram)

    import asyncio

    comparison = asyncio.run(service.compare("Какое давление в шинах?", top_k=3))
    assert comparison.no_rag.answer
    assert comparison.rag.answer
    assert comparison.rag.retrieval_performed is True
    assert comparison.rag.chunks
