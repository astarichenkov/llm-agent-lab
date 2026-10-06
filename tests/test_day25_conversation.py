"""Day 25 multi-turn conversational RAG regression tests.

These cover the follow-up handling that was missing before the fix:

* short follow-ups ("что делать?", "как установить?") are detected;
* the contextual retrieval query is built from the dialogue (active topic,
  previous question, previous grounded source) BEFORE retrieval;
* an ordinary follow-up never rewrites the goal or wipes task memory;
* only an explicit topic switch resets the task;
* manual/PDF neighbours are added as controlled context expansion;
* a verbatim-but-mis-attributed quote is re-attributed by the backend
  (opt-in, Day 25 only) while a paraphrase is still rejected;
* a partial grounded answer is allowed when at least one citation is valid.
"""
from __future__ import annotations

import pytest

from app.schemas.day24 import STATUS_ANSWERED, STATUS_GROUNDING_FAILED
from app.schemas.day25 import (
    TURN_GOAL_CHANGE,
    FactItem,
    MessageEvidence,
    StateOperation,
    TaskState,
    VehicleState,
)
from app.services.day25.contextual_query import (
    build_contextual_query,
    derive_active_topic,
)
from app.services.day25.task_state import (
    apply_operations,
    default_task_state,
    state_to_prompt_text,
)
from app.services.day25.turn_analysis import (
    is_context_dependent_followup,
    is_explicit_topic_switch,
)
from app.services.day25.updater import classify_turn
from app.services.rag.grounding.validator import CitationValidator
from app.services.rag.models import Chunk
from app.schemas.day22 import RetrievedChunk


# ----------------------------------------------------------------------
# Follow-up / switch detection
# ----------------------------------------------------------------------
def test_short_followups_are_detected():
    for message in [
        "что делать?",
        "как?",
        "а дальше?",
        "куда ставить?",
        "какой использовать?",
        "почему?",
        "а этот подойдет?",
        "это подходит?",
        "куда крепить?",
        "а как его проверить?",
    ]:
        assert is_context_dependent_followup(message), message


def test_long_standalone_question_is_not_followup():
    message = (
        "Расскажи подробно какие бывают виды защиты двигателя для "
        "Mitsubishi Xpander и чем они отличаются"
    )
    assert not is_context_dependent_followup(message)


def test_explicit_topic_switch_detection():
    assert is_explicit_topic_switch("с защитой закончили. теперь почему свистит ремень?")
    assert is_explicit_topic_switch("перейдём к обслуживанию вариатора")
    assert is_explicit_topic_switch("другой вопрос: что с электрикой?")
    assert is_explicit_topic_switch("новая проблема с запуском")
    # Pure follow-ups are not switches.
    assert not is_explicit_topic_switch("что делать?")
    assert not is_explicit_topic_switch("теперь что делать?")
    assert not is_explicit_topic_switch("как установить?")
    assert not is_explicit_topic_switch("а дальше?")


# ----------------------------------------------------------------------
# Contextual query builder
# ----------------------------------------------------------------------
def _state_with_topic() -> TaskState:
    return TaskState(
        goal="подобрать и установить защиту двигателя",
        active_topic="защита картера и КПП Sheriff 5307/5308",
        vehicle=VehicleState(model="Mitsubishi Xpander"),
    )


def test_followup_query_is_self_contained():
    state = _state_with_topic()
    recent = [
        {"role": "user", "content": "какая подходит защита двигателя?"},
        {"role": "assistant", "content": "Ответ"},
    ]
    evidence = [
        MessageEvidence(
            message_id="m1",
            chunk_id="c1",
            source="Инструкция_по_установке_защиты_шериф.pdf",
            chunk_text="... 5307/5308 M8x20 M8x30 ...",
        )
    ]
    query = build_contextual_query(
        "что делать?", state, recent_messages=recent, previous_evidence=evidence
    )
    lower = query.lower()
    assert "защита" in lower
    assert "установк" in lower
    assert "mitsubishi xpander" in lower
    assert "5307/5308" in query
    assert "какая подходит защита двигателя" in query


def test_standalone_query_does_not_pull_old_topic():
    state = _state_with_topic()
    query = build_contextual_query(
        "Расскажи подробно про замену масла в двигателе Mitsubishi Xpander",
        state,
        recent_messages=[{"role": "user", "content": "какая защита подходит?"}],
        previous_evidence=[
            MessageEvidence(message_id="m1", chunk_id="c1", source="sheriff.pdf")
        ],
    )
    assert "текущая тема" not in query
    assert "защита картера" not in query


def test_derive_active_topic_keeps_on_followup_and_replaces_on_switch():
    previous = "защита картера и КПП Sheriff 5307/5308"
    assert derive_active_topic("что делать?", previous, followup=True) == previous
    assert (
        derive_active_topic("какая защита двигателя подходит?", followup=False)
        == "какая защита двигателя подходит"
    )
    switched = derive_active_topic(
        "с защитой закончили. теперь почему свистит ремень?",
        previous,
        topic_switch=True,
    )
    assert switched is not None
    assert "ремень" in switched.lower()
    assert "защита" not in switched.lower()


# ----------------------------------------------------------------------
# Turn classification / goal refinement
# ----------------------------------------------------------------------
def test_set_goal_alone_is_not_a_goal_change():
    operations = [StateOperation(op="set_goal", text="защита двигателя")]
    assert classify_turn("какая защита двигателя подходит?", operations) != TURN_GOAL_CHANGE
    assert classify_turn("как установить?", operations) != TURN_GOAL_CHANGE


def test_explicit_switch_is_a_goal_change():
    assert (
        classify_turn("с защитой закончили. теперь почему свистит ремень?", [])
        == TURN_GOAL_CHANGE
    )


def test_active_topic_ops():
    state = apply_operations(
        default_task_state(),
        [StateOperation(op="set_active_topic", text="защита картера")],
    )
    assert state.active_topic == "защита картера"
    assert "ACTIVE TOPIC: защита картера" in state_to_prompt_text(state)
    state = apply_operations(state, [StateOperation(op="clear_active_topic")])
    assert state.active_topic is None


# ----------------------------------------------------------------------
# Citation validation: re-attribution vs paraphrase
# ----------------------------------------------------------------------
def _chunks():
    return [
        RetrievedChunk(
            rank=1,
            similarity=0.9,
            chunk_id="manual-1",
            source_type="manual",
            source="sheriff.pdf",
            text="Защита крепится к штатным отверстиям силовых элементов кузова.",
        ),
        RetrievedChunk(
            rank=2,
            similarity=0.8,
            chunk_id="telegram-1",
            source_type="telegram",
            source="result.json",
            text="Владелец связал вспучивание пластика с пеной для мойки.",
        ),
    ]


def test_validator_reattributes_verbatim_quote_when_enabled():
    validator = CitationValidator(_chunks(), resolve_quote_chunk=True)
    outcome = validator.validate(
        [
            {
                "chunk_id": "manual-1",
                "quote": "Владелец связал вспучивание пластика с пеной для мойки.",
            }
        ]
    )
    assert outcome.all_valid
    assert outcome.citations[0].source_type == "telegram"


def test_validator_reattribution_does_not_accept_paraphrase():
    validator = CitationValidator(_chunks(), resolve_quote_chunk=True)
    outcome = validator.validate(
        [{"chunk_id": "manual-1", "quote": "Совершенно другая фраза."}]
    )
    assert not outcome.all_valid


def test_validator_without_reattribution_stays_strict():
    validator = CitationValidator(_chunks())
    outcome = validator.validate(
        [
            {
                "chunk_id": "manual-1",
                "quote": "Владелец связал вспучивание пластика с пеной для мойки.",
            }
        ]
    )
    assert not outcome.all_valid


# ----------------------------------------------------------------------
# Partial grounded answer + context expansion
# ----------------------------------------------------------------------
async def test_partial_citations_opt_in(day24_service, day24_generation):
    payload = {
        "status": "answered",
        "answer": "Частичный ответ",
        "evidence": [
            {
                "chunk_id": "manual-chunk-1",
                "quote": "Давление в шинах 205/55R16: 2,1 бар.",
            },
            {"chunk_id": "manual-chunk-1", "quote": "Этой цитаты нет в контексте"},
        ],
    }
    day24_generation.payload = payload
    strict = await day24_service.ask("давление?")
    assert strict.status == STATUS_GROUNDING_FAILED

    day24_generation.payload = payload
    partial = await day24_service.ask(
        "давление?", resolve_quote_chunk=True, allow_partial_citations=True
    )
    assert partial.status == STATUS_ANSWERED
    assert partial.sources
    assert all(citation.quote_valid for citation in partial.citations)


async def test_context_expansion_adds_same_document_neighbor(
    day24_answer_service, day22_rag_service
):
    day22_rag_service.neighbors_map = {
        "manual-chunk-1": [
            Chunk(
                chunk_id="manual-neighbor-1",
                text="M8x20 M8x30 моменты затяжки",
                metadata={
                    "source_type": "manual",
                    "source": "20_XPANDER_RU1.pdf",
                    "page": 239,
                },
                doc_id="doc-1",
            )
        ]
    }
    prepared = await day24_answer_service.prepare_improved_retrieval(
        "давление?", expand_context=True
    )
    ids = [candidate.chunk_id for candidate in prepared.context_chunks]
    assert "manual-neighbor-1" in ids
    neighbor = next(
        c for c in prepared.context_chunks if c.chunk_id == "manual-neighbor-1"
    )
    assert neighbor.reason == "doc_context_expansion"


async def test_context_expansion_is_off_by_default(
    day24_answer_service, day22_rag_service
):
    day22_rag_service.neighbors_map = {
        "manual-chunk-1": [
            Chunk(
                chunk_id="manual-neighbor-1",
                text="neighbour",
                metadata={"source_type": "manual", "source": "x.pdf"},
                doc_id="doc-1",
            )
        ]
    }
    prepared = await day24_answer_service.prepare_improved_retrieval("давление?")
    assert "manual-neighbor-1" not in [
        candidate.chunk_id for candidate in prepared.context_chunks
    ]


# ----------------------------------------------------------------------
# Chat pipeline: history used before retrieval, memory preserved
# ----------------------------------------------------------------------
async def test_followup_carries_topic_into_retrieval(
    day25_service, day22_rag_service
):
    session = day25_service.create_session()
    await day25_service.send_message(session.id, "какая подходит защита двигателя?")
    state = day25_service.get_state(session.id)
    assert state.active_topic
    assert state.goal and "защита" in state.goal.lower()

    await day25_service.send_message(session.id, "что делать?")
    followup_query = day22_rag_service.search_calls[-1][0]
    assert "текущая тема" in followup_query
    assert state.active_topic.split(" — ")[0] in followup_query
    assert "Mitsubishi Xpander" in followup_query
    # The follow-up did not replace the goal.
    assert day25_service.get_state(session.id).goal == state.goal


async def test_topic_switch_resets_task_memory(day25_service):
    session = day25_service.create_session()
    await day25_service.send_message(session.id, "какая подходит защита двигателя?")
    await day25_service.send_message(
        session.id, "Защита крепится к штатным отверстиям силовых элементов."
    )
    before = day25_service.get_state(session.id)
    assert any("Защита крепится" in fact.text for fact in before.known_facts)

    await day25_service.send_message(
        session.id, "с защитой закончили. теперь почему свистит ремень?"
    )
    after = day25_service.get_state(session.id)
    assert after.active_topic and "ремень" in after.active_topic.lower()
    assert not any("Защита крепится" in fact.text for fact in after.known_facts)


async def test_goal_refinement_keeps_memory(day25_service):
    session = day25_service.create_session()
    await day25_service.send_message(session.id, "какая подходит защита двигателя?")
    await day25_service.send_message(session.id, "Защита крепится к штатным отверстиям.")
    before = day25_service.get_state(session.id)

    await day25_service.send_message(session.id, "как её установить?")
    after = day25_service.get_state(session.id)
    assert after.goal == before.goal
    assert after.active_topic == before.active_topic
    assert any("Защита крепится" in fact.text for fact in after.known_facts)


# ----------------------------------------------------------------------
# Real integration regression scenario (requires a real model + index)
# ----------------------------------------------------------------------
@pytest.mark.integration
def test_day25_multiturn_engine_undertray_regression(tmp_path) -> None:
    """The 5-turn engine-undertray dialogue that used to lose its topic.

    Run explicitly::

        pytest -m integration tests/test_day25_conversation.py

    Requires the Day 21 index, Ollama/bge-m3 and a reachable generation
    model. It asserts the RETRIEVAL guarantees (which are deterministic) and
    that the critical follow-up ("что делать?") does not fall back to a
    generic refusal while the instruction is still retrievable.
    """
    import asyncio

    from app.config import get_settings
    from app.services.day25 import ChatRepository, Day25ChatService
    from app.schemas.day25 import CreateSessionRequest

    settings = get_settings().model_copy(
        update={"day25_chat_db_path": str(tmp_path / "multiturn.sqlite3")}
    )
    service = Day25ChatService(
        settings, repository=ChatRepository(settings.day25_chat_db_path)
    )
    session = service.create_session(CreateSessionRequest(title="Engine undertray"))

    questions = [
        "какая подходит защита двигателя?",
        "как установить?",
        "что делать?",
        "куда крепить?",
        "какой крепеж используется?",
    ]
    context_queries: list[str] = []
    for question in questions:
        response = asyncio.run(service.send_message(session.id, question))
        trace = response.assistant_message.trace or {}
        context_queries.append(str(trace.get("contextual_query", "")))

    state = service.get_state(session.id)
    assert state.active_topic and "защит" in state.active_topic.lower()
    # Every follow-up carries the topic into retrieval, not just into the prompt.
    for query in context_queries[1:]:
        assert "защит" in query.lower()
    # "что делать?" must not lose the topic and must not generic-refuse while
    # the instruction is retrievable.
    assert context_queries[2].lower().count("защит") >= 1

    # Topic switch: a new task must not carry the undertray topic/source.
    switched = asyncio.run(
        service.send_message(
            session.id, "с защитой закончили. теперь почему свистит ремень?"
        )
    )
    switch_trace = switched.assistant_message.trace or {}
    switch_query = str(switch_trace.get("contextual_query", "")).lower()
    assert "текущая тема" not in switch_query
    assert "5307/5308" not in switch_query
    assert "инструкция по установке защиты шериф" not in switch_query

    # Genuine unknown still refuses.
    unknown = asyncio.run(
        service.send_message(session.id, "как адаптировать вариатор сканером?")
    )
    assert unknown.assistant_message.status in {
        "insufficient_context",
        "grounding_failed",
    }
