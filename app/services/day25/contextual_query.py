"""Day 25 Contextual Query Builder.

A multi-turn user often writes an incomplete follow-up ("А что тогда
проверить первым?" / "что делать?"). Sending that string straight to the
vector index loses the whole task, so the builder combines:

    current user message
        + active topic / previous question
        + previous grounded source topic
        + relevant structured task state

into a self-contained retrieval query. The Day 23 query rewriter then runs on
TOP of this contextual query, exactly like it does for a plain Day 23 ask.

Crucially, the history is used BEFORE retrieval (not only in the generation
prompt). For standalone questions it is omitted so a new topic is not polluted
by the old one.

The builder is deterministic on purpose: the retrieval query must be
explainable in the UI, and tests must not depend on an LLM.
"""
from __future__ import annotations

from typing import Any, Iterable

from app.schemas.day25 import TaskState
from app.services.day25.turn_analysis import (
    evidence_topic_hint,
    is_context_dependent_followup,
    last_meaningful_user_message,
    normalize,
)


def _vehicle_phrase(state: TaskState) -> str:
    parts = [state.vehicle.model or "Mitsubishi Xpander"]
    if state.vehicle.year:
        parts.append(str(state.vehicle.year))
    if state.vehicle.engine:
        parts.append(f"engine {state.vehicle.engine}")
    if state.vehicle.transmission:
        parts.append(f"transmission {state.vehicle.transmission}")
    if state.vehicle.mileage_km is not None:
        parts.append(f"mileage {state.vehicle.mileage_km} km")
    return " ".join(parts)


def build_contextual_query(
    message: str,
    state: TaskState,
    *,
    recent_messages: Iterable[Any] | None = None,
    previous_evidence: Iterable[Any] | None = None,
    max_facts: int = 8,
    max_constraints: int = 4,
    max_terms: int = 6,
) -> str:
    """Build a self-contained retrieval query from the message + dialogue.

    ``recent_messages`` and ``previous_evidence`` are used ONLY when the
    message is a context-dependent follow-up. A standalone question is left
    to stand on its own (plus goal/vehicle/facts).
    """
    message = (message or "").strip()
    followup = is_context_dependent_followup(message)

    segments: list[str] = []
    if followup:
        previous_question = last_meaningful_user_message(
            recent_messages, exclude=message
        )
        hint = evidence_topic_hint(previous_evidence)
        if state.active_topic:
            segments.append(f"текущая тема: {state.active_topic.strip()}")
        if previous_question:
            segments.append(f"предыдущий вопрос: {previous_question}")
        if hint:
            segments.append(f"источник темы: {hint}")

    segments.append(message)

    # The goal is useful context for both standalone and follow-up questions,
    # but avoid repeating the active topic verbatim.
    if state.goal and normalize(state.goal).lower() != normalize(
        state.active_topic or ""
    ).lower():
        segments.append(f"цель: {state.goal.strip()}")
    # NOTE: active_topic is intentionally injected ONLY for follow-ups. A
    # standalone question starts a (possibly new) topic and must not be
    # polluted by the previous one.

    segments.append(_vehicle_phrase(state))

    facts = [f.text.strip() for f in state.known_facts if f.text.strip()]
    if facts:
        segments.append("known context: " + "; ".join(facts[:max_facts]))
    if state.constraints:
        segments.append(
            "constraints: " + "; ".join(state.constraints[:max_constraints])
        )
    if state.terms:
        segments.append("terms: " + "; ".join(state.terms[:max_terms]))

    return "\n".join(segment for segment in segments if segment).strip()


_SWITCH_MARKERS = (
    "теперь ",
    "перейдём к ",
    "перейдем к ",
    "перейти к ",
    "другой вопрос",
    "новая проблема",
    "новая тема",
    "хочу обсудить ",
    "поговорим о ",
)


def _strip_switch_phrases(text: str) -> str:
    """Drop the "let's switch" wrapper and keep the new topic itself."""
    normalized = normalize(text).rstrip("?!. ")
    lowered = normalized.lower()
    cut = 0
    for marker in _SWITCH_MARKERS:
        index = lowered.rfind(marker)
        if index != -1:
            cut = max(cut, index + len(marker))
    result = normalized[cut:].strip(" .,;:—-–")
    return result or normalized


def derive_active_topic(
    question: str,
    previous_topic: str | None = None,
    *,
    evidence: Iterable[Any] | None = None,
    followup: bool = False,
    topic_switch: bool = False,
    max_len: int = 180,
) -> str | None:
    """Update the short active topic after a turn.

    * an explicit topic switch replaces it with the NEW topic;
    * a follow-up keeps the previous topic and only enriches it with the new
      source hint;
    * a standalone question becomes the new topic, enriched with the source.
    """
    hint = evidence_topic_hint(evidence)

    if topic_switch:
        base = _strip_switch_phrases(question)
    elif followup and previous_topic:
        base = previous_topic.strip()
    else:
        base = normalize(question).rstrip("?!. ")
        if not base:
            base = (previous_topic or "").strip()

    if hint and hint.lower() not in base.lower():
        combined = f"{base} — {hint}" if base else hint
    else:
        combined = base or hint
    combined = normalize(combined)
    return combined[:max_len] or None


def contextual_query_terms(message: str, state: TaskState) -> list[str]:
    """Return the task-state keywords a follow-up should carry into retrieval.

    Used by the scenario evaluator to assert that an incomplete follow-up was
    actually made self-contained through task memory.
    """
    terms: list[str] = []
    if state.active_topic:
        terms.append(state.active_topic)
    if state.vehicle.transmission:
        terms.append(state.vehicle.transmission)
    if state.vehicle.mileage_km is not None:
        terms.append(str(state.vehicle.mileage_km))
    for fact in state.known_facts[:4]:
        if fact.key:
            terms.append(fact.text)
    return terms
