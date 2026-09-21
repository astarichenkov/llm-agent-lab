"""Day 11 — MemoryClassifier / MemoryProcessor.

After every user message the system must decide explicitly WHAT (if anything)
to store in working / long-term memory. Short-term memory is NEVER classified
by the LLM: it is filled deterministically from the current session.

The classifier is a small, separate LLM call that returns strict structured
JSON:

    current working memory + current long-term memory + new user message
        -> { nothing_to_save, working_memory{...}, long_term_memory{...} }

It is explicitly allowed (and expected) to return ``nothing_to_save`` for
acknowledgements like «спасибо» / «хорошо» / «продолжай», so they do not
pollute durable memory.

Everything here is pure and unit-testable (prompt building, parsing,
normalisation). The provider call is injected, so tests never hit the API.
"""
from __future__ import annotations

import logging
from typing import Any

from app.schemas.day11 import ClassifierDecision, LongTermMemory, WorkingMemory
from app.services.structured_json import parse_json_object

logger = logging.getLogger("app.services.day11.classifier")

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a MEMORY MANAGER for an assistant. Your ONLY job is to decide "
    "what must be remembered. You are NOT the assistant and you never answer "
    "the user.\n\n"
    "You receive the CURRENT WORKING MEMORY (state of the current task), the "
    "CURRENT LONG-TERM MEMORY (durable preferences across sessions) and ONE "
    "new user message.\n\n"
    "Return a SINGLE strict JSON object with exactly these keys:\n"
    "{\n"
    '  "nothing_to_save": true|false,\n'
    '  "working_memory": {\n'
    '    "goal": string|null,\n'
    '    "constraints": [string],\n'
    '    "requirements": [string],\n'
    '    "decisions": [string],\n'
    '    "data": {string: string}\n'
    "  },\n"
    '  "long_term_memory": {string: string}\n'
    "}\n\n"
    "Rules:\n"
    "- working_memory = facts about the CURRENT TASK ONLY (goal, hard "
    "constraints, requirements, decisions, small structured data such as a "
    "chosen stack). Put only NEW or CHANGED values.\n"
    "- long_term_memory = durable facts useful in FUTURE sessions: stable user "
    "preferences (language, answer style), long-lived technical choices. Use "
    "short snake_case keys, e.g. preferred_language / answer_style.\n"
    "- Do NOT copy whole sentences. Do NOT store small talk.\n"
    "- If the message is an acknowledgement or contains nothing durable, set "
    'nothing_to_save = true and leave both objects empty.\n'
    "- Never invent facts the user did not state.\n"
    "- Output ONLY the JSON object: no markdown fences, no prose.\n"
)

# Safety caps so a runaway model cannot bloat memory indefinitely.
_MAX_ITEMS = 30
_MAX_ITEM_LENGTH = 300
_MAX_GOAL_LENGTH = 600
_MAX_ENTRIES = 50


def working_to_json(working: WorkingMemory) -> str:
    import json

    return json.dumps(working.model_dump(), ensure_ascii=False, indent=2)


def long_term_to_json(long_term: LongTermMemory) -> str:
    import json

    return json.dumps(long_term.entries, ensure_ascii=False, indent=2)


def build_classifier_prompt(
    working: WorkingMemory, long_term: LongTermMemory, user_message: str
) -> list[dict[str, str]]:
    """Build the messages for ONE memory-classification call."""
    user_content = (
        "CURRENT WORKING MEMORY (JSON):\n"
        + working_to_json(working)
        + "\n\nCURRENT LONG-TERM MEMORY (JSON):\n"
        + long_term_to_json(long_term)
        + "\n\nNEW USER MESSAGE:\n"
        + user_message
        + "\n\nReturn the JSON decision now."
    )
    return [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ----------------------------------------------------------------------
# normalisation
# ----------------------------------------------------------------------
def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        if len(text) > _MAX_ITEM_LENGTH:
            text = text[:_MAX_ITEM_LENGTH].rstrip()
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= _MAX_ITEMS:
            break
    return out


def _coerce_goal(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > _MAX_GOAL_LENGTH:
        text = text[:_MAX_GOAL_LENGTH].rstrip()
    return text


def _coerce_string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, item in value.items():
        name = str(key).strip()
        if not name or item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        out[name[:100]] = text[:_MAX_ITEM_LENGTH]
        if len(out) >= _MAX_ENTRIES:
            break
    return out


def parse_decision(raw: str) -> ClassifierDecision:
    """Parse the classifier output into a :class:`ClassifierDecision`.

    Raises ``ValueError`` when no JSON object can be parsed.
    """
    data = parse_json_object(raw)
    working_raw = data.get("working_memory")
    if not isinstance(working_raw, dict):
        working_raw = {}
    working = WorkingMemory(
        goal=_coerce_goal(working_raw.get("goal")),
        constraints=_coerce_string_list(working_raw.get("constraints")),
        requirements=_coerce_string_list(working_raw.get("requirements")),
        decisions=_coerce_string_list(working_raw.get("decisions")),
        data=_coerce_string_dict(working_raw.get("data")),
    )
    long_term = _coerce_string_dict(data.get("long_term_memory"))

    explicit_nothing = bool(data.get("nothing_to_save", False))
    has_updates = not working.is_empty() or bool(long_term)
    # Respect an explicit nothing_to_save; otherwise save iff there is a delta.
    nothing_to_save = explicit_nothing or not has_updates
    if nothing_to_save:
        working = WorkingMemory()
        long_term = {}

    return ClassifierDecision(
        performed=True,
        nothing_to_save=nothing_to_save,
        working_memory=working,
        long_term_memory=long_term,
    )


# ----------------------------------------------------------------------
# classifier
# ----------------------------------------------------------------------
class MemoryClassifier:
    """Decides what to persist into working / long-term memory."""

    name = "memory_classifier"

    async def classify(
        self,
        *,
        user_message: str,
        working: WorkingMemory,
        long_term: LongTermMemory,
        provider: Any,
        model: str,
        max_tokens: int = 800,
    ) -> tuple[ClassifierDecision, dict]:
        """Run one classification call with graceful degradation.

        Returns ``(decision, usage)``. A failure NEVER raises: the decision
        carries ``error`` and ``performed=False`` so the main chat can continue
        and report it. The ``usage`` dict is empty when no call succeeded.
        """
        payload = build_classifier_prompt(working, long_term, user_message)
        try:
            content, _finish, usage = await provider.generate(
                payload,
                model=model,
                max_tokens=max_tokens,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001 - graceful degradation required
            logger.warning(
                "day11 memory classification failed (%s): %s", type(exc).__name__, exc
            )
            return (
                ClassifierDecision(
                    performed=False,
                    nothing_to_save=True,
                    error=(
                        "MemoryClassifier недоступен (ошибка обращения к модели). "
                        "Working/Long-term память в этом запросе не обновлялась."
                    ),
                ),
                {},
            )

        if not (content or "").strip():
            logger.warning("day11 memory classification returned empty content")
            return (
                ClassifierDecision(
                    performed=False,
                    nothing_to_save=True,
                    error=(
                        "MemoryClassifier вернул пустой ответ. Память не обновлялась."
                    ),
                ),
                usage or {},
            )

        try:
            decision = parse_decision(content)
        except ValueError as exc:
            logger.warning("day11 memory classification JSON parse failed: %s", exc)
            return (
                ClassifierDecision(
                    performed=False,
                    nothing_to_save=True,
                    error=(
                        "MemoryClassifier вернул некорректный JSON. Память не "
                        "обновлялась."
                    ),
                ),
                usage or {},
            )

        return decision, usage or {}
