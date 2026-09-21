"""Day 11 — deterministic demo scenario and demo evaluation metric.

The scenario states durable facts EARLY (identity, preferences, task,
constraint), then sends several acknowledgements that must NOT be stored
(``nothing_to_save``), then asks a question. With a small sliding window the
opening message leaves the short-term context, so the answer can only keep the
requirements through working / long-term memory.

The evaluation is explicitly a DEMO metric (keyword match), not a benchmark.
"""
from __future__ import annotations

import re

SCENARIO_MESSAGES: list[str] = [
    "Я Антон. Обычно пишу backend на Python и предпочитаю короткие ответы. "
    "Сейчас проектируем сервис бронирования. PostgreSQL в этой задаче "
    "использовать нельзя.",
    "Спасибо, продолжаем.",
    "Хорошо.",
    "Давай обсудим общие принципы проектирования, без деталей.",
    "Продолжай.",
    "Спасибо, полезно.",
    "Предложи архитектуру хранения данных с учётом моих требований.",
]

# Expected demo facts: label -> case-insensitive substrings.
EXPECTED_FACTS: list[tuple[str, list[str]]] = [
    ("Python", ["python", "питон"]),
    ("Краткие ответы", ["кратк", "коротк", "concise"]),
    ("Бронирование", ["брониров"]),
    ("PostgreSQL запрещён", ["postgresql", "постгрес", "постгре"]),
]

EXPECTED_FACTS_LABELS: list[str] = [label for label, _ in EXPECTED_FACTS]


def evaluate_answer(answer: str, expected: list[str] | None = None) -> dict:
    """Simple, transparent DEMO evaluation metric (keyword match)."""
    text = (answer or "").casefold()
    wanted = expected if expected is not None else EXPECTED_FACTS_LABELS
    lookup = {label: patterns for label, patterns in EXPECTED_FACTS if label in wanted}
    for label in wanted:
        lookup.setdefault(label, [label.casefold()])

    matched: list[str] = []
    missing: list[str] = []
    for label in wanted:
        patterns = lookup.get(label, [label.casefold()])
        if any(re.search(re.escape(p.casefold()), text) for p in patterns):
            matched.append(label)
        else:
            missing.append(label)
    return {
        "matched": matched,
        "missing": missing,
        "score": len(matched),
        "total": len(wanted),
    }
