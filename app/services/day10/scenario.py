"""Day 10 — deterministic demo scenarios.

Two scenarios live here:

* the shared **TS scenario** (15 user messages) used for BOTH Sliding Window
  and Sticky Facts. It states durable requirements EARLY, then discusses other
  things, then asks for the final spec — so a small sliding window objectively
  loses the early details while Sticky Facts keeps them;
* the **branching scenario**: a common prefix up to a checkpoint, plus two
  alternative branches (PostgreSQL/Redis vs MongoDB/no Redis).

The expected-facts evaluation is explicitly labelled a DEMO metric — it is a
simple keyword match, not an objective AI benchmark.
"""
from __future__ import annotations

import re

# ----------------------------------------------------------------------
# Shared TS scenario (10–15 messages)
# ----------------------------------------------------------------------
SCENARIO_MESSAGES: list[str] = [
    "Название проекта — HelpDesk, это внутренний сервис учёта заявок.",
    "Главная цель: разработать внутренний сервис учёта заявок для сотрудников.",
    "Backend обязательно должен быть на Python.",
    "Выбираем фреймворк FastAPI.",
    "Бюджет ограничен, поэтому минимум внешних сервисов.",
    "База данных — PostgreSQL.",
    "MVP должен быть готов за 4 недели.",
    "Пользователи: сотрудники создают заявки, администраторы их обрабатывают.",
    "Авторизация — через JWT.",
    "Деплой в Docker.",
    "Ограничение: запрещено использовать внешние облачные сервисы.",
    "Нужна ролевая модель: user и admin.",
    "Интерфейс — простой веб, без мобильного приложения.",
    "Логирование — только метаданные, без персональных данных.",
    "Составь итоговое ТЗ с учётом всех наших решений.",
]

# Expected demo facts: label -> list of case-insensitive substrings.
EXPECTED_FACTS: list[tuple[str, list[str]]] = [
    ("Python", ["python", "питон"]),
    ("FastAPI", ["fastapi", "фаст-апи", "фастапи"]),
    ("PostgreSQL", ["postgresql", "postgres", "постгрес", "постгре"]),
    ("JWT", ["jwt"]),
    ("Docker", ["docker", "докер"]),
    ("4 недели", ["4 недел", "четыре недел", "4-х недел", "4 недели"]),
    ("Учёт заявок", ["учёт заявок", "учет заявок", "учёта заявок", "helpdesk", "help desk"]),
]

EXPECTED_FACTS_LABELS: list[str] = [label for label, _ in EXPECTED_FACTS]

# The final control question in the TS scenario.
FINAL_QUESTION = SCENARIO_MESSAGES[-1]


def evaluate_answer(
    answer: str, expected: list[str] | None = None
) -> dict:
    """Simple, transparent DEMO evaluation metric.

    Matches expected labels against the answer with case-insensitive keyword
    search. Returns the matched/missing labels and a score. This is a teaching
    aid, not an objective model-quality benchmark.
    """
    text = (answer or "").casefold()
    wanted = expected if expected is not None else EXPECTED_FACTS_LABELS
    lookup = {
        label: patterns for label, patterns in EXPECTED_FACTS if label in wanted
    }
    # Allow caller-supplied labels with no pattern table: match the label text.
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


# ----------------------------------------------------------------------
# Branching scenario
# ----------------------------------------------------------------------
# Common prefix of the conversation (ends at a natural checkpoint).
BRANCHING_COMMON: list[dict[str, str]] = [
    {"role": "user", "content": "Нужен REST API для сервиса учёта заявок HelpDesk."},
    {"role": "assistant", "content": "Понял: проектируем REST API для HelpDesk."},
    {"role": "user", "content": "Backend на Python, фреймворк FastAPI."},
    {"role": "assistant", "content": "Принято: Python и FastAPI."},
    {"role": "user", "content": "Авторизация через JWT, деплой в Docker."},
    {"role": "assistant", "content": "Зафиксировано: JWT и Docker."},
    {"role": "user", "content": "Определили общий backend и API."},
    {"role": "assistant", "content": "Да, общий стек backend и API определён."},
]

# The decision each branch starts with, and the shared control question.
BRANCH_A = {
    "name": "PostgreSQL",
    "decision": "Используем PostgreSQL и Redis.",
}
BRANCH_B = {
    "name": "MongoDB",
    "decision": "Используем MongoDB, без Redis.",
}
BRANCH_CONTROL_QUESTION = "Собери итоговую архитектуру с учётом решений этой ветки."


def branching_setup() -> dict:
    """The branching demo definition exposed to the UI."""
    return {
        "common": list(BRANCHING_COMMON),
        "branches": [dict(BRANCH_A), dict(BRANCH_B)],
        "control_question": BRANCH_CONTROL_QUESTION,
    }
