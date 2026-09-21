"""Day 14 — invariant definitions, prompt block builder and conflict detection.

This module is the SINGLE place that:

* defines the default active invariants;
* converts invariants into the ``ACTIVE INVARIANTS`` model instructions;
* detects (deterministically, in code) when a user request conflicts with one
  or more invariants.

Design choices (kept intentionally small — no policy engine / DSL):

* an invariant is ``{id, category, rule}``;
* conflict detection uses a small, explicit per-invariant trigger list;
* matches are returned for EVERY violated invariant, so a single request can
  report several conflicts.

Code performs the validation so the refusal does not depend on the wording of
the LLM. The invariants are also injected into the model request, so the model
respects them when it DOES answer.
"""
from __future__ import annotations

import re

from app.schemas.day14 import (
    CATEGORY_LABELS,
    Invariant,
    InvariantConflict,
)

# ----------------------------------------------------------------------
# Default active invariants
# ----------------------------------------------------------------------
DEFAULT_INVARIANTS: list[Invariant] = [
    Invariant(
        id="architecture",
        category="architecture",
        rule="Backend должен оставаться монолитным FastAPI-приложением",
    ),
    Invariant(
        id="database",
        category="technical_decision",
        rule="Использовать PostgreSQL",
    ),
    Invariant(
        id="stack",
        category="stack",
        rule="Backend реализуется на Python",
    ),
    Invariant(
        id="business_rule",
        category="business",
        rule=(
            "Удаление пользовательских данных требует явного подтверждения "
            "пользователя"
        ),
    ),
]

# ----------------------------------------------------------------------
# Conflict triggers
# ----------------------------------------------------------------------
# For every invariant id, the fragments that indicate a request is trying to
# violate it. Pure single tokens are matched case-insensitively with word
# boundaries; phrases (containing a space) are matched as substrings.
_CONFLICT_TRIGGERS: dict[str, list[str]] = {
    # "разбей ... на микросервисы", "уйди от монолита", ...
    "architecture": [
        "микросервис",
        "microservice",
        "micro-service",
        "разбить на сервисы",
        "разделить на сервисы",
        "разделение на сервисы",
        "откажись от монолит",
        "уйти от монолит",
    ],
    # any database other than PostgreSQL
    "database": [
        "mongodb",
        "mysql",
        "sqlite",
        "oracle",
        "dynamodb",
        "cassandra",
    ],
    # any backend language other than Python
    "stack": [
        "golang",
        "java",
        "rust",
        "php",
        "nodejs",
        "node.js",
        "kotlin",
        "swift",
        "ruby",
        "go",
        "c#",
        "c++",
    ],
    # deleting user data without confirmation
    "business_rule": [
        "без подтверждения",
        "без явного подтверждения",
        "сразу удал",
        "немедленно удал",
        "удали без",
        "удалить без",
    ],
}


def _matches(text: str, trigger: str) -> bool:
    """True when ``trigger`` occurs in ``text``.

    Single alphanumeric tokens use word boundaries (so ``go`` does not match
    inside ``google``); anything else (phrases with spaces, ``c#``, ``c++``)
    is matched as a case-insensitive substring.
    """
    if re.fullmatch(r"[a-z0-9]+", trigger):
        return re.search(
            rf"(?<![a-z0-9]){re.escape(trigger)}(?![a-z0-9])", text
        ) is not None
    return trigger in text


def detect_conflicts(message: str, invariants: list[Invariant]) -> list[InvariantConflict]:
    """Return EVERY invariant the request conflicts with (order preserved)."""
    lowered = (message or "").lower()
    conflicts: list[InvariantConflict] = []
    for invariant in invariants:
        triggers = _CONFLICT_TRIGGERS.get(invariant.id, [])
        matched = [trigger for trigger in triggers if _matches(lowered, trigger)]
        if matched:
            conflicts.append(
                InvariantConflict(
                    invariant_id=invariant.id,
                    category=invariant.category,
                    category_label=invariant.category_label,
                    rule=invariant.rule,
                    matched=matched,
                )
            )
    return conflicts


# ----------------------------------------------------------------------
# Prompt block
# ----------------------------------------------------------------------
_INVARIANT_INSTRUCTIONS = [
    "When answering:",
    "- respect every invariant;",
    "- do not propose solutions that violate them;",
    "- if the user's request conflicts with an invariant, do not follow the "
    "conflicting part;",
    "- explicitly identify the conflicting invariant;",
    "- explain why the request cannot be fulfilled under the current "
    "constraints;",
    "- when reasonable, suggest an alternative that stays within the "
    "invariants.",
]


def build_invariants_block(invariants: list[Invariant]) -> str:
    """Build the mandatory ``ACTIVE INVARIANTS`` block for a model request."""
    lines = [
        "ACTIVE INVARIANTS",
        "=================",
        "",
        "These constraints are mandatory and cannot be violated.",
        "",
    ]
    if invariants:
        for index, invariant in enumerate(invariants, 1):
            lines.append(
                f"{index}. [{invariant.category}] {invariant.rule}"
            )
    else:
        lines.append("(нет активных инвариантов)")
    lines.append("")
    lines.extend(_INVARIANT_INSTRUCTIONS)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Refusal explanation
# ----------------------------------------------------------------------
def _format_conflict(conflict: InvariantConflict, index: int) -> str:
    found = ", ".join(f'"{m}"' for m in conflict.matched)
    detail = f" (совпадение: {found})" if found else ""
    return (
        f"{index}. {conflict.category_label} — {conflict.rule}{detail}"
    )


def build_refusal(conflicts: list[InvariantConflict]) -> str:
    """Build the explicit refusal that names every conflicting invariant."""
    lines = [
        "Запрос конфликтует с активными инвариантами:",
        "",
    ]
    lines.extend(
        _format_conflict(conflict, index)
        for index, conflict in enumerate(conflicts, 1)
    )
    lines += [
        "",
        "Поэтому я не могу выполнить эту часть запроса: она нарушает "
        "перечисленные ограничения.",
        "",
        "В рамках текущих инвариантов можно продолжить реализацию на "
        "Python/FastAPI с PostgreSQL и сохранить монолитную архитектуру; "
        "удаление пользовательских данных — только с явным подтверждением.",
    ]
    return "\n".join(lines)


# Re-exported for callers that only need the label map.
__all__ = [
    "CATEGORY_LABELS",
    "DEFAULT_INVARIANTS",
    "build_invariants_block",
    "build_refusal",
    "detect_conflicts",
]
