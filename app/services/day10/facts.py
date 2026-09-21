"""Day 10 — Sticky Facts extraction (structured key/value memory).

The extraction is a SEPARATE, small LLM call run after every user message:

    current facts + new user message  ->  UPDATED facts (strict JSON)

It is explicitly NOT a summary: the model must return the COMPLETE updated
memory as structured JSON (goal / constraints / preferences / decisions /
agreements), merging the new message into the previous state and REPLACING
outdated values (e.g. "we changed our mind, use MySQL" removes PostgreSQL).

Everything here is pure and unit-testable: prompt building, robust JSON
parsing and normalisation. The provider call lives in the service.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.schemas.day10 import Facts

FACTS_SYSTEM_PROMPT = (
    "You maintain a structured MEMORY of an ongoing conversation between a "
    "user and an assistant. You are NOT a summarizer: never retell the dialog. "
    "Store only durable, meaningful facts.\n\n"
    "You receive the CURRENT MEMORY as JSON and ONE new user message. Return "
    "the COMPLETE updated memory as a single strict JSON object with exactly "
    "these keys:\n"
    "  goal        — string or null: the overall objective (update if it changed)\n"
    "  constraints — list of strings: hard requirements/limits the user stated\n"
    "  preferences — list of strings: softer wishes/preferences\n"
    "  decisions   — list of strings: concrete choices that were made\n"
    "  agreements  — list of strings: commitments/deadlines/agreements\n\n"
    "Rules:\n"
    "- Merge the new message into the existing memory.\n"
    "- If the user CHANGES a decision, REPLACE the old value; never keep both.\n"
    "- Do not duplicate an item that is already present.\n"
    "- Keep each item short (a name, a value, a requirement).\n"
    "- Leave a category empty when there is nothing durable to store.\n"
    "- Output ONLY the JSON object, with no markdown fences and no prose.\n"
)

# Keys accepted in the model's JSON output.
_FACT_KEYS = ("goal", "constraints", "preferences", "decisions", "agreements")
# Safety cap so a runaway model cannot bloat the memory indefinitely.
_MAX_ITEMS_PER_CATEGORY = 30
_MAX_ITEM_LENGTH = 300
_MAX_GOAL_LENGTH = 600


def facts_to_json(facts: Facts) -> str:
    """Canonical JSON representation of the current memory."""
    return json.dumps(facts.model_dump(), ensure_ascii=False, indent=2)


def build_facts_prompt(facts: Facts, new_message: str) -> list[dict[str, str]]:
    """Build the messages for ONE facts-extraction call (bounded payload)."""
    user_content = (
        "CURRENT MEMORY (JSON):\n"
        + facts_to_json(facts)
        + "\n\nNEW USER MESSAGE:\n"
        + new_message
        + "\n\nReturn the complete updated memory as strict JSON now."
    )
    return [
        {"role": "system", "content": FACTS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _strip_code_fences(text: str) -> str:
    """Remove a leading ```json / ``` fence if the model added one."""
    cleaned = text.strip()
    match = re.match(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of the first top-level JSON object."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in facts extraction output")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unterminated JSON object in facts extraction output")


def _coerce_string_list(value: Any) -> list[str]:
    """Normalise an arbitrary JSON value into a de-duplicated string list."""
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
        if len(out) >= _MAX_ITEMS_PER_CATEGORY:
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


def parse_facts_json(raw: str) -> Facts:
    """Parse the extraction output into a :class:`Facts` object.

    Raises ``ValueError`` when the output is not a JSON object. Robust against
    markdown fences and surrounding prose.
    """
    if raw is None:
        raise ValueError("empty facts extraction output")
    text = _strip_code_fences(raw)
    if not text:
        raise ValueError("empty facts extraction output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_extract_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("facts extraction output is not a JSON object")
    return Facts(
        goal=_coerce_goal(data.get("goal")),
        constraints=_coerce_string_list(data.get("constraints")),
        preferences=_coerce_string_list(data.get("preferences")),
        decisions=_coerce_string_list(data.get("decisions")),
        agreements=_coerce_string_list(data.get("agreements")),
    )
