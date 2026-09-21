"""Small, dependency-free helpers for parsing model JSON output.

LLMs frequently wrap a JSON object in markdown fences or add a short sentence
around it. These helpers make the strict-JSON prompt path robust without
pulling in a schema-validation library.

They are shared by Day 11's MemoryClassifier. (Day 10 already has an
equivalent private implementation; it is intentionally left untouched.)
"""
from __future__ import annotations

import json
from typing import Any


def strip_code_fences(text: str) -> str:
    """Remove a leading ```json / ``` fence if the model added one."""
    import re

    cleaned = (text or "").strip()
    match = re.match(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned


def extract_json_object(text: str) -> str:
    """Best-effort extraction of the first balanced top-level JSON object."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
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
    raise ValueError("unterminated JSON object in model output")


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse model output into a JSON object, tolerating fences/prose.

    Raises ``ValueError`` when no JSON object can be parsed.
    """
    if raw is None:
        raise ValueError("empty model output")
    text = strip_code_fences(raw)
    if not text:
        raise ValueError("empty model output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(extract_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data
