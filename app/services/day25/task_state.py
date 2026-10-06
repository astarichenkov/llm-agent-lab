"""Deterministic task-state operations for Day 25.

This module is deliberately PURE: it takes a :class:`TaskState` and a list of
:class:`StateOperation` and returns a NEW state. It never calls an LLM. The
LLM (or a rule-based extractor) only *proposes* operations; the merge logic
below decides what the active state becomes.

Key invariants (documented in docs/week5/day25.md):

* a fact stated by the user is never promoted to a confirmed cause — a guess
  such as "наверное, виновата подушка" is stored as a HYPOTHESIS;
* a correction SUPERSEDES the previous value instead of keeping two
  contradicting active facts (mileage 80000 -> 92000);
* duplicates are collapsed, so repeating a fact does not grow the state.
"""
from __future__ import annotations

from app.schemas.day25 import (
    FactItem,
    StateOperation,
    TaskState,
    ValueItem,
    VEHICLE_DEFAULT_MODEL,
)

# Vehicle string fields are coerced to int when they carry a number.
_INT_VEHICLE_FIELDS = {"year", "mileage_km"}

# Synonyms keep the operation vocabulary forgiving for a rule-based or LLM
# extractor without changing the schema.
_VEHICLE_FIELD_ALIASES = {
    "модель": "model",
    "model": "model",
    "год": "year",
    "year": "year",
    "двигатель": "engine",
    "engine": "engine",
    "коробка": "transmission",
    "кпп": "transmission",
    "transmission": "transmission",
    "пробег": "mileage_km",
    "mileage": "mileage_km",
    "mileage_km": "mileage_km",
    "пробег_км": "mileage_km",
}


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).strip().casefold()


def _dedupe_append(items: list, text: str) -> None:
    normalized = _normalize(text)
    if not normalized:
        return
    for item in items:
        existing = item.text if hasattr(item, "text") else str(item)
        if _normalize(existing) == normalized:
            return
    items.append(text)


def _find_fact(state: TaskState, key: str | None, text: str = "") -> int | None:
    """Return the index of a matching fact (by key first, then by text)."""
    if key:
        wanted = key.strip().casefold()
        for index, fact in enumerate(state.known_facts):
            if fact.key and fact.key.strip().casefold() == wanted:
                return index
    normalized = _normalize(text)
    if normalized:
        for index, fact in enumerate(state.known_facts):
            if _normalize(fact.text) == normalized:
                return index
    return None


def _find_vehicle_field(field: str | None) -> str | None:
    if not field:
        return None
    return _VEHICLE_FIELD_ALIASES.get(field.strip().casefold())


def _coerce_vehicle_value(field: str, value) -> object:
    if field in _INT_VEHICLE_FIELDS:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else None
    return str(value).strip()


def _apply_vehicle(state: TaskState, field: str, value) -> None:
    coerced = _coerce_vehicle_value(field, value)
    if coerced in (None, ""):
        return
    setattr(state.vehicle, field, coerced)
    if field == "model":
        return
    # A recorded vehicle field is also a known fact, kept in sync so the UI
    # and the contextual query builder can rely on one place.
    label = {
        "year": "год выпуска",
        "engine": "двигатель",
        "transmission": "коробка передач",
        "mileage_km": "пробег",
    }.get(field, field)
    item = f"{label}: {coerced}"
    index = _find_fact(state, key=f"vehicle_{field}")
    if index is not None:
        state.known_facts[index] = FactItem(
            text=item, key=f"vehicle_{field}", origin="user"
        )
    else:
        state.known_facts.append(
            FactItem(text=item, key=f"vehicle_{field}", origin="user")
        )


def apply_operation(state: TaskState, op: StateOperation) -> None:
    """Apply ONE operation in place (the caller deep-copies the state first)."""
    kind = op.op
    text = (op.text or "").strip()
    origin = op.origin or "user"

    if kind in ("set_goal", "update_goal"):
        if text:
            state.goal = text

    elif kind in ("set_active_topic", "update_topic"):
        if text:
            state.active_topic = text

    elif kind == "clear_active_topic":
        state.active_topic = None

    elif kind == "add_fact":
        if not text:
            return
        # add_fact with an existing key is an UPDATE, not a duplicate.
        index = _find_fact(state, op.key, text)
        if index is not None:
            existing = state.known_facts[index]
            state.known_facts[index] = FactItem(
                text=text,
                key=op.key or existing.key,
                origin=origin,
            )
        else:
            state.known_facts.append(
                FactItem(text=text, key=op.key, origin=origin)
            )

    elif kind in ("update_fact", "supersede_fact"):
        index = _find_fact(state, op.key, op.text)
        if index is None:
            if text:
                state.known_facts.append(
                    FactItem(text=text, key=op.key, origin=origin)
                )
            return
        existing = state.known_facts[index]
        if kind == "supersede_fact" and not text:
            # Explicit removal of a stale value.
            state.known_facts.pop(index)
            return
        state.known_facts[index] = FactItem(
            text=text or existing.text,
            key=op.key or existing.key,
            origin=origin,
        )

    elif kind == "remove_fact":
        index = _find_fact(state, op.key, text)
        if index is not None:
            state.known_facts.pop(index)

    elif kind == "add_constraint":
        _dedupe_append(state.constraints, text)

    elif kind == "add_term":
        _dedupe_append(state.terms, text)

    elif kind == "add_check":
        _dedupe_value_append(state.checks_performed, text, origin)

    elif kind == "add_result":
        _dedupe_value_append(state.results, text, origin)

    elif kind == "add_hypothesis":
        _dedupe_value_append(state.hypotheses, text, origin)

    elif kind == "update_vehicle":
        field = _find_vehicle_field(op.field)
        if field is None and op.key:
            field = _find_vehicle_field(op.key)
        if field:
            _apply_vehicle(state, field, op.value if op.value is not None else text)

    elif kind == "add_open_question":
        _dedupe_append(state.open_questions, text)

    elif kind == "remove_open_question":
        normalized = _normalize(text)
        state.open_questions = [
            item for item in state.open_questions if _normalize(item) != normalized
        ]


def _dedupe_value_append(items: list[ValueItem], text: str, origin: str) -> None:
    normalized = _normalize(text)
    if not normalized:
        return
    for item in items:
        if _normalize(item.text) == normalized:
            return
    items.append(ValueItem(text=text, origin=origin))


def apply_operations(state: TaskState, operations: list[StateOperation]) -> TaskState:
    """Return a NEW state with all operations applied (input is not mutated)."""
    updated = state.model_copy(deep=True)
    for op in operations:
        apply_operation(updated, op)
    return updated


def default_task_state() -> TaskState:
    return TaskState(vehicle={"model": VEHICLE_DEFAULT_MODEL})


def state_to_prompt_text(state: TaskState, *, max_facts: int = 12) -> str:
    """Render the task state as a compact, deterministic text block."""
    lines: list[str] = []

    def line(label: str, value: str) -> None:
        lines.append(f"{label}: {value}")

    line("GOAL", state.goal or "(не определена)")
    if state.active_topic:
        line("ACTIVE TOPIC", state.active_topic)
    vehicle_bits = [state.vehicle.model or VEHICLE_DEFAULT_MODEL]
    if state.vehicle.year:
        vehicle_bits.append(f"год {state.vehicle.year}")
    if state.vehicle.engine:
        vehicle_bits.append(f"двигатель {state.vehicle.engine}")
    if state.vehicle.transmission:
        vehicle_bits.append(f"КПП {state.vehicle.transmission}")
    if state.vehicle.mileage_km is not None:
        vehicle_bits.append(f"пробег {state.vehicle.mileage_km} км")
    line("VEHICLE", ", ".join(vehicle_bits))

    def section(label: str, values: list[str]) -> None:
        if not values:
            return
        lines.append(f"{label}:")
        for value in values:
            lines.append(f"- {value}")

    section("KNOWN FACTS", [f.text for f in state.known_facts][:max_facts])
    section("CONSTRAINTS", state.constraints)
    section("TERMS", state.terms)
    section("CHECKS PERFORMED", [c.text for c in state.checks_performed])
    section("RESULTS", [r.text for r in state.results])
    section("HYPOTHESES (unconfirmed)", [h.text for h in state.hypotheses])
    section("OPEN QUESTIONS", state.open_questions)
    return "\n".join(lines)


def state_contains(state: TaskState, needle: str) -> bool:
    """True when ``needle`` appears anywhere in the ACTIVE serialized state."""
    target = _normalize(needle)
    if not target:
        return False
    haystack = _normalize(state.model_dump_json())
    return target in haystack
