"""Day 25 task-state updater.

Pipeline::

    Previous Task State + New User Message
              ↓  (extractor proposes operations)
        StateOperation[]
              ↓  (pure merge, app/services/day25/task_state.py)
        Updated Task State

Two extractors are provided:

* :class:`LLMTaskStateExtractor` — the default for real runs. It asks the
  generation model for a strict JSON list of operations and NEVER invents a
  *confirmed* fact: a guess must be returned as ``add_hypothesis``.
* :class:`RuleBasedTaskStateExtractor` — deterministic, offline fallback used
  when the LLM is unavailable and by tests.

Both only *propose* operations; the merge logic is pure and separately
tested. If extraction fails, the updater returns the PREVIOUS valid state and
records the error — a broken update never wipes memory (Day 25 requirement 37).
"""
from __future__ import annotations

import json
import logging
import re

from app.schemas.day25 import (
    TURN_CORRECTION,
    TURN_FACT_UPDATE,
    TURN_GOAL_CHANGE,
    TURN_QUESTION,
    StateOperation,
    TaskState,
    TaskUpdateInfo,
)
from app.services.day25.task_state import apply_operations
from app.services.day25.turn_analysis import (
    is_context_dependent_followup,
    is_explicit_topic_switch,
    is_generic_followup_only,
    normalize,
)
from app.services.rag.grounding.service import extract_json_object

logger = logging.getLogger("app.services.day25.updater")

# Cues used to classify a turn and (in the rule-based extractor) to detect
# hypotheses / corrections. Kept short and explicit — no heavy classifier.
_HYPOTHESIS_CUES = (
    "наверное",
    "возможно",
    "может быть",
    "может,",
    "похоже",
    "подозреваю",
    "предположу",
    "предполагаю",
    "скорее всего",
    "думаю",
)
_CORRECTION_CUES = (
    "исправляю",
    "исправляюсь",
    "на самом деле",
    "вообще-то",
    "поправка",
    "перепроверил",
    "посмотрел документы",
    "уточняю",
    "я ошибся",
)
_CHECK_CUES = (
    "заменил",
    "заменила",
    "поменял",
    "поменяла",
    "проверил",
    "проверила",
    "почистил",
    "почистила",
    "замерил",
    "замерила",
    "диагностировал",
    "смотрел",
    "смотрела",
    "отрегулировал",
)
_RESULT_CUES = (
    "осталась",
    "осталось",
    "остался",
    "не помогло",
    "без изменений",
    "пропала",
    "пропало",
    "исчезла",
    "исчезло",
    "стало лучше",
    "стало хуже",
    "усилилась",
    "усилилось",
    "не изменилось",
)
_CONSTRAINT_CUES = (
    "не могу",
    "бюджет",
    "срочно",
    "на гарантии",
    "гарантия",
    "нет возможности",
)
# Goal cues intentionally EXCLUDE follow-up phrases ("что делать", "почему"):
# those continue the current task and must not create/replace a goal.
_GOAL_CUES = (
    "хочу разобраться",
    "хочу понять",
    "помогите разобраться",
    "подскажите",
    "нужно разобраться",
    "разобраться",
)

# "Пробег 82 тысячи", "пробег 92000 км", "82 000 км"
_MILEAGE_RE = re.compile(
    r"(?:пробег|пробега|пробегом)[^\d]{0,14}(\d[\d\s]{0,9})"
    r"\s*(тыс(?:яч\w*)?|к|km|км)?",
    re.IGNORECASE,
)
# "не 82, а 92 тысячи" — explicit correction pattern, use the SECOND number.
_MILEAGE_CORRECTION_RE = re.compile(
    r"не\s+(\d[\d\s]{0,9})\D{0,12}а\s+(\d[\d\s]{0,9})\s*(тыс(?:яч\w*)?|к|km|км)?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_ENGINE_RE = re.compile(r"\b(\d\.\d)\s*(?:l|л|литр\w*)?", re.IGNORECASE)
# Ambient temperature ("на улице -25", "мороз -30"); a durable fact even when
# the same message is also a question.
_TEMP_RE = re.compile(
    r"(?:на улице|температура|мороз\w*|на морозе)[^\d-]{0,12}(-?\d{1,2})",
    re.IGNORECASE,
)


def _clean_number(raw: str) -> int | None:
    digits = re.sub(r"\D", "", raw or "")
    return int(digits) if digits else None


def _normalize_mileage(raw: str, unit: str | None) -> int | None:
    value = _clean_number(raw)
    if value is None:
        return None
    unit = (unit or "").lower()
    if unit.startswith("тыс") or unit in ("к",):
        value *= 1000
    return value


class RuleBasedTaskStateExtractor:
    """Deterministic extractor for the common patterns used in the demo."""

    name = "rules"

    async def extract(
        self, state: TaskState, message: str
    ) -> list[StateOperation]:
        text = (message or "").strip()
        lowered = text.lower()
        operations: list[StateOperation] = []

        is_hypothesis = any(cue in lowered for cue in _HYPOTHESIS_CUES)
        is_correction = any(cue in lowered for cue in _CORRECTION_CUES)
        topic_switch = is_explicit_topic_switch(text)
        followup = is_context_dependent_followup(text)

        # --- goal -----------------------------------------------------
        # Explicit switch: a genuinely new task.
        if topic_switch:
            operations.append(StateOperation(op="set_goal", text=text))
            operations.append(StateOperation(op="clear_active_topic"))
        # First substantive message becomes the goal; bare follow-ups never
        # do, even if they are the very first message.
        elif (
            not state.goal
            and not is_generic_followup_only(text)
            and len(normalize(text).split()) >= 2
        ):
            operations.append(StateOperation(op="set_goal", text=text))
        elif not state.goal and any(cue in lowered for cue in _GOAL_CUES):
            operations.append(StateOperation(op="set_goal", text=text))

        # --- vehicle facts -------------------------------------------
        correction_match = _MILEAGE_CORRECTION_RE.search(text)
        if correction_match:
            mileage = _normalize_mileage(
                correction_match.group(2), correction_match.group(3)
            )
        else:
            mileage_match = _MILEAGE_RE.search(text)
            mileage = (
                _normalize_mileage(mileage_match.group(1), mileage_match.group(2))
                if mileage_match
                else None
            )
        if mileage:
            operations.append(
                StateOperation(
                    op="update_vehicle", field="mileage_km", value=mileage
                )
            )

        year_match = _YEAR_RE.search(text)
        if year_match and any(
            word in lowered for word in ("год", "года", "году", "выпуск")
        ):
            operations.append(
                StateOperation(
                    op="update_vehicle",
                    field="year",
                    value=int(year_match.group(1)),
                )
            )

        if "cvt" in lowered or "вариатор" in lowered:
            operations.append(
                StateOperation(op="update_vehicle", field="transmission", value="CVT")
            )
        elif "механик" in lowered or "мкпп" in lowered:
            operations.append(
                StateOperation(op="update_vehicle", field="transmission", value="MT")
            )
        elif "автомат" in lowered or "акпп" in lowered:
            operations.append(
                StateOperation(op="update_vehicle", field="transmission", value="AT")
            )

        engine_match = _ENGINE_RE.search(text)
        if engine_match and "двигател" in lowered:
            operations.append(
                StateOperation(
                    op="update_vehicle",
                    field="engine",
                    value=f"{engine_match.group(1)} л",
                )
            )

        temp_match = _TEMP_RE.search(text)
        if temp_match:
            operations.append(
                StateOperation(
                    op="add_fact",
                    text=f"температура: {temp_match.group(1)}",
                    key="temperature",
                )
            )

        # --- constraints / terms -------------------------------------
        for cue in _CONSTRAINT_CUES:
            if cue in lowered:
                operations.append(StateOperation(op="add_constraint", text=text))
                break

        # --- hypothesis vs check/result ------------------------------
        if is_hypothesis:
            operations.append(StateOperation(op="add_hypothesis", text=text))
        else:
            if any(cue in lowered for cue in _CHECK_CUES):
                operations.append(StateOperation(op="add_check", text=text))
            if any(cue in lowered for cue in _RESULT_CUES):
                operations.append(StateOperation(op="add_result", text=text))
            if any(cue in lowered for cue in _CHECK_CUES):
                # A performed check is also a durable fact about the vehicle.
                operations.append(
                    StateOperation(op="add_fact", text=text, key="check:" + text[:40])
                )

        # --- plain fact fallback -------------------------------------
        if not operations and text and ("?" not in text):
            operations.append(
                StateOperation(op="add_fact", text=text, key="fact:" + text[:40])
            )

        if is_correction and not any(
            op.op in ("update_fact", "supersede_fact") for op in operations
        ):
            # The correction cue means "replace what you had", so the keyed
            # vehicle updates above already supersede; mark a generic fact
            # correction for the turn classifier.
            operations.append(
                StateOperation(
                    op="update_fact",
                    text=text,
                    key="correction:" + text[:40],
                    origin="user",
                )
            )
        return operations


class LLMTaskStateExtractor:
    """Ask the generation model for a strict JSON list of state operations."""

    name = "llm"

    SYSTEM_PROMPT = (
        "Ты ведёшь структурированную память задачи (task state) в диалоге о "
        "Mitsubishi Xpander. Ты НЕ отвечаешь на вопрос и НЕ ставишь диагноз. "
        "Твоя единственная задача — вернуть список операций, которые нужно "
        "применить к уже известному состоянию по новому сообщению "
        "пользователя.\n\n"
        "Правила:\n"
        "- НЕ изменяй goal только потому, что пользователь задал уточняющий "
        "вопрос. Вопросы 'как установить?', 'что делать?', 'а дальше?', "
        "'почему?', 'куда крепить?', 'какой тогда?', 'это подходит?' — это "
        "продолжение ТЕКУЩЕЙ цели, а не новая цель.\n"
        "- update_goal разрешён ТОЛЬКО при явной смене задачи: 'с этим "
        "закончили', 'теперь хочу разобраться с...', 'другой вопрос', "
        "'новая проблема', 'перейдём к...'.\n"
        "- Поддерживай active_topic — короткую текущую тему разговора "
        "(например 'защита картера и КПП Sheriff 5307/5308'). Обновляй её "
        "через set_active_topic. При явной смене задачи вызывай "
        "clear_active_topic и выставь новую тему.\n"
        "- Факт можно добавлять только если пользователь сообщил его прямо.\n"
        "- Предположение ('наверное', 'возможно', 'может быть') — это всегда "
        "add_hypothesis, а НЕ add_fact и НЕ причина.\n"
        "- Если пользователь исправляет ранее известное значение "
        "('исправляю', 'на самом деле', 'посмотрел документы'), используй "
        "update_vehicle или update_fact с тем же key — старое значение должно "
        "быть заменено, а не добавлено вторым.\n"
        "- Цель формулируй через set_goal/update_goal.\n"
        "- Ничего не выдумывай. Если операций нет — верни пустой список.\n\n"
        "Верни СТРОГО JSON-объект:\n"
        '{"operations": [{"op": "...", "text": "...", "key": null, '
        '"field": null, "value": null, "origin": "user"}]}\n'
        "Допустимые op: set_goal, update_goal, add_fact, update_fact, "
        "supersede_fact, remove_fact, set_active_topic, clear_active_topic, "
        "add_constraint, add_term, add_check, add_result, add_hypothesis, "
        "update_vehicle, add_open_question, remove_open_question. Для "
        "update_vehicle field: model, year, engine, transmission, mileage_km."
    )

    def __init__(self, generation, *, temperature: float = 0.0, max_tokens: int = 700):
        self.generation = generation
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def extract(
        self, state: TaskState, message: str
    ) -> list[StateOperation]:
        state_json = json.dumps(
            state.model_dump(), ensure_ascii=False, indent=2
        )
        user = (
            "ТЕКУЩЕЕ СОСТОЯНИЕ (JSON):\n"
            f"{state_json}\n\n"
            "НОВОЕ СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n"
            f"{message}\n\n"
            "Верни только JSON с operations."
        )
        result = await self.generation.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        payload = extract_json_object(result.content)
        if payload is None:
            raise ValueError("task-state extractor returned invalid JSON")
        raw_ops = payload.get("operations")
        if raw_ops is None:
            return []
        if not isinstance(raw_ops, list):
            raise ValueError("operations must be a list")
        operations: list[StateOperation] = []
        for item in raw_ops:
            if not isinstance(item, dict):
                continue
            try:
                operations.append(StateOperation.model_validate(item))
            except Exception:  # noqa: BLE001 - skip malformed op, keep the rest
                logger.warning("Skipping malformed state operation: %s", item)
        return operations


def classify_turn(
    message: str,
    operations: list[StateOperation],
    previous_goal: str | None = None,
) -> str:
    """Lightweight turn classification derived from the message + operations.

    ``set_goal`` / ``update_goal`` alone are NOT a goal change: the first goal
    and refinements continue the same task. A real change requires an explicit
    switch signal.
    """
    kinds = {op.op for op in operations}
    lowered = (message or "").lower()
    if is_explicit_topic_switch(message):
        return TURN_GOAL_CHANGE
    if kinds & {"update_fact", "supersede_fact", "remove_fact"}:
        return TURN_CORRECTION
    if any(cue in lowered for cue in _CORRECTION_CUES):
        return TURN_CORRECTION
    if "?" in message:
        return TURN_QUESTION
    if kinds & {
        "add_fact",
        "update_vehicle",
        "add_check",
        "add_result",
        "add_hypothesis",
    }:
        return TURN_FACT_UPDATE
    return TURN_QUESTION


class TaskStateUpdater:
    """Apply an extractor's operations with a safe fallback to the old state."""

    def __init__(self, extractor):
        self.extractor = extractor

    async def update(
        self, previous: TaskState, message: str
    ) -> tuple[TaskState, TaskUpdateInfo]:
        try:
            operations = await self.extractor.extract(previous, message)
        except Exception as exc:  # noqa: BLE001 - never block the chat
            logger.warning("Task-state update failed: %s", exc)
            return previous, TaskUpdateInfo(
                applied_operations=0,
                operations=[],
                turn_type=classify_turn(message, [], previous.goal),
                error=str(exc),
            )
        updated = apply_operations(previous, operations)
        return updated, TaskUpdateInfo(
            applied_operations=len(operations),
            operations=operations,
            turn_type=classify_turn(message, operations, previous.goal),
            error=None,
        )
