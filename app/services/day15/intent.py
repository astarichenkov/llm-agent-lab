"""Day 15 — natural-language intent recognition.

The LLM is used ONLY to understand WHAT the user wants. It returns a semantic
:class:`~app.schemas.day15.IntentResult` (``{"intent": ..., "confidence": ...}``)
and is never allowed to pick a ``target_state``. The application maps the
intent to an action and the :class:`~app.services.day15.lifecycle.LifecycleMachine`
decides whether that action is legal.

    User message
          │
          ▼
    LLM intent recognition   (this module)
          │
          ▼
    structured action  ("approve_plan")
          │
          ▼
    State Machine (allowed transitions + guards)
       ↙            ↘
    ALLOWED        BLOCKED
       ↓              ↓
    new state      old state

The context passed to the model ALWAYS contains the current state, the current
step, the expected action, the guard flags and the allowed transitions, so the
same phrase ("ok, go on") can be interpreted differently in different states.
"""
from __future__ import annotations

import logging

from app.schemas.day15 import (
    Day15TaskState,
    IntentResult,
)
from app.services.structured_json import parse_json_object

logger = logging.getLogger("app.services.day15.intent")

# Every intent the model may return. Anything else becomes ``normal_message``.
VALID_INTENTS: tuple[str, ...] = (
    "continue_planning",
    "approve_plan",
    "request_plan_changes",
    "revise_plan",
    "start_execution",
    "request_validation",
    "validation_passed",
    "validation_failed",
    "complete_task",
    "pause",
    "resume",
    "normal_message",
)

# Short descriptions that go into the prompt. The model chooses ONE of these.
INTENT_DESCRIPTIONS: dict[str, str] = {
    "continue_planning": "prepare, generate or redraft the plan (usually in planning)",
    "approve_plan": "approve the plan / agree that it is good (usually in plan_approval)",
    "request_plan_changes": "ask to change, fix or rework the plan while it still awaits approval (plan_approval -> planning)",
    "revise_plan": "go BACK to planning to revise the plan during implementation, because the current plan is wrong or must change (execution -> planning rollback)",
    "start_execution": "start or continue the implementation of an approved plan",
    "request_validation": "ask to run the check / validation",
    "validation_passed": "state that the validation succeeded / the result is good",
    "validation_failed": "state that the validation failed / the result needs fixing",
    "complete_task": "finish the task, wrap up, we are done",
    "pause": "pause, stop for now, take a break",
    "resume": "resume, continue after a pause",
    "normal_message": "a question, discussion, ambiguous, uncertain, or none of the above",
}

# Small normalization map for common ways the model might shorten an intent.
_ALIASES: dict[str, str] = {
    "approve": "approve_plan",
    "approve plan": "approve_plan",
    "plan_approval": "approve_plan",
    "request_changes": "request_plan_changes",
    "change_plan": "request_plan_changes",
    "plan_changes": "request_plan_changes",
    "revise": "revise_plan",
    "revise plan": "revise_plan",
    "reviseplan": "revise_plan",
    "rework_plan": "revise_plan",
    "redesign_plan": "revise_plan",
    "back_to_planning": "revise_plan",
    "back_to_plan": "revise_plan",
    "rollback": "revise_plan",
    "rollback_to_planning": "revise_plan",
    "return_to_planning": "revise_plan",
    "replan": "revise_plan",
    "start": "start_execution",
    "execute": "start_execution",
    "start_implementation": "start_execution",
    "run_validation": "request_validation",
    "validate": "request_validation",
    "validation_pass": "validation_passed",
    "pass": "validation_passed",
    "validation_fail": "validation_failed",
    "fail": "validation_failed",
    "complete": "complete_task",
    "finish": "complete_task",
    "done": "complete_task",
    "stop": "pause",
    "continue": "resume",
    "normal": "normal_message",
    "unknown": "normal_message",
    "none": "normal_message",
}

INTENT_SYSTEM_PROMPT = (
    "You are the INTENT RECOGNISER of a task assistant with a strictly "
    "controlled lifecycle. You do NOT change the task state. You only read "
    "ONE user message and classify the user's intent.\n\n"
    "The lifecycle states are: planning, plan_approval, execution, validation, "
    "done.\n"
    "Allowed transitions (enforced by code, NOT by you):\n"
    "  planning -> plan_approval (needs a plan)\n"
    "  plan_approval -> execution (needs plan approved)\n"
    "  plan_approval -> planning (request changes)\n"
    "  execution -> planning (rollback: revise the plan, resets approval)\n"
    "  execution -> validation\n"
    "  validation -> done (needs validation passed)\n"
    "  validation -> execution (validation failed)\n"
    "Pause / resume are control actions available in any unfinished state.\n\n"
    "Choose exactly ONE intent:\n"
    + "\n".join(
        f"- {name}: {description}"
        for name, description in INTENT_DESCRIPTIONS.items()
    )
    + "\n\n"
    "Interpret the message IN CONTEXT (current state, current step, expected "
    "action). The SAME phrase can mean different things in different states: "
    "'ok, go on' in plan_approval likely means approve_plan, while after a "
    "pause it likely means resume.\n"
    "Decide only WHAT THE USER WANTS, not whether it is legal. Do not try to "
    "enforce the lifecycle yourself.\n"
    "If the message is a question, a discussion, ambiguous, or you are not "
    "sure, use normal_message with low confidence.\n\n"
    "Respond with STRICT JSON only, no prose and no markdown:\n"
    '{"intent": "<intent>", "confidence": <number between 0 and 1>}'
)


def build_intent_context(state: Day15TaskState) -> str:
    """The formal task context the model needs to disambiguate the message.

    Besides the lifecycle metadata it includes the RECENT CONVERSATION, because
    short replies ("ок", "продолжай") only make sense in the context of the
    previous turn.
    """
    total = len(state.plan) if state.plan else 0
    step_line = f"{state.current_step} / {total}" if total else str(state.current_step)
    if state.validation_passed is True:
        validation = "true"
    elif state.validation_passed is False:
        validation = "false"
    else:
        validation = "(not run)"
    allowed = [
        name
        for name in ("planning", "plan_approval", "execution", "validation", "done")
    ]
    lines = [
        "TASK CONTEXT",
        "============",
        "GOAL:",
        state.goal or "(not set)",
        "",
        f"STATE: {state.state}",
        f"CURRENT STEP: {step_line}",
        f"EXPECTED ACTION: {state.expected_action or '(not set)'}",
        f"PLAN APPROVED: {'true' if state.plan_approved else 'false'}",
        f"VALIDATION PASSED: {validation}",
        f"PAUSED: {'true' if state.paused else 'false'}",
        f"PREVIOUS STATE (when paused): {state.previous_state or '(none)'}",
    ]
    if state.plan:
        lines += [
            "",
            "PLAN:",
            "\n".join(f"{i}. {s}" for i, s in enumerate(state.plan, 1)),
        ]
    history = [
        message
        for message in state.chat_messages
        if message.role in {"user", "assistant"} and message.kind != "debug"
    ][-6:]
    if history:
        lines += [
            "",
            "RECENT CONVERSATION:",
            "\n".join(
                f"{message.role.upper()}: {message.content[:400]}"
                for message in history
            ),
        ]
    lines += ["", f"CURRENT STATE IS ONE OF: {', '.join(allowed)}"]
    return "\n".join(lines)


def build_intent_request(
    state: Day15TaskState, message: str
) -> tuple[list[dict[str, str]], str]:
    """Return ``(model_messages, context_block)`` for one classification call."""
    context = build_intent_context(state)
    user_content = (
        f"{context}\n\n"
        f"USER MESSAGE:\n{message}\n\n"
        "Return the JSON intent now."
    )
    messages = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, context


def parse_intent(raw: str) -> IntentResult:
    """Parse the model output into a safe :class:`IntentResult`.

    A recognition FAILURE (empty output, invalid JSON, unknown intent) is
    returned as ``normal_message`` **with ``error`` set**, so the caller can
    tell it apart from the user genuinely saying ``normal_message``. It is
    never silently treated as a confident decision.
    """
    text = (raw or "").strip()
    if not text:
        return IntentResult(
            intent="normal_message", confidence=0.0, raw=text,
            error="intent recognition returned empty output",
        )
    try:
        data = parse_json_object(text)
    except ValueError as exc:
        return IntentResult(
            intent="normal_message", confidence=0.0, raw=text,
            error=f"intent output is not valid JSON: {exc}",
        )

    intent = str(data.get("intent", "")).strip().lower()
    intent = _ALIASES.get(intent, intent)
    if intent not in VALID_INTENTS:
        return IntentResult(
            intent="normal_message", confidence=0.0, raw=text,
            error=f"intent output has an unknown intent: {intent!r}",
        )

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return IntentResult(intent=intent, confidence=confidence, raw=text)  # type: ignore[arg-type]


async def recognize_intent(
    deepseek,
    state: Day15TaskState,
    message: str,
    model: str | None,
    *,
    attempts: int = 2,
) -> tuple[IntentResult, str]:
    """Call the LLM and return ``(intent_result, error)``.

    This is the ONLY place the semantic intent is produced. It never hides a
    provider/parse failure: every failed attempt is logged with its real cause
    and, if all attempts fail, the returned error string explains why. The
    caller decides how to present that to the user — but it is no longer
    indistinguishable from a legitimate ``normal_message``.
    """
    messages, _ = build_intent_request(state, message)
    last_error = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            content, finish_reason, _ = await deepseek.generate(
                messages, model=model, max_tokens=300, thinking=False
            )
        except Exception as exc:  # noqa: BLE001 - report, never hide
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "day15 intent recognition attempt %s/%s failed to call the "
                "provider: %s",
                attempt,
                attempts,
                last_error,
            )
            continue
        result = parse_intent(content)
        if not result.error:
            if attempt > 1:
                logger.info(
                    "day15 intent recognition succeeded on attempt %s", attempt
                )
            return result, ""
        last_error = result.error
        logger.warning(
            "day15 intent recognition attempt %s/%s produced unusable output "
            "(finish_reason=%s): %s",
            attempt,
            attempts,
            finish_reason,
            last_error,
        )
    return (
        IntentResult(
            intent="normal_message", confidence=0.0, raw="", error=last_error
        ),
        last_error,
    )


__all__ = [
    "INTENT_DESCRIPTIONS",
    "INTENT_SYSTEM_PROMPT",
    "VALID_INTENTS",
    "build_intent_context",
    "build_intent_request",
    "parse_intent",
    "recognize_intent",
]
