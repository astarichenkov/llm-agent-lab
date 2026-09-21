"""Day 15 — the WORK layer of the conversational task agent.

The state machine decides WHICH stage the task is in; this module is
responsible for actually DOING the work of a stage with the LLM and turning it
into human-readable chat content:

    PLANNING      -> a real plan (list of steps)
    EXECUTION     -> a concrete result / deliverable
    VALIDATION    -> a pass/fail verdict about the result
    NORMAL MESSAGE-> a conversational answer in the context of the task

Nothing here can change ``state``. The service feeds the output back through
:class:`~app.services.day15.lifecycle.LifecycleMachine`, exactly like any other
transition. Keeping the prompts and the parsing in one place means the
"LLM writes content / code decides state" boundary stays obvious.

The planning prompt and the plan parser are reused VERBATIM from Day 13 so the
two demos behave consistently; Day 13 itself is not modified.
"""
from __future__ import annotations

from app.schemas.day15 import Day15TaskState
from app.services.day13.service import (  # reused, not duplicated
    PLANNING_SYSTEM_PROMPT,
    build_planning_request,
    parse_plan,
    parse_verdict,
)

# Safety cap: the chat never gets an unbounded wall of text.
_MAX_CHAT_CONTEXT = 8

EXECUTION_SYSTEM_PROMPT = (
    "You are an autonomous assistant WORKING on a task. You receive the formal "
    "TASK CONTEXT (goal, plan, current step, expected action, previous result) "
    "and one user message. Produce the CONCRETE result of the expected action "
    "for the current step: the actual deliverable, not a plan and not a promise.\n"
    "Rules:\n"
    "- do the work instead of describing how you would do it;\n"
    "- stay on the current plan step; do not redo completed steps;\n"
    "- answer in the SAME LANGUAGE as the task goal;\n"
    "- be concise and concrete (short sections, code or a list when useful)."
)

VALIDATION_SYSTEM_PROMPT = (
    "You are a strict TASK VALIDATOR. Check whether the produced RESULT really "
    "satisfies the task GOAL and every PLAN step. Reply with STRICT JSON:\n"
    '{"verdict": "pass" | "fail", "summary": "one short sentence"}\n\n'
    "Use \"pass\" only when the result genuinely satisfies the goal and all "
    "steps. Be conservative: if something required is missing, use \"fail\".\n"
    "Answer in the same language as the task goal, but keep the JSON keys in "
    "English. Output ONLY the JSON object."
)

ANSWER_SYSTEM_PROMPT = (
    "You are the task assistant. A task is already in progress with a fixed "
    "lifecycle. You receive the formal TASK CONTEXT and the recent conversation."
    " Answer the user's message about the task in a natural, helpful way.\n"
    "Rules:\n"
    "- answer the question that was actually asked;\n"
    "- stay consistent with the current state and the current plan;\n"
    "- do NOT claim that the state changed and do NOT invent progress;\n"
    "- answer in the SAME LANGUAGE as the user's message;\n"
    "- be concise."
)


def build_task_context(state: Day15TaskState, *, include_history: bool = True) -> str:
    """The formal, human-readable TASK CONTEXT block used by every prompt."""
    total = len(state.plan) if state.plan else 0
    step_line = f"{state.current_step} / {total}" if total else str(state.current_step)
    if state.validation_passed is True:
        validation = "true"
    elif state.validation_passed is False:
        validation = "false"
    else:
        validation = "(not run)"

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
    ]
    if state.plan:
        lines += [
            "",
            "PLAN:",
            "\n".join(f"{i}. {step}" for i, step in enumerate(state.plan, 1)),
        ]
    if state.completed_steps:
        lines += [
            "",
            "COMPLETED STEPS:",
            "\n".join(f"- {step}" for step in state.completed_steps),
        ]
    if state.execution_result:
        lines += ["", "CURRENT RESULT:", state.execution_result]
    if include_history and state.chat_messages:
        history = [
            message
            for message in state.chat_messages
            if message.role in {"user", "assistant"} and message.kind != "debug"
        ][-_MAX_CHAT_CONTEXT:]
        if history:
            lines += [
                "",
                "RECENT CONVERSATION:",
                "\n".join(
                    f"{message.role.upper()}: {message.content[:500]}"
                    for message in history
                ),
            ]
    return "\n".join(lines)


def build_execution_request(
    state: Day15TaskState, message: str
) -> tuple[list[dict[str, str]], str]:
    """Messages for producing the result of the current plan step."""
    context = build_task_context(state)
    user_content = (
        f"{context}\n\n"
        f"USER MESSAGE:\n{message or '(no extra instruction)'}\n\n"
        "Produce the concrete result of the EXPECTED ACTION now."
    )
    messages = [
        {"role": "system", "content": EXECUTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, context


def build_validation_request(
    state: Day15TaskState, message: str
) -> tuple[list[dict[str, str]], str]:
    """Messages for validating the produced result against the goal/plan."""
    context = build_task_context(state)
    user_content = (
        f"{context}\n\n"
        f"USER MESSAGE:\n{message or '(validate the result)'}\n\n"
        "Return the JSON verdict now."
    )
    messages = [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, context


def build_answer_request(
    state: Day15TaskState, message: str
) -> tuple[list[dict[str, str]], str]:
    """Messages for a normal conversational answer about the task."""
    context = build_task_context(state)
    user_content = (
        f"{context}\n\n"
        f"USER MESSAGE:\n{message}\n\n"
        "Answer the user now."
    )
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, context


__all__ = [
    "ANSWER_SYSTEM_PROMPT",
    "EXECUTION_SYSTEM_PROMPT",
    "PLANNING_SYSTEM_PROMPT",
    "VALIDATION_SYSTEM_PROMPT",
    "build_answer_request",
    "build_execution_request",
    "build_planning_request",
    "build_task_context",
    "build_validation_request",
    "parse_plan",
    "parse_verdict",
]
