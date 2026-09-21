"""Day 14 — invariants service.

The service owns two LOGICALLY SEPARATE pieces of state:

    Invariants    — mandatory constraints, persisted in their own JSON file;
    Conversation  — ordinary user/assistant messages, kept in memory.

They are never mixed: invariants are not appended to ``messages`` and the
conversation is not part of the invariants file. On every request the service:

    1. runs a DETERMINISTIC conflict check (code, not the LLM);
    2. if a conflict is found, refuses in code and names the invariant(s);
    3. otherwise builds the model request with an explicit
       ``ACTIVE INVARIANTS`` system block and forwards it to DeepSeek.

Day 13's Task State Machine is left completely untouched: Day 14 is an
additional, independent constraint layer.
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.schemas.day14 import (
    Day14ChatResponse,
    Day14Message,
    Day14StateResponse,
    Invariant,
    InvariantConflict,
)
from app.services.day14.invariants import (
    DEFAULT_INVARIANTS,
    build_invariants_block,
    build_refusal,
    detect_conflicts,
)
from app.services.day14.store import InvariantStore
from app.services.deepseek import DeepSeekService

logger = logging.getLogger("app.services.day14")

# System rules sent BEFORE the invariants block (which repeats the mandatory
# part as a separate, clearly delimited context block).
DAY14_SYSTEM_RULES = (
    "You are a helpful software engineering assistant. You work under a set "
    "of mandatory invariants supplied in a separate ACTIVE INVARIANTS block. "
    "Respect them at all times and never propose a solution that violates "
    "them. Answer in the same language as the user's request. Be concrete."
)


class Day14InvariantService:
    """Owns the active invariants and the (separate) conversation history."""

    def __init__(
        self,
        settings: Settings,
        deepseek: DeepSeekService | None = None,
        store: InvariantStore | None = None,
        invariants: list[Invariant] | None = None,
    ) -> None:
        self._settings = settings
        self._deepseek = deepseek if deepseek is not None else DeepSeekService(settings)
        self._store = store or InvariantStore(settings.day14_invariants_path)
        if invariants is not None:
            self._invariants = [inv.model_copy(deep=True) for inv in invariants]
        else:
            self._invariants = self._store.load()
        # Conversation state is deliberately NOT persisted with the invariants.
        self._messages: list[Day14Message] = []
        logger.info(
            "day14 invariant service initialised invariants=%s path=%s",
            len(self._invariants),
            self._store.path,
        )

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------
    @property
    def model(self) -> str:
        return self._settings.deepseek_model

    @property
    def invariants(self) -> list[Invariant]:
        return list(self._invariants)

    @property
    def messages(self) -> list[Day14Message]:
        return list(self._messages)

    def state_response(self) -> Day14StateResponse:
        return Day14StateResponse(
            model=self.model,
            invariants=self.invariants,
            messages=self.messages,
            count=len(self._messages),
        )

    def reset_conversation(self) -> None:
        """Clear ONLY the conversation; invariants are untouched."""
        self._messages = []
        logger.info("day14 conversation cleared")

    # ------------------------------------------------------------------
    # context building (pure, testable)
    # ------------------------------------------------------------------
    def build_context(
        self,
        message: str,
        history: list[Day14Message] | None = None,
    ) -> tuple[list[dict[str, str]], str]:
        """Return ``(model_messages, invariants_block)``.

        The returned messages always contain the invariants block as a
        separate ``system`` message, so the model receives the constraints as
        an explicit execution context (not merely as UI text).
        """
        block = build_invariants_block(self._invariants)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": DAY14_SYSTEM_RULES},
            {"role": "system", "content": block},
        ]
        for item in history if history is not None else self._messages:
            messages.append({"role": item.role, "content": item.content})
        messages.append({"role": "user", "content": message})
        return messages, block

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    async def chat(
        self,
        *,
        message: str,
        model: str | None = None,
    ) -> Day14ChatResponse:
        """Validate one request against the invariants and answer if allowed."""
        resolved_model = model or self.model
        block = build_invariants_block(self._invariants)

        conflicts: list[InvariantConflict] = detect_conflicts(
            message, self._invariants
        )

        if conflicts:
            # Deterministic refusal: the model is not asked to agree with a
            # request that violates an invariant.
            self._messages.append(Day14Message(role="user", content=message))
            answer = build_refusal(conflicts)
            self._messages.append(
                Day14Message(role="assistant", content=answer)
            )
            logger.info(
                "day14 request refused conflicts=%s",
                ",".join(c.invariant_id for c in conflicts),
            )
            return Day14ChatResponse(
                answer=answer,
                model=resolved_model,
                allowed=False,
                conflicts=conflicts,
                invariants=self.invariants,
                context_block=block,
            )

        # Build the context from PRIOR history; ``message`` is appended once as
        # the final user turn by ``build_context``.
        messages, _ = self.build_context(message, history=self._messages)
        content, finish_reason, usage = await self._deepseek.generate(
            messages, model=resolved_model, max_tokens=1024, thinking=False
        )
        self._messages.append(Day14Message(role="user", content=message))
        self._messages.append(Day14Message(role="assistant", content=content))
        logger.info(
            "day14 request allowed model=%s invariants=%s",
            resolved_model,
            len(self._invariants),
        )
        return Day14ChatResponse(
            answer=content,
            model=resolved_model,
            finish_reason=finish_reason,
            allowed=True,
            invariants=self.invariants,
            context_block=block,
            usage=usage,
        )


__all__ = ["DAY14_SYSTEM_RULES", "Day14InvariantService", "DEFAULT_INVARIANTS"]
