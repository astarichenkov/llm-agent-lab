"""Day 10 — pluggable context strategies.

Every strategy answers one question: **what does the model receive?** They all
produce the same :class:`BuiltContext` shape, so the service (and the UI) never
needs strategy-specific branching logic beyond the extraction hook.

* :class:`SlidingWindowStrategy` — last N messages only;
* :class:`StickyFactsStrategy`   — structured facts + last N messages;
* :class:`BranchingStrategy`     — history of the ACTIVE branch only.

Day 10 deliberately does not summarise history (that is Day 9).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.schemas.day10 import BranchInfo, Facts
from app.services.day10.facts import build_facts_prompt, parse_facts_json
from app.services.day10.models import Branch, Conversation

logger = logging.getLogger("app.services.day10.strategies")

# Small, bounded budget for the facts extraction JSON.
FACTS_EXTRACTION_MAX_TOKENS = 800


@dataclass
class BuiltContext:
    """The context a strategy produced for ONE turn.

    ``messages`` is the COMPLETE payload (system + facts + history + the new
    user message) ready to hand to the provider.
    """

    messages: list[dict[str, str]]
    system_prompt: str | None
    history_messages: list[dict[str, str]]
    facts_message: dict[str, str] | None = None
    dropped_messages: list[dict[str, str]] = field(default_factory=list)
    total_history_messages: int = 0
    sent_history_messages: int = 0
    window_size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Outcome of an optional pre-build step (facts extraction)."""

    performed: bool = False
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ContextStrategy:
    """Base interface every context strategy implements."""

    name: str = "base"

    async def before_build(
        self,
        *,
        user_message: str,
        provider: Any,
        model: str,
        enabled: bool = True,
    ) -> ExtractionResult | None:
        """Optional LLM step that must run BEFORE the context is built.

        Default: nothing to prepare. Sticky Facts overrides this to update its
        memory. Returning ``None`` means "no extra call happened".
        """
        return None

    def build_context(
        self, *, system_prompt: str | None, user_message: str
    ) -> BuiltContext:  # pragma: no cover - abstract
        raise NotImplementedError

    def record_turn(self, user_message: str, assistant_message: str) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        return {}


# ----------------------------------------------------------------------
# 1) Sliding Window
# ----------------------------------------------------------------------
class SlidingWindowStrategy(ContextStrategy):
    """Keep only the last ``window_size`` history messages.

    Semantics of N: it is the number of ``user``/``assistant`` history items,
    NOT the number of pairs. The system prompt is never counted.
    """

    name = "sliding_window"

    def __init__(self, window_size: int = 6) -> None:
        self.window_size = window_size
        self.full_history: list[dict[str, str]] = []

    def _split(self) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        size = max(self.window_size, 0)
        if size == 0:
            return [], list(self.full_history)
        return self.full_history[-size:], self.full_history[:-size]

    def build_context(
        self, *, system_prompt: str | None, user_message: str
    ) -> BuiltContext:
        sent, dropped = self._split()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(sent)
        messages.append({"role": "user", "content": user_message})
        return BuiltContext(
            messages=messages,
            system_prompt=system_prompt,
            history_messages=list(sent),
            dropped_messages=list(dropped),
            total_history_messages=len(self.full_history),
            sent_history_messages=len(sent),
            window_size=self.window_size,
        )

    def record_turn(self, user_message: str, assistant_message: str) -> None:
        self.full_history.append({"role": "user", "content": user_message})
        self.full_history.append({"role": "assistant", "content": assistant_message})

    def reset(self) -> None:
        self.full_history = []

    def snapshot(self) -> dict[str, Any]:
        sent, dropped = self._split()
        return {
            "window_size": self.window_size,
            "full_history": list(self.full_history),
            "sent_history": sent,
            "dropped_history": dropped,
        }


# ----------------------------------------------------------------------
# 2) Sticky Facts / Key-Value Memory
# ----------------------------------------------------------------------
class StickyFactsStrategy(ContextStrategy):
    """Inject a structured FACTS block + a short recent window.

    Payload order: system prompt, FACTS block, last ``recent_messages_limit``
    history messages, then the new user message.
    """

    name = "sticky_facts"

    def __init__(self, recent_messages_limit: int = 6) -> None:
        self.recent_messages_limit = recent_messages_limit
        self.facts = Facts()
        self.full_history: list[dict[str, str]] = []
        self.last_extraction_error: str | None = None

    # ---- extraction ---------------------------------------------------
    async def before_build(
        self,
        *,
        user_message: str,
        provider: Any,
        model: str,
        enabled: bool = True,
    ) -> ExtractionResult | None:
        if not enabled:
            return ExtractionResult(performed=False)
        payload = build_facts_prompt(self.facts, user_message)
        try:
            content, _finish, usage = await provider.generate(
                payload,
                model=model,
                max_tokens=FACTS_EXTRACTION_MAX_TOKENS,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001 - graceful degradation is required
            # A failed extraction must NEVER destroy the previous memory.
            logger.warning(
                "day10 facts extraction failed (%s): %s", type(exc).__name__, exc
            )
            self.last_extraction_error = (
                "Не удалось обновить facts (ошибка обращения к модели). "
                "Сохранены предыдущие facts."
            )
            return ExtractionResult(performed=False, error=self.last_extraction_error)

        try:
            updated = parse_facts_json(content or "")
        except ValueError as exc:
            logger.warning("day10 facts JSON parse failed: %s", exc)
            self.last_extraction_error = (
                "Не удалось обновить facts (некорректный JSON от модели). "
                "Сохранены предыдущие facts."
            )
            return ExtractionResult(performed=False, error=self.last_extraction_error)

        # Commit ONLY after a successful parse — previous memory otherwise stays.
        self.facts = updated
        self.last_extraction_error = None
        usage = usage or {}
        return ExtractionResult(
            performed=True,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    # ---- context ------------------------------------------------------
    def _recent(self) -> list[dict[str, str]]:
        size = max(self.recent_messages_limit, 0)
        if size == 0:
            return []
        return self.full_history[-size:]

    def build_context(
        self, *, system_prompt: str | None, user_message: str
    ) -> BuiltContext:
        recent = self._recent()
        dropped = self.full_history[: len(self.full_history) - len(recent)]
        facts_message: dict[str, str] | None = None
        if not self.facts.is_empty():
            facts_message = {"role": "system", "content": self.facts.to_block()}

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if facts_message:
            messages.append(facts_message)
        messages.extend(recent)
        messages.append({"role": "user", "content": user_message})

        return BuiltContext(
            messages=messages,
            system_prompt=system_prompt,
            history_messages=list(recent),
            facts_message=facts_message,
            dropped_messages=list(dropped),
            total_history_messages=len(self.full_history),
            sent_history_messages=len(recent),
            window_size=self.recent_messages_limit,
            extra={
                "facts": self.facts.model_dump(),
                "facts_block": self.facts.to_block(),
                "facts_count": self.facts.count(),
            },
        )

    def record_turn(self, user_message: str, assistant_message: str) -> None:
        self.full_history.append({"role": "user", "content": user_message})
        self.full_history.append({"role": "assistant", "content": assistant_message})

    def reset(self) -> None:
        self.full_history = []
        self.facts = Facts()
        self.last_extraction_error = None

    def snapshot(self) -> dict[str, Any]:
        recent = self._recent()
        return {
            "recent_messages_limit": self.recent_messages_limit,
            "full_history": list(self.full_history),
            "recent_history": recent,
            "facts": self.facts.model_dump(),
            "facts_block": self.facts.to_block(),
            "facts_count": self.facts.count(),
            "facts_error": self.last_extraction_error,
        }


# ----------------------------------------------------------------------
# 3) Branching
# ----------------------------------------------------------------------
def branch_info(conversation: Conversation, branch: Branch) -> BranchInfo:
    return BranchInfo(
        id=branch.id,
        name=branch.name,
        parent_branch_id=branch.parent_branch_id,
        checkpoint_message_id=branch.checkpoint_message_id,
        message_count=len(conversation.branch_history(branch.id)),
        is_active=branch.id == conversation.active_branch_id,
    )


class BranchingStrategy(ContextStrategy):
    """Send only the ACTIVE branch's history (which shares the ancestor prefix)."""

    name = "branching"

    def __init__(self) -> None:
        self.conversation = Conversation.create_root()

    def build_context(
        self, *, system_prompt: str | None, user_message: str
    ) -> BuiltContext:
        branch = self.conversation.active_branch
        history = self.conversation.branch_history_payload(branch.id)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return BuiltContext(
            messages=messages,
            system_prompt=system_prompt,
            history_messages=history,
            dropped_messages=[],
            total_history_messages=len(history),
            sent_history_messages=len(history),
            window_size=None,
            extra={
                "branches": [
                    branch_info(self.conversation, b).model_dump()
                    for b in self.conversation.branches.values()
                ],
                "active_branch_id": branch.id,
                "active_branch_name": branch.name,
                "checkpoint_message_id": branch.checkpoint_message_id,
                "checkpoint_index": self.conversation.checkpoint_index(branch.id),
                "own_messages": len(self.conversation.own_messages(branch.id)),
            },
        )

    def record_turn(self, user_message: str, assistant_message: str) -> None:
        self.conversation.add_message("user", user_message)
        self.conversation.add_message("assistant", assistant_message)

    def create_branch(
        self,
        *,
        name: str | None = None,
        parent_branch_id: str | None = None,
        checkpoint_message_id: str | None = None,
    ) -> Branch:
        return self.conversation.create_branch(
            name=name,
            parent_branch_id=parent_branch_id,
            checkpoint_message_id=checkpoint_message_id,
        )

    def activate(self, branch_id: str) -> Branch:
        return self.conversation.activate(branch_id)

    def reset(self) -> None:
        self.conversation.reset()

    def snapshot(self) -> dict[str, Any]:
        return {
            "conversation": self.conversation,
            "branches": [
                branch_info(self.conversation, b).model_dump()
                for b in self.conversation.branches.values()
            ],
            "active_branch_id": self.conversation.active_branch_id,
        }
