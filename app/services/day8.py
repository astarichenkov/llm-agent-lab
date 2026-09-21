"""Day 8 — token accounting service.

This service owns an ISOLATED, in-memory dialog for the Day 8 tab. It never
touches the Day 7 SQLite history, and it never trims the context: the whole
point of the overflow experiment is to show what actually happens when the
prompt exceeds the model context window.

Responsibilities
----------------
* run one chat turn through the existing :class:`DeepSeekService` (no second
  provider client is created);
* combine EXACT provider ``usage`` with LOCAL per-part estimates;
* estimate cost from the documented DeepSeek tariffs;
* report the model limits so the UI can show "estimated input vs limit".

Nothing here invents a successful answer: provider errors propagate to the
route, which maps them to real HTTP statuses.
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.schemas.day8 import (
    Day8ChatRequest,
    Day8ChatResponse,
    Day8Cost,
    Day8EstimateResponse,
    Day8Limits,
    Day8OverflowResponse,
    Day8Usage,
)
from app.services.deepseek import DeepSeekService
from app.services import model_params
from app.services.token_estimate import estimate_context_breakdown, estimate_message_tokens

logger = logging.getLogger("app.services.day8")

# Deterministic filler used to grow the synthetic context. One fixed sentence,
# repeated — no randomness, no external data, easy to reason about.
_FILLER_UNIT = "context filler data "
# ~240k ASCII chars per synthetic turn.
_FILLER_CHARS_PER_TURN = 240_000


class Day8TokenService:
    """Runs Day 8 turns and assembles the token/cost/limit report."""

    def __init__(self, settings: Settings, deepseek: DeepSeekService | None = None) -> None:
        self._settings = settings
        # Reuse the existing provider client; Day 8 does not implement its own.
        # Injectable so tests can supply a deterministic fake (no network).
        self._deepseek = deepseek if deepseek is not None else DeepSeekService(settings)
        # Isolated in-memory dialog (NOT the persistent Day 7 agent).
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _resolved_model(self, model: str | None) -> str:
        return model or self._settings.deepseek_model

    def _system_prompt(self, system_prompt: str | None) -> str:
        return system_prompt if system_prompt is not None else self._settings.system_prompt

    def _build_usage(
        self,
        *,
        system_prompt: str | None,
        history: list[dict[str, str]],
        current_user_message: str,
        provider_usage: dict | None,
    ) -> Day8Usage:
        parts = estimate_context_breakdown(
            system_prompt=system_prompt,
            history=history,
            current_user_message=current_user_message,
        )
        provider_usage = provider_usage or {}
        return Day8Usage(
            prompt_tokens=provider_usage.get("prompt_tokens"),
            completion_tokens=provider_usage.get("completion_tokens"),
            total_tokens=provider_usage.get("total_tokens"),
            prompt_cache_hit_tokens=provider_usage.get("prompt_cache_hit_tokens"),
            prompt_cache_miss_tokens=provider_usage.get("prompt_cache_miss_tokens"),
            current_user_tokens_estimated=parts["current_user_tokens"],
            history_tokens_estimated=parts["history_tokens"],
            system_prompt_tokens_estimated=parts["system_prompt_tokens"],
            estimated_input_tokens=parts["estimated_input_tokens"],
        )

    def _limits(self, model: str) -> Day8Limits:
        limits = model_params.get_limits(model)
        return Day8Limits(
            context_window=limits.context_window,
            max_output_tokens=limits.max_output_tokens,
            model_known=model_params.is_known_model(model),
        )

    def _cost(self, model: str, usage: Day8Usage) -> Day8Cost:
        raw = model_params.estimate_cost(
            model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cache_hit_tokens=usage.prompt_cache_hit_tokens,
            cache_miss_tokens=usage.prompt_cache_miss_tokens,
        )
        return Day8Cost(
            input=float(raw["input"]) if raw["input"] is not None else None,
            output=float(raw["output"]) if raw["output"] is not None else None,
            total=float(raw["total"]) if raw["total"] is not None else None,
            currency=raw["currency"],
            estimated=raw["estimated"],
            reason=raw.get("reason"),
        )

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    def estimate(
        self,
        *,
        message: str,
        history: list[dict[str, str]] | None,
        model: str | None,
        system_prompt: str | None,
    ) -> Day8EstimateResponse:
        """Local-only estimation: no provider call, no API spend."""
        resolved_model = self._resolved_model(model)
        system = self._system_prompt(system_prompt)
        effective_history = history if history is not None else list(self._history)
        usage = self._build_usage(
            system_prompt=system,
            history=effective_history,
            current_user_message=message,
            provider_usage=None,
        )
        limits = self._limits(resolved_model)
        # The provider counts the OUTPUT budget toward the context window too
        # (its error message states: "you requested N tokens (X in the
        # messages, Y in the completion)"). For the overflow experiment we
        # therefore compare messages + completion budget against the window.
        effective_input = usage.estimated_input_tokens + limits.max_output_tokens
        overflow = max(effective_input - limits.context_window, 0)
        return Day8EstimateResponse(
            model=resolved_model,
            usage=usage,
            limits=limits,
            effective_input_tokens=effective_input,
            exceeds_context=overflow > 0,
            overflow_tokens=overflow,
        )

    @staticmethod
    def _build_filler() -> str:
        """One deterministic filler block of ~``_FILLER_CHARS_PER_TURN`` chars."""
        repeats = max(_FILLER_CHARS_PER_TURN // len(_FILLER_UNIT), 1)
        return _FILLER_UNIT * repeats

    def build_overflow(
        self,
        *,
        model: str | None,
        system_prompt: str | None,
        overshoot_factor: float,
        message: str = "Продолжи.",
    ) -> Day8OverflowResponse:
        """Build a deterministic synthetic context that exceeds the window.

        Generated SERVER-SIDE so the browser never uploads a multi-megabyte
        payload. The returned ``request`` is the exact chat payload the UI
        should send to execute the real request, guaranteeing that the
        estimated context and the executed one are identical.
        """
        resolved_model = self._resolved_model(model)
        system = self._system_prompt(system_prompt)
        limits = self._limits(resolved_model)

        filler = self._build_filler()
        # Target MESSAGE tokens so that messages + output budget exceed the
        # window by ``overshoot_factor``.
        target_message_tokens = int(
            limits.context_window * overshoot_factor
        ) - limits.max_output_tokens
        tokens_per_turn = estimate_message_tokens(filler)
        turns = max(int(target_message_tokens / max(tokens_per_turn, 1)) + 1, 1)

        history: list[dict[str, str]] = []
        for turn in range(turns):
            history.append(
                {"role": "user", "content": f"Part {turn + 1}. {filler}"}
            )
            history.append(
                {"role": "assistant", "content": f"Acknowledged {turn + 1}. {filler}"}
            )

        usage = self._build_usage(
            system_prompt=system,
            history=history,
            current_user_message=message,
            provider_usage=None,
        )
        effective_input = usage.estimated_input_tokens + limits.max_output_tokens
        overflow = max(effective_input - limits.context_window, 0)
        return Day8OverflowResponse(
            model=resolved_model,
            usage=usage,
            limits=limits,
            effective_input_tokens=effective_input,
            exceeds_context=overflow > 0,
            overflow_tokens=overflow,
            turns=turns,
            message_tokens=usage.estimated_input_tokens,
            request=Day8ChatRequest(
                message=message, history=history, model=resolved_model
            ),
        )

    async def chat(
        self,
        *,
        message: str,
        history: list[dict[str, str]] | None,
        model: str | None,
        system_prompt: str | None,
    ) -> Day8ChatResponse:
        """Run one real turn through the existing DeepSeek service.

        When ``history`` is provided it is used as the prior context for this
        call only (synthetic experiment context). The internal in-memory
        history is updated afterwards so a normal dialog keeps its context.
        """
        resolved_model = self._resolved_model(model)
        system = self._system_prompt(system_prompt)
        effective_history = history if history is not None else list(self._history)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(effective_history)
        messages.append({"role": "user", "content": message})

        # max_tokens caps the OUTPUT only; it is unrelated to the context
        # window. We use the model's documented maximum output budget here so
        # the overflow experiment is driven purely by INPUT size.
        limits = self._limits(resolved_model)
        content, finish_reason, provider_usage = await self._deepseek.generate(
            messages,
            model=resolved_model,
            max_tokens=limits.max_output_tokens,
            thinking=False,
        )

        usage = self._build_usage(
            system_prompt=system,
            history=effective_history,
            current_user_message=message,
            provider_usage=provider_usage,
        )

        # Commit to the isolated in-memory dialog (only for the user-supplied
        # "normal" chat path, i.e. when no synthetic history was passed).
        if history is None:
            self._history.append({"role": "user", "content": message})
            self._history.append({"role": "assistant", "content": content})

        logger.info(
            "day8 turn completed model=%s messages_sent=%s finish_reason=%s "
            "prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            resolved_model,
            len(messages),
            finish_reason,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )

        return Day8ChatResponse(
            answer=content,
            model=resolved_model,
            provider="deepseek",
            finish_reason=finish_reason,
            usage=usage,
            cost=self._cost(resolved_model, usage),
            limits=limits,
            message_count=len(self._history),
        )

    def reset(self) -> None:
        """Clear the isolated Day 8 dialog (does not touch Day 7)."""
        self._history = []

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)
