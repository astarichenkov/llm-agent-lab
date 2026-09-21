"""Day 9 — context compression service.

Runs ONE dialog that can be executed in two modes:

* ``full``       — the whole history is sent every turn (the Day 8 baseline);
* ``compressed`` — a separate summary replaces the old messages, and only the
  last ``recent_messages_limit`` messages are sent verbatim.

The compression POLICY lives in :mod:`app.services.context_compression` (pure
and unit-tested). This service only orchestrates provider calls and token
accounting, reusing the Day 8 estimator and the shared ``DeepSeekService``.

A failed summary build NEVER corrupts the dialog: the previous summary and the
``summarized_count`` are only committed after a successful provider response.
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.schemas.day9 import (
    Day9ChatResponse,
    Day9Cost,
    Day9Diagnostics,
    Day9DiagnosticsResponse,
    Day9SeedResponse,
    Day9Usage,
)
from app.services import model_params
from app.services.context_compression import (
    CompressionConfig,
    CompressionState,
    build_summary_prompt,
)
from app.services.deepseek import DeepSeekError, DeepSeekService
from app.services.token_estimate import (
    estimate_message_tokens,
    estimate_messages_tokens,
)

logger = logging.getLogger("app.services.day9")

# Facts stated at the START of the demo dialog. They must survive compression
# so the final control question can be answered from the summary alone.
_DEMO_OPENING = [
    ("user", "Запомни ключевые решения проекта: название проекта — Orion."),
    ("assistant", "Запомнил: проект называется Orion."),
    ("user", "Backend — FastAPI, база данных — PostgreSQL."),
    ("assistant", "Принято: FastAPI и PostgreSQL."),
    ("user", "Ограничение: Redis использовать запрещено."),
    ("assistant", "Понял, Redis запрещён."),
    ("user", "Требование: API должен оставаться stateless."),
    ("assistant", "Зафиксировано: API stateless."),
    ("user", "Аутентификация — JWT."),
    ("assistant", "Запомнил: аутентификация через JWT."),
]

# Filler discussion so the opening facts leave the recent window.
_DEMO_FILLER = [
    "Обсудим детали развёртывания сервиса.",
    "Хорошо, давай разберём деплой подробнее.",
    "Как лучше организовать миграции схемы базы данных?",
    "Для миграций подойдёт простой versioned-подход.",
    "Стоит ли кэшировать ответы на уровне приложения?",
    "Кэш можно оставить в памяти процесса без внешних сервисов.",
    "Как тестировать stateless-эндпоинты?",
    "Достаточно интеграционных тестов с изоляцией состояния.",
    "Нужны ли фоновые задачи?",
    "На текущем этапе фоновые задачи не требуются.",
    "Как логировать запросы без утечки секретов?",
    "Логируем только метаданные и коды ответов.",
    "Есть ли ещё ограничения по инфраструктуре?",
    "Дополнительных ограничений не добавляем.",
]

DEMO_CONTROL_QUESTION = (
    "Какие ключевые архитектурные решения и ограничения мы выбрали "
    "в начале разговора?"
)


class Day9CompressionService:
    """Owns one compressible dialog and its token accounting."""

    def __init__(
        self, settings: Settings, deepseek: DeepSeekService | None = None
    ) -> None:
        self._settings = settings
        # Reuse the shared provider client; injectable for deterministic tests.
        self._deepseek = deepseek if deepseek is not None else DeepSeekService(settings)
        self._state = CompressionState()
        self._config = CompressionConfig(
            recent_messages_limit=settings.day9_recent_messages_limit,
            compression_batch_size=settings.day9_compression_batch_size,
        )
        self._last_answer: str | None = None

    # ------------------------------------------------------------------
    # config / helpers
    # ------------------------------------------------------------------
    def _effective_config(
        self, recent_messages_limit: int | None, compression_batch_size: int | None
    ) -> CompressionConfig:
        return CompressionConfig(
            recent_messages_limit=(
                recent_messages_limit
                if recent_messages_limit is not None
                else self._config.recent_messages_limit
            ),
            compression_batch_size=(
                compression_batch_size
                if compression_batch_size is not None
                else self._config.compression_batch_size
            ),
        )

    def _resolved_model(self, model: str | None) -> str:
        return model or self._settings.deepseek_model

    def _system_prompt(self, system_prompt: str | None) -> str:
        return (
            system_prompt
            if system_prompt is not None
            else self._settings.system_prompt
        )

    def _limits(self, model: str):
        return model_params.get_limits(model)

    def _cost(self, model: str, usage: Day9Usage) -> Day9Cost:
        raw = model_params.estimate_cost(
            model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return Day9Cost(
            input=float(raw["input"]) if raw["input"] is not None else None,
            output=float(raw["output"]) if raw["output"] is not None else None,
            total=float(raw["total"]) if raw["total"] is not None else None,
            currency=raw["currency"],
            estimated=raw["estimated"],
            reason=raw.get("reason"),
        )

    @staticmethod
    def _savings(full_tokens: int, compressed_tokens: int) -> tuple[int, float]:
        """Tokens saved and savings percent, safe against divide-by-zero."""
        saved = full_tokens - compressed_tokens
        if full_tokens <= 0:
            return saved, 0.0
        return saved, round(saved / full_tokens * 100.0, 2)

    def _diagnostics(self, config: CompressionConfig) -> Day9Diagnostics:
        state = self._state
        full_tokens = estimate_messages_tokens(state.full_history)
        summary_tokens = (
            estimate_message_tokens(state.summary) if state.summary else 0
        )
        recent = state.recent_window(config)
        recent_tokens = estimate_messages_tokens(recent)
        # The compressed context is exactly what we send: summary + recent.
        compressed_tokens = summary_tokens + recent_tokens
        saved, saved_pct = self._savings(full_tokens, compressed_tokens)
        return Day9Diagnostics(
            full_history_messages=len(state.full_history),
            summarized_messages=state.summarized_count,
            recent_messages_sent=len(recent),
            full_history_tokens_estimated=full_tokens,
            summary_tokens_estimated=summary_tokens,
            recent_tokens_estimated=recent_tokens,
            compressed_context_tokens_estimated=compressed_tokens,
            tokens_saved=saved,
            tokens_saved_percent=saved_pct,
            compression_cycles=state.compression_cycles,
            summary=state.summary,
            recent_messages_limit=config.recent_messages_limit,
            compression_batch_size=config.compression_batch_size,
            pending_messages=len(state.pending_messages(config)),
        )

    # ------------------------------------------------------------------
    # compression
    # ------------------------------------------------------------------
    async def _maybe_compress(
        self, config: CompressionConfig, model: str
    ) -> str | None:
        """Rebuild the summary when a full batch accumulated.

        Returns an error message when the summary call failed, else ``None``.
        On failure NOTHING is committed — the old summary and count survive.
        """
        if not self._state.should_compress(config):
            return None

        block = self._state.pending_messages(config)
        next_count = self._state.next_summarized_count(config)
        previous = self._state.summary
        payload = build_summary_prompt(previous, block)
        try:
            summary_text, _finish, usage = await self._deepseek.generate(
                payload,
                model=model,
                max_tokens=model_params.get_limits(model).max_output_tokens,
                thinking=False,
            )
        except DeepSeekError as exc:
            logger.warning("day9 summary build failed: %s", exc.message)
            return exc.message

        cleaned = (summary_text or "").strip()
        if not cleaned:
            logger.warning("day9 summary build returned empty content")
            return "Не удалось построить summary: модель вернула пустой ответ."

        # Commit ONLY after a successful, non-empty response.
        self._state.summary = cleaned
        self._state.summarized_count = next_count
        self._state.compression_cycles += 1
        if usage:
            self._state.last_summary_prompt_tokens = usage.get("prompt_tokens")
            self._state.last_summary_completion_tokens = usage.get("completion_tokens")
        logger.info(
            "day9 compression cycle=%s summarized_count=%s block=%s summary_len=%s",
            self._state.compression_cycles,
            self._state.summarized_count,
            len(block),
            len(cleaned),
        )
        return None

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    async def chat(
        self,
        *,
        message: str,
        mode: str,
        recent_messages_limit: int | None = None,
        compression_batch_size: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Day9ChatResponse:
        """Run one turn in ``full`` or ``compressed`` mode."""
        resolved_model = self._resolved_model(model)
        system = self._system_prompt(system_prompt)
        config = self._effective_config(recent_messages_limit, compression_batch_size)

        compression_error: str | None = None
        compression_performed = False

        if mode == "compressed":
            before_cycles = self._state.compression_cycles
            compression_error = await self._maybe_compress(config, resolved_model)
            compression_performed = self._state.compression_cycles > before_cycles

        # ---- build the ACTUAL payload for this turn ----
        if mode == "compressed":
            context = self._state.build_working_context(config, system)
        else:
            context = []
            if system:
                context.append({"role": "system", "content": system})
            context.extend(self._state.full_history)

        messages = context + [{"role": "user", "content": message}]
        limits = self._limits(resolved_model)
        content, finish_reason, provider_usage = await self._deepseek.generate(
            messages,
            model=resolved_model,
            max_tokens=limits.max_output_tokens,
            thinking=False,
        )
        provider_usage = provider_usage or {}

        # ---- token accounting ----
        full_context_tokens = estimate_messages_tokens(
            ([{"role": "system", "content": system}] if system else [])
            + self._state.full_history
        )
        summary_tokens = (
            estimate_message_tokens(self._state.summary)
            if self._state.summary
            else 0
        )
        recent_tokens = estimate_messages_tokens(self._state.recent_window(config))
        compressed_context_tokens = summary_tokens + recent_tokens
        # In full mode the model really receives the whole history, so the
        # "compressed" counter must reflect that (savings become zero).
        if mode == "full":
            compressed_context_tokens = full_context_tokens
            summary_tokens = 0
            recent_tokens = estimate_messages_tokens(self._state.full_history)

        saved, saved_pct = self._savings(
            full_context_tokens, compressed_context_tokens
        )
        current_tokens = estimate_message_tokens(message)
        system_tokens = estimate_message_tokens(system) if system else 0

        usage = Day9Usage(
            full_context_tokens_estimated=full_context_tokens,
            summary_tokens_estimated=summary_tokens,
            recent_tokens_estimated=recent_tokens,
            compressed_context_tokens_estimated=compressed_context_tokens,
            current_user_tokens_estimated=current_tokens,
            system_prompt_tokens_estimated=system_tokens,
            prompt_tokens=provider_usage.get("prompt_tokens"),
            completion_tokens=provider_usage.get("completion_tokens"),
            total_tokens=provider_usage.get("total_tokens"),
            tokens_saved=saved,
            tokens_saved_percent=saved_pct,
        )

        # ---- commit the turn to the FULL history (never trimmed) ----
        self._state.full_history.append({"role": "user", "content": message})
        self._state.full_history.append({"role": "assistant", "content": content})
        self._last_answer = content

        logger.info(
            "day9 turn completed mode=%s model=%s full_msgs=%s sent_msgs=%s "
            "summarized=%s cycles=%s finish_reason=%s prompt_tokens=%s "
            "completion_tokens=%s",
            mode,
            resolved_model,
            len(self._state.full_history),
            len(messages),
            self._state.summarized_count,
            self._state.compression_cycles,
            finish_reason,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        return Day9ChatResponse(
            answer=content,
            mode=mode,
            model=resolved_model,
            provider="deepseek",
            finish_reason=finish_reason,
            usage=usage,
            cost=self._cost(resolved_model, usage),
            diagnostics=self._diagnostics(config),
            compression_performed=compression_performed,
            compression_error=compression_error,
        )

    # ------------------------------------------------------------------
    # demo seed
    # ------------------------------------------------------------------
    def seed_demo(
        self,
        *,
        recent_messages_limit: int | None = None,
        compression_batch_size: int | None = None,
    ) -> Day9SeedResponse:
        """Install the deterministic demo dialog (facts first, filler after).

        No provider call: this only fills ``full_history`` so the UI can run
        the same scenario in both modes. The facts are placed FIRST so they
        leave the recent window and must survive via the summary.
        """
        config = self._effective_config(recent_messages_limit, compression_batch_size)
        history: list[dict[str, str]] = []
        for role, text in _DEMO_OPENING:
            history.append({"role": role, "content": text})
        for index, text in enumerate(_DEMO_FILLER):
            history.append({"role": "user", "content": text})
            history.append(
                {
                    "role": "assistant",
                    "content": f"Ответ {index + 1}: продолжаем обсуждение.",
                }
            )
        self._state = CompressionState(full_history=history)
        self._config = config
        self._last_answer = None
        return Day9SeedResponse(
            messages=list(history),
            count=len(history),
            diagnostics=self._diagnostics(config),
        )

    def reset(self) -> None:
        """Clear the dialog AND its summary/compression metadata."""
        self._state = CompressionState()
        self._last_answer = None

    def diagnostics(
        self,
        recent_messages_limit: int | None = None,
        compression_batch_size: int | None = None,
    ) -> Day9DiagnosticsResponse:
        config = self._effective_config(recent_messages_limit, compression_batch_size)
        return Day9DiagnosticsResponse(
            model=self._resolved_model(None),
            diagnostics=self._diagnostics(config),
            last_answer=self._last_answer,
        )

    @property
    def state(self) -> CompressionState:
        return self._state
