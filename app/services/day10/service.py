"""Day 10 — context-management service.

Orchestrates the three strategies, the optional facts-extraction call and the
token accounting. The provider call reuses the EXISTING ``DeepSeekService`` and
the token estimator from Day 8 — no second LLM client and no second tokenizer.

All strategy state is in-memory and isolated from the Day 7 SQLite history.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.schemas.day10 import (
    BranchInfo,
    Day10BranchActivateResponse,
    Day10BranchCreateResponse,
    Day10ChatResponse,
    Day10ContextInfo,
    Day10Cost,
    Day10DemoRunResponse,
    Day10DemoStep,
    Day10EvaluateResponse,
    Day10ScenarioResponse,
    Day10StateResponse,
    Day10Usage,
    Facts,
)
from app.services import model_params
from app.services.day10.models import Conversation
from app.services.day10.scenario import (
    BRANCH_A,
    BRANCH_B,
    BRANCHING_COMMON,
    EXPECTED_FACTS_LABELS,
    SCENARIO_MESSAGES,
    branching_setup,
    evaluate_answer,
)
from app.services.day10.strategies import (
    BranchingStrategy,
    ContextStrategy,
    ExtractionResult,
    SlidingWindowStrategy,
    StickyFactsStrategy,
    branch_info,
)
from app.services.deepseek import DeepSeekService
from app.services.token_estimate import (
    estimate_message_tokens,
    estimate_messages_tokens,
)

logger = logging.getLogger("app.services.day10")

_PREVIEW_LIMIT = 16


def _sum_optional(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def scenario_definition() -> Day10ScenarioResponse:
    """The shared demo scenario (no service instance needed)."""
    return Day10ScenarioResponse(
        messages=list(SCENARIO_MESSAGES),
        expected=list(EXPECTED_FACTS_LABELS),
        branching=branching_setup(),
    )


class Day10ContextService:
    """Owns one independent dialog PER strategy and switches between them."""

    def __init__(
        self, settings: Settings, deepseek: DeepSeekService | None = None
    ) -> None:
        self._settings = settings
        self._deepseek = deepseek if deepseek is not None else DeepSeekService(settings)
        self._strategies: dict[str, ContextStrategy] = {
            "sliding_window": SlidingWindowStrategy(
                window_size=settings.day10_window_size
            ),
            "sticky_facts": StickyFactsStrategy(
                recent_messages_limit=settings.day10_recent_messages_limit
            ),
            "branching": BranchingStrategy(),
        }

    # ------------------------------------------------------------------
    # config / helpers
    # ------------------------------------------------------------------
    def _strategy(self, name: str) -> ContextStrategy:
        strategy = self._strategies.get(name)
        if strategy is None:
            raise ValueError(f"unknown context strategy: {name}")
        return strategy

    def _apply_params(
        self,
        strategy: ContextStrategy,
        window_size: int | None,
        recent_messages_limit: int | None,
    ) -> None:
        if isinstance(strategy, SlidingWindowStrategy):
            if window_size is not None:
                strategy.window_size = window_size
            else:
                strategy.window_size = self._settings.day10_window_size
        elif isinstance(strategy, StickyFactsStrategy):
            if recent_messages_limit is not None:
                strategy.recent_messages_limit = recent_messages_limit
            else:
                strategy.recent_messages_limit = self._settings.day10_recent_messages_limit

    def _resolved_model(self, model: str | None) -> str:
        return model or self._settings.deepseek_model

    def _system_prompt(self, system_prompt: str | None) -> str:
        if system_prompt is not None:
            return system_prompt
        return self._settings.system_prompt

    # ------------------------------------------------------------------
    # response assembly
    # ------------------------------------------------------------------
    def _context_info(
        self, strategy: ContextStrategy, built
    ) -> Day10ContextInfo:
        info = Day10ContextInfo(
            strategy=strategy.name,
            total_history_messages=built.total_history_messages,
            sent_history_messages=built.sent_history_messages,
            dropped_messages=len(built.dropped_messages),
            dropped_preview=list(built.dropped_messages[-_PREVIEW_LIMIT:]),
            sent_preview=list(built.history_messages[-_PREVIEW_LIMIT:]),
            window_size=built.window_size,
        )
        extra = built.extra or {}
        if "facts" in extra:
            info.facts = Facts(**extra["facts"])
            info.facts_block = extra.get("facts_block")
        if "branches" in extra:
            info.branches = [BranchInfo(**b) for b in extra["branches"]]
            info.active_branch_id = extra.get("active_branch_id")
            info.active_branch_name = extra.get("active_branch_name")
            info.checkpoint_message_id = extra.get("checkpoint_message_id")
        return info

    def _build_usage(
        self,
        built,
        *,
        user_message: str,
        provider_usage: dict | None,
        extraction: ExtractionResult | None,
    ) -> Day10Usage:
        system_tokens = (
            estimate_message_tokens(built.system_prompt) if built.system_prompt else 0
        )
        facts_tokens = (
            estimate_message_tokens(built.facts_message["content"])
            if built.facts_message
            else 0
        )
        recent_tokens = estimate_messages_tokens(built.history_messages)
        current_tokens = estimate_message_tokens(user_message) if user_message else 0
        provider_usage = provider_usage or {}
        extraction = extraction or ExtractionResult()
        return Day10Usage(
            prompt_tokens=provider_usage.get("prompt_tokens"),
            completion_tokens=provider_usage.get("completion_tokens"),
            total_tokens=provider_usage.get("total_tokens"),
            facts_extraction_prompt_tokens=extraction.prompt_tokens,
            facts_extraction_completion_tokens=extraction.completion_tokens,
            facts_extraction_total_tokens=extraction.total_tokens,
            system_prompt_tokens_estimated=system_tokens,
            facts_tokens_estimated=facts_tokens,
            recent_tokens_estimated=recent_tokens,
            current_user_tokens_estimated=current_tokens,
            estimated_input_tokens=system_tokens + facts_tokens + recent_tokens + current_tokens,
            total_history_messages=built.total_history_messages,
            sent_history_messages=built.sent_history_messages,
            dropped_messages=len(built.dropped_messages),
            window_size=built.window_size,
        )

    def _cost(self, model: str, usage: Day10Usage) -> Day10Cost:
        raw = model_params.estimate_cost(
            model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return Day10Cost(
            input=float(raw["input"]) if raw["input"] is not None else None,
            output=float(raw["output"]) if raw["output"] is not None else None,
            total=float(raw["total"]) if raw["total"] is not None else None,
            currency=raw["currency"],
            estimated=raw["estimated"],
            reason=raw.get("reason"),
        )

    def _state_context(self, strategy: ContextStrategy) -> Day10ContextInfo:
        """Build a read-only context preview for the state endpoints."""
        built = strategy.build_context(
            system_prompt=self._system_prompt(None), user_message=""
        )
        return self._context_info(strategy, built)

    # ------------------------------------------------------------------
    # main operation
    # ------------------------------------------------------------------
    async def chat(
        self,
        *,
        strategy: str,
        message: str,
        window_size: int | None = None,
        recent_messages_limit: int | None = None,
        update_facts: bool = True,
        branch_id: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Day10ChatResponse:
        strat = self._strategy(strategy)
        self._apply_params(strat, window_size, recent_messages_limit)
        resolved_model = self._resolved_model(model)
        system = self._system_prompt(system_prompt)

        if branch_id is not None:
            if not isinstance(strat, BranchingStrategy):
                raise ValueError("branch_id is only valid for the branching strategy")
            try:
                strat.activate(branch_id)
            except KeyError as exc:
                raise ValueError(f"unknown branch: {branch_id}") from exc

        extraction = await strat.before_build(
            user_message=message,
            provider=self._deepseek,
            model=resolved_model,
            enabled=update_facts,
        )

        built = strat.build_context(system_prompt=system, user_message=message)
        limits = model_params.get_limits(resolved_model)
        content, finish_reason, provider_usage = await self._deepseek.generate(
            built.messages,
            model=resolved_model,
            max_tokens=limits.max_output_tokens,
            thinking=False,
        )
        strat.record_turn(message, content)

        usage = self._build_usage(
            built,
            user_message=message,
            provider_usage=provider_usage,
            extraction=extraction,
        )
        context = self._context_info(strat, built)

        logger.info(
            "day10 turn completed strategy=%s model=%s history=%s sent=%s "
            "dropped=%s facts_extraction=%s finish_reason=%s prompt_tokens=%s "
            "completion_tokens=%s",
            strat.name,
            resolved_model,
            usage.total_history_messages,
            usage.sent_history_messages,
            usage.dropped_messages,
            bool(extraction and extraction.performed),
            finish_reason,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        return Day10ChatResponse(
            strategy=strat.name,
            answer=content,
            model=resolved_model,
            finish_reason=finish_reason,
            usage=usage,
            cost=self._cost(resolved_model, usage),
            context=context,
            facts_extraction_performed=bool(extraction and extraction.performed),
            facts_error=extraction.error if extraction else None,
            active_branch_id=context.active_branch_id,
        )

    # ------------------------------------------------------------------
    # state / reset / branches
    # ------------------------------------------------------------------
    def state(
        self,
        strategy: str,
        window_size: int | None = None,
        recent_messages_limit: int | None = None,
    ) -> Day10StateResponse:
        strat = self._strategy(strategy)
        self._apply_params(strat, window_size, recent_messages_limit)
        return Day10StateResponse(
            strategy=strat.name,
            model=self._resolved_model(None),
            context=self._state_context(strat),
        )

    def reset(self, strategy: str) -> None:
        self._strategy(strategy).reset()

    def create_branch(
        self,
        *,
        name: str | None,
        parent_branch_id: str | None,
        checkpoint_message_id: str | None,
    ) -> Day10BranchCreateResponse:
        strat = self._strategies["branching"]
        assert isinstance(strat, BranchingStrategy)
        try:
            branch = strat.create_branch(
                name=name,
                parent_branch_id=parent_branch_id,
                checkpoint_message_id=checkpoint_message_id,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        return Day10BranchCreateResponse(
            branch=branch_info(strat.conversation, branch),
            context=self._state_context(strat),
        )

    def activate_branch(self, branch_id: str) -> Day10BranchActivateResponse:
        strat = self._strategies["branching"]
        assert isinstance(strat, BranchingStrategy)
        try:
            strat.activate(branch_id)
        except KeyError as exc:
            raise ValueError(f"unknown branch: {branch_id}") from exc
        return Day10BranchActivateResponse(context=self._state_context(strat))

    def branching_snapshot(self) -> dict[str, Any]:
        strat = self._strategies["branching"]
        assert isinstance(strat, BranchingStrategy)
        conv: Conversation = strat.conversation
        return {
            "active_branch_id": conv.active_branch_id,
            "branches": strat.snapshot()["branches"],
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "branch_id": m.branch_id,
                }
                for m in conv.messages
            ],
        }

    # ------------------------------------------------------------------
    # demo scenarios
    # ------------------------------------------------------------------
    def scenario(self) -> Day10ScenarioResponse:
        return scenario_definition()

    def evaluate(self, answer: str, expected: list[str] | None = None) -> Day10EvaluateResponse:
        return Day10EvaluateResponse(**evaluate_answer(answer, expected))

    def seed_branching_demo(self) -> Day10StateResponse:
        """Install the deterministic branching demo (common prefix + 2 branches)."""
        strat = self._strategies["branching"]
        assert isinstance(strat, BranchingStrategy)
        strat.reset()
        conv = strat.conversation
        main = conv.active_branch
        for message in BRANCHING_COMMON:
            conv.add_message(message["role"], message["content"], branch_id=main.id)
        main = conv.active_branch
        checkpoint_id = conv.branch_history(main.id)[-1].id if conv.branch_history(main.id) else None
        branch_a = strat.create_branch(
            name=BRANCH_A["name"],
            parent_branch_id=main.id,
            checkpoint_message_id=checkpoint_id,
        )
        strat.create_branch(
            name=BRANCH_B["name"],
            parent_branch_id=main.id,
            checkpoint_message_id=checkpoint_id,
        )
        strat.activate(branch_a.id)
        return self.state("branching")

    async def run_scenario(
        self,
        *,
        strategy: str,
        window_size: int | None = None,
        recent_messages_limit: int | None = None,
        update_facts: bool = True,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Day10DemoRunResponse:
        """Run the shared TS scenario end-to-end under one strategy.

        Used by tests and by the optional batch endpoint. The UI normally
        drives the same scenario one message at a time to show progress.
        """
        self.reset(strategy)
        steps: list[Day10DemoStep] = []
        usages: list[Day10Usage] = []
        last_context: Day10ContextInfo | None = None
        for index, message in enumerate(SCENARIO_MESSAGES, start=1):
            response = await self.chat(
                strategy=strategy,
                message=message,
                window_size=window_size,
                recent_messages_limit=recent_messages_limit,
                update_facts=update_facts,
                model=model,
                system_prompt=system_prompt,
            )
            usages.append(response.usage)
            last_context = response.context
            steps.append(
                Day10DemoStep(
                    index=index,
                    message=message,
                    answer=response.answer,
                    usage=response.usage,
                    context=response.context,
                    facts_error=response.facts_error,
                )
            )
        final_answer = steps[-1].answer if steps else ""
        return Day10DemoRunResponse(
            strategy=strategy,  # type: ignore[arg-type]
            steps=steps,
            final_answer=final_answer,
            usage=self._aggregate_usage(usages),
            evaluation=self.evaluate(final_answer),
            context=last_context or Day10ContextInfo(strategy=strategy),  # type: ignore[arg-type]
        )

    @staticmethod
    def _aggregate_usage(usages: list[Day10Usage]) -> Day10Usage:
        if not usages:
            return Day10Usage()
        return Day10Usage(
            prompt_tokens=_sum_optional([u.prompt_tokens for u in usages]),
            completion_tokens=_sum_optional([u.completion_tokens for u in usages]),
            total_tokens=_sum_optional([u.total_tokens for u in usages]),
            facts_extraction_prompt_tokens=_sum_optional(
                [u.facts_extraction_prompt_tokens for u in usages]
            ),
            facts_extraction_completion_tokens=_sum_optional(
                [u.facts_extraction_completion_tokens for u in usages]
            ),
            facts_extraction_total_tokens=_sum_optional(
                [u.facts_extraction_total_tokens for u in usages]
            ),
            system_prompt_tokens_estimated=sum(
                u.system_prompt_tokens_estimated for u in usages
            ),
            facts_tokens_estimated=sum(u.facts_tokens_estimated for u in usages),
            recent_tokens_estimated=sum(u.recent_tokens_estimated for u in usages),
            current_user_tokens_estimated=sum(
                u.current_user_tokens_estimated for u in usages
            ),
            estimated_input_tokens=sum(u.estimated_input_tokens for u in usages),
            total_history_messages=usages[-1].total_history_messages,
            sent_history_messages=usages[-1].sent_history_messages,
            dropped_messages=usages[-1].dropped_messages,
            window_size=usages[-1].window_size,
        )
