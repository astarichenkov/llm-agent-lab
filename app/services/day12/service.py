"""Day 12 — personalization service.

The service owns exactly one thing: the **user profile**. It does NOT
re-implement memory or the provider client. Every personalized turn is
delegated to the existing Day 11 memory service, which receives an extra
``personalization_block``:

    user message
      -> Day12ProfileService builds USER PROFILE instructions (single place)
      -> Day11MemoryService assembles:
             system rules
             + USER PROFILE block      <-- personalization (presentation only)
             + long-term / working memory
             + short-term window
             + current user message
      -> existing DeepSeek client

So personalization works ON TOP of the memory model. The profile is stored
separately (its own JSON file) and is never duplicated inside memory/facts.

``compare`` runs the SAME question under several profiles in ISOLATED
temporary sessions with memory disabled, which makes the effect of the profile
alone observable (and cheap to test with a mocked provider).
"""
from __future__ import annotations

import logging
from uuid import uuid4

from app.config import Settings
from app.schemas.day12 import (
    Day12ChatResponse,
    Day12CompareItem,
    Day12CompareResponse,
    Day12ProfileResponse,
    ProfilePreset,
    UserProfile,
)
from app.services.day12.profile import (
    DAY12_DEMO_PROMPT,
    PROFILE_PRESETS,
    ProfileStore,
    build_personalization_instructions,
    get_preset,
)
from app.services.day11.service import Day11MemoryService

logger = logging.getLogger("app.services.day12")


def _presets_for(preset_ids: list[str] | None) -> list[ProfilePreset]:
    if not preset_ids:
        return list(PROFILE_PRESETS)
    return [get_preset(preset_id) for preset_id in preset_ids]


class Day12ProfileService:
    """Owns the user profile and applies it to every chat request."""

    def __init__(
        self,
        settings: Settings,
        memory_service: Day11MemoryService,
        profile_store: ProfileStore | None = None,
    ) -> None:
        self._settings = settings
        self._memory = memory_service
        self._store = profile_store or ProfileStore(settings.day12_profile_path)
        self._profile = self._store.load()
        logger.info(
            "day12 profile service initialised style=%s path=%s",
            self._profile.style,
            self._store.path,
        )

    # ------------------------------------------------------------------
    # profile access
    # ------------------------------------------------------------------
    @property
    def profile(self) -> UserProfile:
        return self._profile

    @property
    def instructions(self) -> str:
        """The exact USER PROFILE instruction block for the active profile."""
        return build_personalization_instructions(self._profile)

    def presets(self) -> list[ProfilePreset]:
        return list(PROFILE_PRESETS)

    def response(self) -> Day12ProfileResponse:
        return Day12ProfileResponse(
            profile=self._profile,
            instructions=self.instructions,
            presets=self.presets(),
            demo_prompt=DAY12_DEMO_PROMPT,
        )

    def update_profile(self, profile: UserProfile) -> Day12ProfileResponse:
        """Replace the active profile and persist it."""
        self._profile = profile
        self._store.save(self._profile)
        logger.info(
            "day12 profile updated style=%s format=%s constraints=%s",
            profile.style,
            profile.format.model_dump(),
            profile.constraints.model_dump(),
        )
        return self.response()

    def apply_preset(self, preset_id: str) -> Day12ProfileResponse:
        """Apply one of the built-in profiles (raises ``ValueError`` if unknown)."""
        preset = get_preset(preset_id)
        return self.update_profile(preset.profile.model_copy(deep=True))

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    async def chat(
        self,
        *,
        message: str,
        session_id: str | None = None,
        apply_profile: bool = True,
        use_memory: bool = True,
        classify: bool = True,
        strategy: str = "sliding_window",
        window_size: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Day12ChatResponse:
        """One personalized turn: profile + memory + answer."""
        personalization_block = (
            build_personalization_instructions(self._profile)
            if apply_profile
            else None
        )
        result = await self._memory.chat(
            message=message,
            session_id=session_id,
            strategy=strategy,  # type: ignore[arg-type]
            window_size=window_size,
            use_memory=use_memory,
            classify=classify,
            model=model,
            system_prompt=system_prompt,
            personalization_block=personalization_block,
        )
        return Day12ChatResponse(
            session_id=result.session_id,
            answer=result.answer,
            model=result.model,
            finish_reason=result.finish_reason,
            profile=self._profile,
            profile_applied=apply_profile,
            personalization_block=personalization_block,
            memory=result.memory,
            context=result.context,
            usage=result.usage,
            cost=result.cost,
        )

    # ------------------------------------------------------------------
    # compare (same question, different profiles)
    # ------------------------------------------------------------------
    async def compare(
        self,
        *,
        message: str,
        preset_ids: list[str] | None = None,
    ) -> Day12CompareResponse:
        """Answer the SAME message under several profiles.

        Each profile runs in a fresh temporary session with memory layers
        disabled, so the ONLY difference between the requests is the USER
        PROFILE block. This never modifies the user's active session, working
        or long-term memory.
        """
        presets = _presets_for(preset_ids)
        previous_session = self._memory.state().session_id
        items: list[Day12CompareItem] = []
        try:
            for preset in presets:
                block = build_personalization_instructions(preset.profile)
                # Isolated session: unique id => empty short-term context.
                session_id = f"day12-compare-{preset.id}-{uuid4().hex[:6]}"
                result = await self._memory.chat(
                    message=message,
                    session_id=session_id,
                    strategy="full",
                    use_memory=False,
                    classify=False,
                    personalization_block=block,
                )
                items.append(
                    Day12CompareItem(
                        preset_id=preset.id,
                        title=preset.title,
                        profile=preset.profile,
                        personalization_block=block,
                        answer=result.answer,
                        finish_reason=result.finish_reason,
                        model=result.model,
                        usage=result.usage,
                        cost=result.cost,
                    )
                )
        finally:
            # Restore the user's session so compare is side-effect free.
            self._memory.activate_session(previous_session)

        logger.info(
            "day12 compare completed message_chars=%s presets=%s",
            len(message),
            ",".join(preset.id for preset in presets),
        )
        return Day12CompareResponse(message=message, items=items)
