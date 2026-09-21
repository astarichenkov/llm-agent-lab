"""Day 12 — user profile: presets, instruction builder and storage.

This module is the SINGLE place that converts a structured ``UserProfile``
into model instructions. Neither the API layer nor the frontend builds
prompt text: the frontend edits the structured profile, the backend turns it
into instructions.

Priority of instructions is enforced here by construction:

    application system rules          (highest)
       -> USER PROFILE block          (presentation only)
          -> memory / context
             -> conversation history
                -> current user message

The profile block explicitly states that system instructions always win, so a
profile can never disable or weaken the application rules.

Storage is a small atomic JSON file, matching the project's "minimal local
storage" approach (no DB/Redis/vector store just for a profile).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app.schemas.day12 import (
    ProfilePreset,
    ResponseConstraints,
    ResponseFormat,
    UserProfile,
)

logger = logging.getLogger("app.services.day12.profile")

# ----------------------------------------------------------------------
# Instruction fragments (the ONLY place profile -> prompt text happens)
# ----------------------------------------------------------------------
_STYLE_INSTRUCTIONS: dict[str, str] = {
    "concise": (
        "Preferred response style: concise. Keep the answer short and to the "
        "point; omit everything that is not needed."
    ),
    "detailed": (
        "Preferred response style: detailed. Give a thorough explanation, "
        "cover the reasoning and relevant context."
    ),
    "technical": (
        "Preferred response style: technical. Use precise technical "
        "terminology, concrete details and avoid small talk."
    ),
}

_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ru": "Respond in Russian.",
    "en": "Respond in English.",
}

_PROFILE_HEADER = "USER PROFILE (presentation preferences)"

# Kept as an explicit, testable sentence: personalization is presentation-only.
_SYSTEM_PRIORITY_RULE = (
    "IMPORTANT: These are presentation preferences only. They never override, "
    "replace or weaken the application system instructions. If a preference "
    "conflicts with the system rules, the system rules always win."
)


def build_personalization_instructions(profile: UserProfile) -> str:
    """Convert a structured profile into the USER PROFILE instruction block.

    The result is deterministic and contains no conversation data. It is
    inserted as a ``system`` message AFTER the application system prompt (so
    the application rules keep priority) and BEFORE any memory/context.
    """
    lines: list[str] = [_PROFILE_HEADER]
    if profile.name:
        lines.append(f"The user's name is {profile.name}.")

    lines.append(_STYLE_INSTRUCTIONS[profile.style])

    fmt = profile.format
    constraints = profile.constraints

    if fmt.structured:
        lines.append("Structure the answer into clear, labelled sections.")
    # ``no_lists`` is a constraint and therefore overrides ``use_lists``.
    if constraints.no_lists:
        lines.append("Do not use bullet lists; write in flowing prose.")
    elif fmt.use_lists:
        lines.append("Use bullet lists where they improve readability.")
    if fmt.use_examples:
        lines.append("Include concrete examples where they are useful.")
    if fmt.use_code:
        lines.append("Include short code snippets when they are appropriate.")

    if constraints.no_emoji:
        lines.append("Do not use emoji.")
    if constraints.no_intro:
        lines.append(
            "Do not add unnecessary introductions or preambles; start directly "
            "with the answer."
        )

    lines.append(_LANGUAGE_INSTRUCTIONS[constraints.language])

    # Free-form user instructions are part of PERSONALIZATION (presentation
    # only). They are appended verbatim, after the structured preferences and
    # BEFORE the explicit priority rule, so they can never outrank the
    # mandatory application system rules.
    if profile.custom_instructions:
        lines.append("")
        lines.append("Additional user instructions:")
        lines.append(profile.custom_instructions.strip())

    lines.append("")
    lines.append(_SYSTEM_PRIORITY_RULE)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------
PROFILE_PRESETS: list[ProfilePreset] = [
    ProfilePreset(
        id="concise",
        title="Краткий",
        description=(
            "Короткие ответы, минимум пояснений, без вступления и без emoji."
        ),
        profile=UserProfile(
            style="concise",
            format=ResponseFormat(
                structured=False,
                use_lists=False,
                use_examples=False,
                use_code=False,
            ),
            constraints=ResponseConstraints(
                no_emoji=True,
                no_intro=True,
                no_lists=True,
                language="ru",
            ),
        ),
    ),
    ProfilePreset(
        id="detailed",
        title="Подробный",
        description=(
            "Развёрнутые объяснения, структура по разделам, примеры, где полезно."
        ),
        profile=UserProfile(
            style="detailed",
            format=ResponseFormat(
                structured=True,
                use_lists=True,
                use_examples=True,
                use_code=False,
            ),
            constraints=ResponseConstraints(
                no_emoji=True,
                no_intro=False,
                no_lists=False,
                language="ru",
            ),
        ),
    ),
    ProfilePreset(
        id="technical",
        title="Технический",
        description=(
            "Техническая терминология, конкретика, структура, код, когда уместно."
        ),
        profile=UserProfile(
            style="technical",
            format=ResponseFormat(
                structured=True,
                use_lists=True,
                use_examples=False,
                use_code=True,
            ),
            constraints=ResponseConstraints(
                no_emoji=True,
                no_intro=True,
                no_lists=False,
                language="ru",
            ),
        ),
    ),
]

PRESETS_BY_ID: dict[str, ProfilePreset] = {preset.id: preset for preset in PROFILE_PRESETS}

# Sensible default profile when nothing was saved yet.
DEFAULT_PROFILE = PROFILE_PRESETS[0].profile.model_copy(deep=True)

# One shared, style-neutral demo question for the video/demo. It contains NO
# instructions like "answer briefly"/"give a technical answer": any visible
# difference between the answers comes from the USER PROFILE alone.
DAY12_DEMO_PROMPT = (
    "Помоги Антону разобраться: как устроен индекс в базе данных "
    "и почему он ускоряет поиск?"
)


def get_preset(preset_id: str) -> ProfilePreset:
    """Return a preset or raise ``ValueError`` for an unknown id."""
    preset = PRESETS_BY_ID.get(preset_id)
    if preset is None:
        raise ValueError(f"Unknown profile preset: {preset_id!r}")
    return preset


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
class ProfileStore:
    """Minimal, atomic JSON-file persistence for the user profile."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> UserProfile:
        """Read the stored profile; missing/corrupt file -> default profile."""
        try:
            if not self._path.exists():
                return DEFAULT_PROFILE.model_copy(deep=True)
            with open(self._path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return DEFAULT_PROFILE.model_copy(deep=True)
            # Accept both {"profile": {...}} and a bare profile object.
            payload = data.get("profile", data)
            return UserProfile.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - profile must never crash the app
            logger.warning(
                "day12 profile load failed (%s): %s", type(exc).__name__, exc
            )
            return DEFAULT_PROFILE.model_copy(deep=True)

    def save(self, profile: UserProfile) -> None:
        """Atomically persist the profile."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"profile": profile.model_dump()}
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "day12 profile save failed (%s): %s", type(exc).__name__, exc
            )
