"""Pydantic schemas for Day 12 — user profile / personalization.

Day 12 separates THREE concepts that are often mixed together:

* **conversation history** — the user/assistant messages;
* **memory / context**     — what the agent already remembers (Day 11);
* **user profile**         — durable PRESENTATION preferences that decide HOW
  the assistant answers (style, format, constraints).

The profile is deliberately small and structured. It is stored SEPARATELY
from the dialog and from memory, and the backend (never the frontend) turns it
into model instructions. The profile may only tweak presentation: it can never
override the application system rules (see
``app.services.day12.profile.build_personalization_instructions``).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.chat import ChatRequest
from app.schemas.day11 import (
    Day11ContextInfo,
    Day11Cost,
    Day11MemoryState,
    Day11Usage,
    MemoryContextStrategy,
)

# Style of the answer (the "how", not the "what").
ResponseStyle = Literal["concise", "detailed", "technical"]
# Language the assistant should answer in.
ProfileLanguage = Literal["ru", "en"]


class ResponseFormat(BaseModel):
    """Controllable formatting preferences (all optional booleans)."""

    structured: bool = False
    use_lists: bool = True
    use_examples: bool = False
    use_code: bool = False


class ResponseConstraints(BaseModel):
    """Hard-ish presentation constraints. ``language`` is structured too."""

    no_emoji: bool = True
    no_intro: bool = False
    no_lists: bool = False
    language: ProfileLanguage = "ru"


class UserProfile(BaseModel):
    """The durable profile of the user.

    It is intentionally compact: style + format + constraints plus one
    free-form ``custom_instructions`` field. ``name`` is an optional
    convenience used only for a friendly personalization line.
    """

    name: str | None = Field(default=None, max_length=100)
    style: ResponseStyle = "concise"
    format: ResponseFormat = Field(default_factory=ResponseFormat)
    constraints: ResponseConstraints = Field(default_factory=ResponseConstraints)
    # Free-form, user-written presentation preferences. Part of PERSONALIZATION
    # (never application system rules) and injected automatically by the backend.
    custom_instructions: str | None = Field(default=None, max_length=4000)

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("custom_instructions")
    @classmethod
    def custom_instructions_must_not_be_blank(
        cls, value: str | None
    ) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ProfilePreset(BaseModel):
    """A named, ready-to-use profile (concise / detailed / technical)."""

    id: str
    title: str
    description: str
    profile: UserProfile


# ----------------------------------------------------------------------
# Profile API
# ----------------------------------------------------------------------
class Day12ProfileResponse(BaseModel):
    """Current active profile + the exact instructions it produces."""

    profile: UserProfile
    instructions: str
    presets: list[ProfilePreset] = Field(default_factory=list)
    demo_prompt: str = ""


# ----------------------------------------------------------------------
# Chat / compare
# ----------------------------------------------------------------------
class Day12ChatRequest(ChatRequest):
    """One personalized turn.

    The user message NEVER carries style instructions: the profile is applied
    automatically by the backend. Memory/context parameters are forwarded to
    the existing Day 11 memory service, so personalization works ON TOP of the
    memory model instead of replacing it.
    """

    session_id: str | None = None
    apply_profile: bool = True
    use_memory: bool = True
    classify: bool = True
    strategy: MemoryContextStrategy = "sliding_window"
    window_size: int | None = Field(default=None, ge=1, le=100)
    model: str | None = None
    system_prompt: str | None = None


class Day12ChatResponse(BaseModel):
    """Answer + the profile that shaped it + memory/context diagnostics."""

    session_id: str
    answer: str
    model: str
    finish_reason: str | None = None
    profile: UserProfile
    profile_applied: bool = True
    personalization_block: str | None = None
    memory: Day11MemoryState
    context: Day11ContextInfo
    usage: Day11Usage
    cost: Day11Cost


class Day12CompareRequest(ChatRequest):
    """Run the SAME question under several profiles (isolated, no memory)."""

    preset_ids: list[str] | None = None


class Day12CompareItem(BaseModel):
    """One profile's answer to the shared demo question."""

    preset_id: str
    title: str
    profile: UserProfile
    personalization_block: str
    answer: str
    finish_reason: str | None = None
    model: str
    usage: Day11Usage
    cost: Day11Cost


class Day12CompareResponse(BaseModel):
    """Same question, different profiles — the core Day 12 demonstration."""

    message: str
    items: list[Day12CompareItem] = Field(default_factory=list)
