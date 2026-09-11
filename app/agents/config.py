"""Per-agent configuration.

Each :class:`~app.agents.agent.Agent` has its own ``AgentConfig``. This is
what makes it possible to spawn several independent agents (different
provider/model/system prompt) that all share the same storage backend.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import Settings

Provider = Literal["deepseek", "openrouter"]

# Agent-layer defaults (Day 6 / Day 7). These are deliberately independent of
# ``app.services.deepseek.DEFAULT_MAX_TOKENS`` so Days 2-5 keep their own
# behaviour. Agents run without reasoning by default: they answer using the
# dialog context, and reasoning would only consume the output-token budget.
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7
DEFAULT_AGENT_NAME = "Default Agent"

# Bumped when a managed default changes and stored configs must be migrated.
CONFIG_VERSION = 2


class AgentConfig(BaseModel):
    """Configuration owned by a single agent (not by a request handler).

    The model is provider-neutral: ``thinking`` is a provider-independent
    intent ("let the model reason before answering"). Each LLM adapter maps
    it to the provider's own mechanism (DeepSeek: ``thinking.type``).
    """

    name: str = DEFAULT_AGENT_NAME
    provider: Provider = "deepseek"
    # ``None`` means "use the provider default" (Settings.deepseek_model /
    # Settings.openrouter_model).
    model: str | None = None
    system_prompt: str = ""
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    # ``False`` (default) = reasoning/thinking disabled for this agent.
    thinking: bool = False
    # Schema version, used to migrate configs persisted by older releases.
    config_version: int = CONFIG_VERSION

    def resolved_model(self, settings: Settings) -> str:
        """Concrete model id for this agent, falling back to the env default."""
        if self.model:
            return self.model
        if self.provider == "openrouter":
            return settings.openrouter_model
        return settings.deepseek_model

    @classmethod
    def default(cls, settings: Settings) -> "AgentConfig":
        """Build the configuration for the automatic ``default`` agent."""
        provider: Provider = (
            settings.agent_default_provider
            if settings.agent_default_provider in ("deepseek", "openrouter")
            else "deepseek"
        )
        model = (
            settings.openrouter_model
            if provider == "openrouter"
            else settings.deepseek_model
        )
        return cls(
            name=DEFAULT_AGENT_NAME,
            provider=provider,
            model=model,
            system_prompt=settings.system_prompt,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            thinking=False,
            config_version=CONFIG_VERSION,
        )
