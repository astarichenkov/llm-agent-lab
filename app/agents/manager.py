"""AgentManager — a tiny registry/factory for :class:`Agent` instances.

It is intentionally simple: Python objects may be recreated on restart; the
durable state lives in the repository (SQLite). The manager only remembers the
live objects created during the current process and can spawn any number of
independent agents::

    agent_a = await manager.get_or_create("agent-a", config_a)
    agent_b = await manager.get_or_create("agent-b", config_b)
"""
from __future__ import annotations

import logging

from app.agents.agent import Agent
from app.agents.config import AgentConfig, CONFIG_VERSION
from app.agents.llm import DeepSeekLLMClient, LLMClient, OpenRouterLLMClient
from app.agents.repository import (
    ContextRepository,
    InMemoryContextRepository,
    SQLiteContextRepository,
)
from app.config import Settings
from app.services.deepseek import DeepSeekService
from app.services.openrouter_service import OpenRouterService

logger = logging.getLogger("app.agents.manager")

DEFAULT_AGENT_ID = "default"


class AgentManager:
    """Owns live agents and the shared storage/LLM dependencies."""

    def __init__(
        self,
        settings: Settings,
        repository: ContextRepository,
        llm_clients: dict[str, LLMClient] | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        # Built lazily so merely creating the app/repository never constructs
        # provider HTTP clients (keeps imports/tests fast and side-effect free).
        self._llm_clients = llm_clients
        self._agents: dict[str, Agent] = {}

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_clients(settings: Settings) -> dict[str, LLMClient]:
        return {
            "deepseek": DeepSeekLLMClient(DeepSeekService(settings)),
            "openrouter": OpenRouterLLMClient(OpenRouterService(settings)),
        }

    @classmethod
    def for_settings(cls, settings: Settings) -> "AgentManager":
        """Build the production manager (SQLite + real provider adapters).

        Constructing the repository/SQLite repository does not touch the disk;
        the database file is created lazily on first read/write.
        """
        return cls(
            settings=settings,
            repository=SQLiteContextRepository(settings.agent_db_path),
        )

    @property
    def repository(self) -> ContextRepository:
        return self._repository

    @property
    def settings(self) -> Settings:
        return self._settings

    # ------------------------------------------------------------------
    # registry
    # ------------------------------------------------------------------
    def _client_for(self, provider: str) -> LLMClient | None:
        if self._llm_clients is None:
            self._llm_clients = self._build_clients(self._settings)
        return self._llm_clients.get(provider)

    def _register(self, agent_id: str, config: AgentConfig) -> Agent:
        client = self._client_for(config.provider)
        if client is None:
            raise ValueError(f"No LLM client registered for provider '{config.provider}'")
        agent = Agent(
            agent_id=agent_id,
            config=config,
            llm_client=client,
            context_repository=self._repository,
            settings=self._settings,
        )
        self._agents[agent_id] = agent
        logger.info("agent created agent_id=%s provider=%s", agent_id, config.provider)
        return agent

    def _migrate_default_config(
        self, agent_id: str, stored: AgentConfig
    ) -> AgentConfig:
        """Bring the app-managed ``default`` agent up to the current defaults.

        Older releases persisted ``max_tokens=1024`` and had no ``thinking``
        field. For the automatically-created ``default`` agent we re-sync the
        app-managed fields (output budget, reasoning flag, config version) so
        an upgrade cannot leave it stuck on the old configuration. Dialog
        messages are never touched. Non-default agents keep their own
        (possibly user-configured) values.
        """
        if agent_id != DEFAULT_AGENT_ID:
            return stored
        defaults = AgentConfig.default(self._settings)
        updates: dict = {}
        if stored.max_tokens != defaults.max_tokens:
            updates["max_tokens"] = defaults.max_tokens
        if stored.thinking != defaults.thinking:
            updates["thinking"] = defaults.thinking
        if stored.config_version != CONFIG_VERSION:
            updates["config_version"] = CONFIG_VERSION
        if not updates:
            return stored
        logger.info(
            "migrating persisted config agent_id=%s from_version=%s updates=%s",
            agent_id,
            stored.config_version,
            sorted(updates),
        )
        return stored.model_copy(update=updates)

    async def get_or_create(
        self, agent_id: str, config: AgentConfig | None = None
    ) -> Agent:
        """Return the live agent, restoring/persisting config as needed.

        * an explicit ``config`` is always saved (upsert) and used;
        * otherwise a stored config is reused (the ``default`` agent is
          migrated to the current defaults first);
        * otherwise the default config is created and saved.
        """
        if agent_id in self._agents and config is None:
            return self._agents[agent_id]

        stored = await self._repository.load_config(agent_id)
        if config is not None:
            chosen = config
            await self._repository.save_config(agent_id, chosen)
        elif stored is not None:
            chosen = self._migrate_default_config(agent_id, stored)
            if chosen is not stored:
                await self._repository.save_config(agent_id, chosen)
        else:
            chosen = AgentConfig.default(self._settings)
            await self._repository.save_config(agent_id, chosen)
        return self._register(agent_id, chosen)

    async def create_agent(self, agent_id: str, config: AgentConfig) -> Agent:
        """Explicitly create (or reconfigure) an agent and persist its config."""
        return await self.get_or_create(agent_id, config=config)

    def create_transient(
        self,
        config: AgentConfig | None = None,
        agent_id: str = "day6-agent",
    ) -> Agent:
        """Build a STATELESS agent backed by a fresh in-memory repository.

        Used by Day 6: every call creates a new agent with an empty history,
        so no dialog context and no config is written to SQLite. The same
        :class:`Agent` class is reused — only the lifecycle/repository differ.
        """
        chosen = config or AgentConfig.default(self._settings)
        client = self._client_for(chosen.provider)
        if client is None:
            raise ValueError(
                f"No LLM client registered for provider '{chosen.provider}'"
            )
        return Agent(
            agent_id=agent_id,
            config=chosen,
            llm_client=client,
            context_repository=InMemoryContextRepository(),
            settings=self._settings,
        )

    def get(self, agent_id: str) -> Agent | None:
        """Return the live agent if it exists in this process, else ``None``."""
        return self._agents.get(agent_id)

    async def list_agent_ids(self) -> list[str]:
        """All agent ids known to this process or to storage."""
        stored = await self._repository.list_agent_ids()
        return sorted(set(self._agents) | set(stored))
