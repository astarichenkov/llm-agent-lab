"""The Agent layer (Day 6 stateless + Day 7 persistent).

Public surface:

* :class:`~app.agents.config.AgentConfig` — per-agent configuration;
* :class:`~app.agents.models.Message` — one stored user/assistant message;
* :class:`~app.agents.agent.Agent` — owns dialog state, calls the LLM and
  (optionally) persists the conversation;
* :class:`~app.agents.repository.ContextRepository` with
  :class:`~app.agents.repository.InMemoryContextRepository` (Day 6,
  transient) and :class:`~app.agents.repository.SQLiteContextRepository`
  (Day 7, persistent);
* :class:`~app.agents.manager.AgentManager` — registry/factory for agents.
"""
from app.agents.agent import Agent, AgentChatResult
from app.agents.config import AgentConfig
from app.agents.llm import DeepSeekLLMClient, LLMClient, LLMResult, OpenRouterLLMClient
from app.agents.manager import AgentManager
from app.agents.models import Message
from app.agents.repository import (
    ContextRepository,
    InMemoryContextRepository,
    SQLiteContextRepository,
)

__all__ = [
    "Agent",
    "AgentChatResult",
    "AgentConfig",
    "AgentManager",
    "ContextRepository",
    "DeepSeekLLMClient",
    "InMemoryContextRepository",
    "LLMClient",
    "LLMResult",
    "Message",
    "OpenRouterLLMClient",
    "SQLiteContextRepository",
]
