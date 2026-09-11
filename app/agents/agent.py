"""The :class:`Agent` — owner of one dialog, its config and its persistence.

An Agent encapsulates everything the task asked for:

* its own :class:`~app.agents.config.AgentConfig`;
* its own message history (restored from storage on first use);
* adding user / assistant messages;
* building the ``messages`` payload for the model (system prompt + history);
* calling the injected LLM client;
* persisting every turn through the injected repository.

The HTTP layer only does ``response = await agent.chat(message)``.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel

from app.agents.config import AgentConfig
from app.agents.llm import LLMClient
from app.agents.models import Message
from app.agents.repository import ContextRepository

logger = logging.getLogger("app.agents.agent")


class AgentChatResult(BaseModel):
    """Structured result of :meth:`Agent.chat`."""

    agent_id: str
    answer: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: dict | None = None
    message_count: int


class Agent:
    """A self-contained conversational agent bound to one ``agent_id``."""

    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        llm_client: LLMClient,
        context_repository: ContextRepository,
        settings,
    ) -> None:
        self.agent_id = agent_id
        self.config = config
        self._llm_client = llm_client
        self._context_repository = context_repository
        self._settings = settings
        self._history: list[Message] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------
    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._history = await self._context_repository.load_messages(self.agent_id)
            self._loaded = True

    async def get_history(self) -> list[Message]:
        """Return the full dialog history (loading it from storage if needed)."""
        await self._ensure_loaded()
        return list(self._history)

    async def clear_history(self) -> None:
        """Erase the dialog history (the agent config is preserved)."""
        await self._context_repository.clear_messages(self.agent_id)
        self._history = []
        self._loaded = True

    def build_messages(self) -> list[dict[str, str]]:
        """System prompt (from config) + stored user/assistant turns."""
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(message.as_payload() for message in self._history)
        return messages

    # ------------------------------------------------------------------
    # main operation
    # ------------------------------------------------------------------
    async def chat(self, user_message: str) -> AgentChatResult:
        """Send one user message and return the assistant reply.

        The turn is atomic with respect to persistence: the user message is
        only written to storage AFTER the provider answered successfully. A
        failed provider call (network error, output-limit, malformed response)
        therefore does NOT leave an unanswered user turn behind in the
        persistent history — repeated failures cannot accumulate orphaned
        user messages. On success the user turn and the assistant reply are
        persisted in order.
        """
        text = user_message.strip()
        if not text:
            raise ValueError("user_message must not be empty")

        await self._ensure_loaded()

        user = Message(role="user", content=text)
        # Build the payload for THIS turn without mutating the stored history.
        pending_messages = self.build_messages() + [user.as_payload()]

        model = self.config.resolved_model(self._settings)
        result = await self._llm_client.complete(
            pending_messages,
            model=model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            thinking=self.config.thinking,
        )

        # Provider answered -> commit the whole turn exactly once.
        await self._context_repository.add_message(self.agent_id, user)
        self._history.append(user)
        assistant = Message(role="assistant", content=result.content)
        await self._context_repository.add_message(self.agent_id, assistant)
        self._history.append(assistant)

        logger.info(
            "agent turn completed agent_id=%s provider=%s model=%s "
            "max_tokens=%s thinking=%s history_messages=%s "
            "finish_reason=%s content_length=%s usage=%s",
            self.agent_id,
            self.config.provider,
            model,
            self.config.max_tokens,
            self.config.thinking,
            len(self._history),
            result.finish_reason,
            len(result.content),
            result.usage,
        )

        return AgentChatResult(
            agent_id=self.agent_id,
            answer=result.content,
            provider=self.config.provider,
            model=model,
            finish_reason=result.finish_reason,
            usage=result.usage,
            message_count=len(self._history),
        )
