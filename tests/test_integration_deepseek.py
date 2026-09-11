"""Optional real-API smoke test for DeepSeek.

This test is EXCLUDED from the normal test run (see ``addopts`` in
``pyproject.toml``). Run it explicitly with:

    pytest -m integration

It is skipped unless ``DEEPSEEK_API_KEY`` is present in the environment
(or in the local ``.env`` file). It performs exactly ONE tiny request, so
it consumes only a negligible amount of API balance.
"""
import os

import pytest

from app.config import get_settings
from app.services.deepseek import DeepSeekService

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not set; skipping real API smoke test",
)
async def test_real_deepseek_smoke():
    settings = get_settings()
    service = DeepSeekService(settings)
    answer = await service.chat("Reply with exactly: OK")
    assert answer.answer.strip(), "DeepSeek returned an empty answer"
    assert "OK" in answer.answer.upper()


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not set; skipping real API Agent smoke test",
)
async def test_real_deepseek_agent_smoke(tmp_path):
    """One real API call through the persistent Agent (isolated temp DB)."""
    from app.agents.manager import AgentManager
    from app.agents.repository import SQLiteContextRepository

    base = get_settings()
    settings = base.model_copy(update={"agent_db_path": str(tmp_path / "agents.db")})
    repository = SQLiteContextRepository(settings.agent_db_path)
    manager = AgentManager(settings, repository)
    agent = await manager.get_or_create("smoke-agent")

    result = await agent.chat("Reply with exactly: OK")
    assert result.answer.strip(), "DeepSeek returned an empty answer"
    assert "OK" in result.answer.upper()

    # the turn was persisted and restorable
    history = await repository.load_messages("smoke-agent")
    assert [m.role for m in history] == ["user", "assistant"]
