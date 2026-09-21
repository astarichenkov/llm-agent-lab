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


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not set; skipping real Day 8 usage smoke test",
)
async def test_real_deepseek_day8_usage_smoke():
    """One tiny real Day 8 turn: confirms the API returns an EXACT usage block.

    This is deliberately small (a one-word answer) and never runs the
    expensive context-overflow scenario.
    """
    from app.services.day8 import Day8TokenService

    settings = get_settings()
    service = Day8TokenService(settings)
    result = await service.chat(
        message="Reply with exactly: OK",
        history=None,
        model=None,
        system_prompt=None,
    )
    assert result.answer.strip()
    # exact values MUST come from the provider usage block
    assert result.usage.prompt_tokens is not None and result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens is not None
    assert result.usage.total_tokens == (
        result.usage.prompt_tokens + result.usage.completion_tokens
    )
    # local estimate is present but separate
    assert result.usage.estimated_input_tokens > 0
    # documented limits are attached
    assert result.limits.context_window == 1_000_000
    assert result.limits.max_output_tokens == 384_000


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not set; skipping real Day 9 compression smoke test",
)
async def test_real_deepseek_day9_compression_smoke():
    """One tiny real smoke: confirms summary + answer usage and real savings.

    Uses a small recent window and batch so compression triggers with a short
    dialog. Never runs the expensive scenario automatically.
    """
    from app.services.day9 import Day9CompressionService

    settings = get_settings()
    service = Day9CompressionService(settings)
    service.seed_demo(recent_messages_limit=4, compression_batch_size=6)
    result = await service.chat(
        message="Какие ключевые решения мы приняли?",
        mode="compressed",
        recent_messages_limit=4,
        compression_batch_size=6,
    )
    assert result.answer.strip()
    # compression ran and real usage came back from the API
    assert result.diagnostics.compression_cycles >= 1
    assert result.diagnostics.summary
    assert result.usage.prompt_tokens is not None
    # the compressed context is smaller than the full one
    assert result.usage.tokens_saved > 0
    # facts from the beginning must survive in the summary (not verbatim)
    assert "Orion" in result.diagnostics.summary or "FastAPI" in result.diagnostics.summary
