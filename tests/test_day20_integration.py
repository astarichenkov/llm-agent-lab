"""Day 20 — opt-in integration smoke test against the REAL Gitea homelab.

This test is EXCLUDED from the normal suite (``pyproject.toml`` sets
``addopts = "-m 'not integration'"``). Run it explicitly with::

    pytest -m integration tests/test_day20_integration.py -q

It is skipped automatically when the ``GITEA_*`` environment variables are not
configured, so it never invents credentials and never touches a real service in
CI. The token is never printed.
"""
from __future__ import annotations

import os

import pytest

from app.services.gitea.client import GiteaClient, GiteaError
from app.services.mcp import MCPClient, gitea_server_config

pytestmark = pytest.mark.integration


def _gitea_configured() -> bool:
    return all(
        os.environ.get(name)
        for name in (
            "GITEA_BASE_URL",
            "GITEA_REPOSITORY_OWNER",
            "GITEA_REPOSITORY_NAME",
        )
    )


@pytest.mark.skipif(
    not _gitea_configured(),
    reason="GITEA_BASE_URL / GITEA_REPOSITORY_OWNER / GITEA_REPOSITORY_NAME not set",
)
async def test_gitea_mcp_real_smoke() -> None:
    """Real MCP stdio handshake + tools/list + read-only calls against Gitea."""
    env = {
        "GITEA_BASE_URL": os.environ["GITEA_BASE_URL"],
        "GITEA_TOKEN": os.environ.get("GITEA_TOKEN", ""),
        "GITEA_REPOSITORY_OWNER": os.environ["GITEA_REPOSITORY_OWNER"],
        "GITEA_REPOSITORY_NAME": os.environ["GITEA_REPOSITORY_NAME"],
        "GITEA_DEFAULT_BRANCH": os.environ.get("GITEA_DEFAULT_BRANCH", "main"),
    }
    client = MCPClient(gitea_server_config(env=env))

    status = await client.discover_tools()
    assert status.connected is True, status.error
    names = {tool.name for tool in status.tools}
    assert names == {"recent_commits", "get_commit", "get_commit_diff"}

    recent = await client.call_tool("recent_commits", {"limit": 1})
    payload = recent.as_payload()
    assert not recent.is_error, payload
    assert "commits" in payload

    commits = payload.get("commits") or []
    if not commits:
        pytest.skip("repository has no commits to inspect")

    sha = commits[0]["sha"]
    commit = (await client.call_tool("get_commit", {"sha": sha})).as_payload()
    assert commit.get("sha") == sha

    diff = (await client.call_tool("get_commit_diff", {"sha": sha})).as_payload()
    assert diff.get("sha") == sha
    assert isinstance(diff.get("diff"), str)


@pytest.mark.skipif(not _gitea_configured(), reason="Gitea not configured")
async def test_gitea_http_client_real_smoke() -> None:
    """Direct read-only HTTP client smoke (no token is ever printed)."""
    try:
        result = await GiteaClient.from_env().recent_commits(limit=1)
    except GiteaError as exc:  # pragma: no cover - environment dependent
        if exc.kind in ("unavailable", "timeout"):
            pytest.skip(f"Gitea unreachable: {exc.kind}")
        raise
    assert result["repository"]
    assert "commits" in result
