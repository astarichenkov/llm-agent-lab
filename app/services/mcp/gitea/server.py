"""Day 20 — Gitea MCP server (local stdio transport).

This is a REAL MCP server (``FastMCP``) spawned as a subprocess by the backend
``MCPClient`` over ``stdio``. It exposes exactly three READ-ONLY tools::

    recent_commits   -> commits in a window (fixed repository)
    get_commit       -> one commit's metadata
    get_commit_diff  -> the bounded unified diff for one commit

The repository is fixed by server configuration (``GITEA_REPOSITORY_OWNER`` /
``GITEA_REPOSITORY_NAME``); it is NOT a tool argument, so the LLM cannot point
the tools at an arbitrary repository. Access is read-only by construction: the
underlying :class:`GiteaClient` only implements ``GET`` requests.

Run standalone for a manual protocol check::

    python -m app.services.mcp.gitea.server
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.services.gitea.client import (
    DEFAULT_BRANCH,
    GiteaClient,
    GiteaConfigError,
    GiteaError,
)

SERVER_NAME = "Gitea MCP"
SERVER_VERSION = "1.0"

RECENT_COMMITS_DESCRIPTION = (
    "List recent git commits for the configured repository (read-only). Use "
    "this to find changes around a point in time (e.g. shortly before an "
    "incident started). Optional ISO8601 UTC `since`/`until` bound the time "
    "range. The repository is fixed by server configuration."
)

GET_COMMIT_DESCRIPTION = (
    "Get metadata (message, author, date, parents) for ONE commit of the "
    "configured repository by sha or ref. Read-only."
)

GET_COMMIT_DIFF_DESCRIPTION = (
    "Get the unified diff (changed files + patch text) for ONE commit sha of "
    "the configured repository. Read-only; the diff is truncated to a bounded "
    "size. Use this to inspect whether a code/config change is related to an "
    "incident."
)


def _client_from_env() -> GiteaClient:
    return GiteaClient.from_env()


def _error(exc: GiteaError) -> dict[str, Any]:
    """Return a controlled, secret-free tool error."""
    return {"error": exc.message, "error_kind": exc.kind}


def build_server() -> FastMCP:
    """Create the Gitea MCP server with its three read-only tools."""
    server = FastMCP(SERVER_NAME)

    @server.tool(description=RECENT_COMMITS_DESCRIPTION)
    async def recent_commits(
        since: str | None = Field(
            None,
            description="Optional ISO8601 UTC lower bound, e.g. 2026-09-20T10:00:00Z.",
        ),
        until: str | None = Field(
            None,
            description="Optional ISO8601 UTC upper bound, e.g. 2026-09-20T11:00:00Z.",
        ),
        branch: str | None = Field(
            None,
            description="Optional branch/tag/sha. Defaults to the configured branch.",
        ),
        limit: int = Field(
            10,
            ge=1,
            le=100,
            description="Maximum number of commits to return (bounded server-side).",
        ),
    ) -> dict[str, Any]:
        """List recent commits for the configured repository (read-only)."""
        client = _client_from_env()
        since_dt = _parse_time(since)
        if isinstance(since_dt, dict):
            return since_dt
        until_dt = _parse_time(until)
        if isinstance(until_dt, dict):
            return until_dt
        try:
            return await client.recent_commits(
                since=since_dt, until=until_dt, branch=branch, limit=limit
            )
        except GiteaConfigError as exc:
            return _error(exc)
        except GiteaError as exc:
            return _error(exc)
        except Exception as exc:  # noqa: BLE001 - controlled tool error
            return {"error": f"Gitea request failed ({exc.__class__.__name__})"}

    @server.tool(description=GET_COMMIT_DESCRIPTION)
    async def get_commit(
        sha: str = Field(
            ...,
            min_length=4,
            max_length=200,
            description="Commit sha (or safe branch/tag ref).",
        ),
    ) -> dict[str, Any]:
        """Get metadata for one commit (read-only)."""
        client = _client_from_env()
        try:
            return await client.get_commit(sha)
        except GiteaConfigError as exc:
            return _error(exc)
        except GiteaError as exc:
            return _error(exc)
        except Exception as exc:  # noqa: BLE001 - controlled tool error
            return {"error": f"Gitea request failed ({exc.__class__.__name__})"}

    @server.tool(description=GET_COMMIT_DIFF_DESCRIPTION)
    async def get_commit_diff(
        sha: str = Field(
            ...,
            min_length=4,
            max_length=64,
            description="Full hexadecimal commit sha (4-64 hex characters).",
        ),
    ) -> dict[str, Any]:
        """Get the bounded unified diff for one commit (read-only)."""
        client = _client_from_env()
        try:
            return await client.get_commit_diff(sha)
        except GiteaConfigError as exc:
            return _error(exc)
        except GiteaError as exc:
            return _error(exc)
        except Exception as exc:  # noqa: BLE001 - controlled tool error
            return {"error": f"Gitea request failed ({exc.__class__.__name__})"}

    return server


def _parse_time(value: str | None) -> datetime | None | dict[str, Any]:
    """Parse an optional ISO8601 timestamp; return a tool error on bad input."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return {
            "error": f"invalid ISO8601 timestamp: {text[:40]}",
            "error_kind": "validation",
        }


mcp = build_server()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    mcp.run(transport="stdio")


__all__ = [
    "SERVER_NAME",
    "RECENT_COMMITS_DESCRIPTION",
    "GET_COMMIT_DESCRIPTION",
    "GET_COMMIT_DIFF_DESCRIPTION",
    "DEFAULT_BRANCH",
    "build_server",
]
