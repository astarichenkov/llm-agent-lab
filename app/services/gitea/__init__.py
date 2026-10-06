"""Day 20 — read-only Gitea REST integration.

The package is deliberately split in two:

* ``client``  — the HTTP client (URL construction, auth header, timeouts,
  non-2xx handling, response normalization). It knows nothing about MCP/LLM.
* the MCP server lives under ``app.services.mcp.gitea`` and only wraps the
  client with tool definitions.

Only read-only Gitea endpoints are implemented. There are intentionally NO
functions to create commits, branches, files, pull requests or to delete
anything.
"""
from app.services.gitea.client import (
    DEFAULT_BRANCH,
    GiteaAuthError,
    GiteaClient,
    GiteaConfig,
    GiteaConfigError,
    GiteaError,
    GiteaNotFoundError,
    GiteaTimeoutError,
    GiteaUnavailableError,
)

__all__ = [
    "DEFAULT_BRANCH",
    "GiteaAuthError",
    "GiteaClient",
    "GiteaConfig",
    "GiteaConfigError",
    "GiteaError",
    "GiteaNotFoundError",
    "GiteaTimeoutError",
    "GiteaUnavailableError",
]
