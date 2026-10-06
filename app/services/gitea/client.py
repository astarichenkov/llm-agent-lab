"""Day 20 — read-only HTTP client for the Gitea REST API.

Responsibilities (and ONLY these):

* resolve the base URL, fixed repository and token from configuration/env;
* issue READ-ONLY ``GET`` requests with a bounded timeout;
* map non-2xx responses, timeouts and transport failures to small,
  secret-free domain errors;
* normalize the parts of the Gitea payloads the agent actually needs;
* bound the amount of data (commit count, diff length) handed back.

No MCP tool definitions and no LLM logic live here. The Gitea MCP server
(``app.services.mcp.gitea.server``) is a thin wrapper around this class.

Security notes
--------------
* The token is sent only as a request header and is never logged, never
  returned in an error message and never included in a result payload.
* Only ``GET`` requests are implemented. There is no code path that can write
  to the repository (no create/update/delete/merge/push).
* The repository is fixed by configuration; it is not a tool argument.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

DEFAULT_BRANCH = "main"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_DIFF_CHARS = 15000
DEFAULT_MAX_COMMITS = 20
MAX_COMMIT_MESSAGE_CHARS = 300

# A commit-ish reference that is safe to interpolate into the API path. Full
# 40-char SHAs, short SHAs and ordinary branch/tag names all match. The pattern
# rejects spaces, control characters, ``..`` and path separators.
_SHA_RE = re.compile(r"^[0-9a-fA-F]{4,64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)


class GiteaError(Exception):
    """Base class for controlled Gitea failures (safe to expose)."""

    kind = "gitea"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class GiteaConfigError(GiteaError):
    """The base URL or the fixed repository is missing."""

    kind = "config"


class GiteaAuthError(GiteaError):
    """Gitea rejected the token (401/403)."""

    kind = "auth"


class GiteaNotFoundError(GiteaError):
    """The repository or commit was not found (404)."""

    kind = "not_found"


class GiteaTimeoutError(GiteaError):
    """The HTTP request exceeded the configured timeout."""

    kind = "timeout"


class GiteaUnavailableError(GiteaError):
    """DNS / TCP / TLS failure reaching Gitea."""

    kind = "unavailable"


@dataclass(frozen=True)
class GiteaConfig:
    """Resolved, read-only Gitea connection settings."""

    base_url: str
    owner: str
    repository: str
    token: str = ""
    default_branch: str = DEFAULT_BRANCH
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_commits: int = DEFAULT_MAX_COMMITS
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS

    @property
    def repo_path(self) -> str:
        """Return the configured ``owner/name`` repository."""
        return f"{self.owner}/{self.repository}" if self.owner and self.repository else ""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "GiteaConfig":
        """Build the config from environment variables.

        ``GITEA_BASE_URL`` / ``GITEA_REPOSITORY_OWNER`` /
        ``GITEA_REPOSITORY_NAME`` are required only when the Gitea tools are
        actually used; validation is lazy (on the first call) so a missing
        configuration never breaks application startup or the other servers.
        """
        source = os.environ if env is None else env

        def _float(name: str, default: float) -> float:
            try:
                return float(source.get(name, str(default)))
            except (TypeError, ValueError):
                return default

        def _int(name: str, default: int) -> int:
            try:
                return int(source.get(name, str(default)))
            except (TypeError, ValueError):
                return default

        return cls(
            base_url=(source.get("GITEA_BASE_URL") or "").strip(),
            owner=(source.get("GITEA_REPOSITORY_OWNER") or "").strip(),
            repository=(source.get("GITEA_REPOSITORY_NAME") or "").strip(),
            token=(source.get("GITEA_TOKEN") or "").strip(),
            default_branch=(
                source.get("GITEA_DEFAULT_BRANCH") or DEFAULT_BRANCH
            ).strip()
            or DEFAULT_BRANCH,
            timeout_seconds=_float("GITEA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            max_commits=max(1, _int("GITEA_MAX_COMMITS", DEFAULT_MAX_COMMITS)),
            max_diff_chars=max(
                1000, _int("GITEA_MAX_DIFF_CHARS", DEFAULT_MAX_DIFF_CHARS)
            ),
        )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def format_time(value: datetime) -> str:
    """Format a datetime as the RFC3339 UTC string Gitea accepts."""
    return _as_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_sha(sha: str) -> str:
    """Return a validated hex SHA (4-64 chars) or raise ``GiteaConfigError``."""
    candidate = (sha or "").strip()
    if not _SHA_RE.match(candidate):
        raise GiteaConfigError(
            "invalid commit sha: expected 4-64 hexadecimal characters"
        )
    return candidate


def validate_ref(ref: str) -> str:
    """Return a validated ref (sha or branch/tag name) or raise."""
    candidate = (ref or "").strip()
    if not candidate:
        raise GiteaConfigError("commit ref must not be empty")
    if _SHA_RE.match(candidate) or _REF_RE.match(candidate):
        return candidate
    raise GiteaConfigError(
        "invalid commit ref: use a hex sha or a safe branch/tag name"
    )


def _short_sha(sha: str) -> str:
    return sha[:7] if sha else ""


def _first_line(message: str) -> str:
    text = (message or "").strip().splitlines()
    line = text[0].strip() if text else ""
    return line[:MAX_COMMIT_MESSAGE_CHARS]


def parse_commit(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize ONE Gitea commit object into the compact agent shape.

    Handles both the list-commits shape and the single-commit shape. Missing
    fields degrade to ``None`` instead of raising, so one odd commit never
    breaks the whole response.
    """
    sha = str(item.get("sha") or "")
    commit = item.get("commit") or {}
    author = commit.get("author") or {}
    if not isinstance(author, dict):
        author = {}
    created = (
        author.get("date")
        or (commit.get("committer") or {}).get("date")
        or item.get("created")
    )
    author_name = author.get("name") or author.get("email") or ""
    item_author = item.get("author")
    if isinstance(item_author, dict):
        author_name = author_name or item_author.get("login") or ""
    parents = [
        str(p.get("sha"))
        for p in (item.get("parents") or [])
        if isinstance(p, dict) and p.get("sha")
    ]
    return {
        "sha": sha,
        "short_sha": _short_sha(sha),
        "message": _first_line(str(commit.get("message") or "")),
        "author": author_name or author.get("email") or "",
        "created_at": created,
        "parents": parents,
    }


def extract_diff_files(diff: str) -> list[str]:
    """Return the file paths mentioned by a unified diff (order preserved)."""
    files: list[str] = []
    for match in _DIFF_FILE_RE.finditer(diff or ""):
        path = match.group(2)
        if path not in files:
            files.append(path)
    return files


class GiteaClient:
    """Read-only, bounded Gitea REST client for ONE configured repository."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        owner: str | None = None,
        repository: str | None = None,
        default_branch: str | None = None,
        timeout_seconds: float | None = None,
        max_commits: int | None = None,
        max_diff_chars: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        base = GiteaConfig.from_env(env)
        self.config = GiteaConfig(
            base_url=(base_url if base_url is not None else base.base_url).rstrip("/"),
            owner=owner if owner is not None else base.owner,
            repository=repository if repository is not None else base.repository,
            token=token if token is not None else base.token,
            default_branch=(
                default_branch if default_branch is not None else base.default_branch
            )
            or DEFAULT_BRANCH,
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else base.timeout_seconds
            ),
            max_commits=(
                max(1, max_commits) if max_commits is not None else base.max_commits
            ),
            max_diff_chars=(
                max(1000, max_diff_chars)
                if max_diff_chars is not None
                else base.max_diff_chars
            ),
        )
        self._transport = transport

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> "GiteaClient":
        return cls(env=env, transport=transport)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_config(self) -> GiteaConfig:
        if not self.config.base_url:
            raise GiteaConfigError("GITEA_BASE_URL is not configured")
        if not self.config.base_url.startswith(("http://", "https://")):
            raise GiteaConfigError("GITEA_BASE_URL must start with http:// or https://")
        if not self.config.owner or not self.config.repository:
            raise GiteaConfigError(
                "GITEA_REPOSITORY_OWNER and GITEA_REPOSITORY_NAME are not configured"
            )
        return self.config

    def _url(self, suffix: str) -> str:
        config = self._require_config()
        repo = config.repo_path
        return f"{config.base_url}/api/v1/repos/{repo}{suffix}"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        token = self.config.token
        if token:
            # Sent as a header only; never logged or returned.
            headers["Authorization"] = f"token {token}"
        return headers

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            transport=self._transport,
            follow_redirects=True,
        )

    async def _get(
        self, url: str, *, params: dict[str, Any] | None = None, raw: bool = False
    ) -> Any:
        """Run one bounded GET and classify failures into domain errors."""
        async with self._client() as client:
            try:
                response = await client.get(url, params=params, headers=self._headers())
            except httpx.TimeoutException as exc:
                raise GiteaTimeoutError("Gitea request timed out") from exc
            except httpx.TransportError as exc:
                raise GiteaUnavailableError("Could not reach Gitea") from exc

        status = response.status_code
        if status in (401, 403):
            raise GiteaAuthError(
                "Gitea rejected the request (missing or insufficient token scope)"
            )
        if status == 404:
            raise GiteaNotFoundError("Gitea resource not found")
        if status >= 400:
            raise GiteaError(f"Gitea returned HTTP {status}")

        if raw:
            return response.text
        try:
            return response.json()
        except ValueError as exc:
            raise GiteaError("Gitea returned a malformed JSON response") from exc

    # ------------------------------------------------------------------
    # Read-only operations
    # ------------------------------------------------------------------
    async def recent_commits(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        branch: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return recent commits for the FIXED repository over a time range."""
        config = self._require_config()
        bounded_limit = max(1, min(int(limit or 10), config.max_commits))
        ref = validate_ref(branch) if branch else config.default_branch
        params: dict[str, Any] = {
            "sha": ref,
            "limit": bounded_limit,
        }
        if since is not None:
            params["since"] = format_time(since)
        if until is not None:
            params["until"] = format_time(until)

        payload = await self._get(self._url("/commits"), params=params)
        if not isinstance(payload, list):
            raise GiteaError("Gitea returned an unexpected commits payload")
        commits = [parse_commit(item) for item in payload if isinstance(item, dict)]
        return {
            "repository": config.repo_path,
            "branch": ref,
            "count": len(commits),
            "limit": bounded_limit,
            "commits": commits,
        }

    async def get_commit(self, sha: str) -> dict[str, Any]:
        """Return one commit's metadata for the FIXED repository."""
        config = self._require_config()
        ref = validate_ref(sha)
        payload = await self._get(self._url(f"/git/commits/{ref}"))
        if not isinstance(payload, dict):
            raise GiteaError("Gitea returned an unexpected commit payload")
        commit = parse_commit(payload)
        commit["repository"] = config.repo_path
        return commit

    async def get_commit_diff(self, sha: str) -> dict[str, Any]:
        """Return the bounded unified diff for one commit sha."""
        config = self._require_config()
        ref = validate_sha(sha)
        text = await self._get(self._url(f"/git/commits/{ref}.diff"), raw=True)
        diff = text or ""
        truncated = len(diff) > config.max_diff_chars
        if truncated:
            diff = diff[: config.max_diff_chars]
        return {
            "repository": config.repo_path,
            "sha": ref,
            "short_sha": _short_sha(ref),
            "files": extract_diff_files(diff),
            "diff": diff,
            "diff_chars": len(diff),
            "truncated": truncated,
        }


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
    "extract_diff_files",
    "format_time",
    "parse_commit",
    "validate_ref",
    "validate_sha",
]
