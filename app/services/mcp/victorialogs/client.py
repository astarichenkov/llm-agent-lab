"""Async HTTP client for the VictoriaLogs ``/select/logsql/query`` endpoint.

All HTTP concerns for Day 17 live here so the MCP tool function only performs
query construction + normalization:

* reads ``VICTORIA_LOGS_BASE_URL`` (never hardcoded);
* validates the URL lazily (a missing URL must not break application startup);
* issues a bounded request (``query`` / ``start`` / ``end`` / ``limit`` /
  ``timeout``);
* parses the **JSON Lines** response line by line (streaming, bounded);
* maps network/HTTP/parse failures to controlled, secret-free exceptions.

The client is also used by the automated tests with an injected
``httpx.MockTransport``; no test ever contacts a real VictoriaLogs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx

# Endpoint documented at:
# https://docs.victoriametrics.com/victorialogs/querying/#http-api
QUERY_PATH = "/select/logsql/query"

DEFAULT_SERVICE_FIELD = "service"


class VictoriaLogsError(Exception):
    """Base class for controlled VictoriaLogs failures (never leaks traces)."""

    kind = "error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class VictoriaLogsConfigError(VictoriaLogsError):
    """``VICTORIA_LOGS_BASE_URL`` is missing or malformed."""

    kind = "config"


class VictoriaLogsTimeoutError(VictoriaLogsError):
    """The request exceeded the HTTP timeout."""

    kind = "timeout"


class VictoriaLogsNetworkError(VictoriaLogsError):
    """DNS / TCP / TLS failure reaching VictoriaLogs."""

    kind = "network"


class VictoriaLogsHTTPError(VictoriaLogsError):
    """VictoriaLogs returned an HTTP 4xx/5xx status."""

    kind = "http"

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        self.detail = detail
        message = f"VictoriaLogs returned HTTP {status_code}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class VictoriaLogsConfig:
    """Resolved connection settings (URL + query behaviour)."""

    base_url: str
    service_field: str = DEFAULT_SERVICE_FIELD
    verify_ssl: bool = True
    # When set, this CA bundle path is passed to httpx as the ``verify`` value
    # (preferred over ``verify_ssl=False`` for internal corporate CAs).
    ca_bundle: str = ""
    timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "VictoriaLogsConfig":
        """Build the config from environment variables.

        Only ``VICTORIA_LOGS_BASE_URL`` is required; it is validated by
        :class:`VictoriaLogsClient` at call time, not at import/startup.
        """
        source = os.environ if env is None else env
        raw_verify = source.get("VICTORIA_LOGS_VERIFY_SSL", "true").strip().lower()
        verify = raw_verify not in ("0", "false", "no", "off")
        try:
            timeout = float(source.get("VICTORIA_LOGS_TIMEOUT_SECONDS", "5"))
        except (TypeError, ValueError):
            timeout = 5.0
        return cls(
            base_url=(source.get("VICTORIA_LOGS_BASE_URL") or "").strip(),
            service_field=(
                source.get("VICTORIA_LOGS_SERVICE_FIELD") or DEFAULT_SERVICE_FIELD
            ).strip()
            or DEFAULT_SERVICE_FIELD,
            verify_ssl=verify,
            ca_bundle=(source.get("VICTORIA_LOGS_CA_BUNDLE") or "").strip(),
            timeout_seconds=timeout,
        )

    def query_url(self) -> str:
        """Return the full endpoint URL regardless of trailing slashes."""
        return self.base_url.rstrip("/") + QUERY_PATH

    def verify_option(self) -> bool | str:
        """Value passed to httpx ``verify`` (CA bundle path wins over bool)."""
        return self.ca_bundle or self.verify_ssl


def format_time(value: datetime) -> str:
    """Format a datetime as an RFC3339 UTC timestamp VictoriaLogs accepts."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class VictoriaLogsClient:
    """Bounded, streaming client around VictoriaLogs ``/select/logsql/query``."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        service_field: str | None = None,
        verify_ssl: bool | None = None,
        ca_bundle: str | None = None,
        http_timeout_seconds: float | None = None,
        query_timeout_seconds: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        config = VictoriaLogsConfig.from_env(env)
        self.config = VictoriaLogsConfig(
            base_url=base_url if base_url is not None else config.base_url,
            service_field=(
                service_field if service_field is not None else config.service_field
            )
            or DEFAULT_SERVICE_FIELD,
            verify_ssl=config.verify_ssl if verify_ssl is None else verify_ssl,
            ca_bundle=config.ca_bundle if ca_bundle is None else ca_bundle,
            timeout_seconds=(
                config.timeout_seconds
                if http_timeout_seconds is None
                else http_timeout_seconds
            ),
        )
        # Server-side LogsQL timeout (seconds), sent as the ``timeout`` arg.
        self.query_timeout_seconds = int(query_timeout_seconds or 5)
        self._transport = transport

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "VictoriaLogsClient":
        return cls(env=env)

    def _require_base_url(self) -> str:
        if not self.config.base_url:
            raise VictoriaLogsConfigError(
                "VICTORIA_LOGS_BASE_URL is not configured"
            )
        if not self.config.base_url.startswith(("http://", "https://")):
            raise VictoriaLogsConfigError(
                "VICTORIA_LOGS_BASE_URL must start with http:// or https://"
            )
        return self.config.base_url

    async def search(
        self,
        *,
        query: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Run one bounded query.

        Returns ``(rows, malformed_line_count)`` where ``rows`` is a list of
        parsed JSON objects. Empty lines are ignored; a malformed line is
        skipped and counted instead of failing the whole request.
        """
        base_url = self._require_base_url()
        url = base_url.rstrip("/") + QUERY_PATH
        params = {
            "query": query,
            "start": format_time(start),
            "end": format_time(end),
            "limit": str(limit),
            "timeout": f"{self.query_timeout_seconds}s",
        }

        rows: list[dict[str, Any]] = []
        malformed = 0

        try:
            async with httpx.AsyncClient(
                verify=self.config.verify_option(),
                timeout=self.config.timeout_seconds,
                transport=self._transport,
            ) as client:
                async with client.stream("GET", url, params=params) as response:
                    if response.status_code >= 400:
                        detail = await _read_error_detail(response)
                        raise VictoriaLogsHTTPError(response.status_code, detail)

                    async for line in response.aiter_lines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            parsed = json.loads(stripped)
                        except json.JSONDecodeError:
                            malformed += 1
                            continue
                        if not isinstance(parsed, dict):
                            malformed += 1
                            continue
                        rows.append(parsed)
                        # Defensive client-side cap in addition to the
                        # server-side ``limit`` parameter.
                        if len(rows) >= limit:
                            break
        except httpx.TimeoutException as exc:
            raise VictoriaLogsTimeoutError(
                "VictoriaLogs request timed out"
            ) from exc
        except httpx.TransportError as exc:
            raise VictoriaLogsNetworkError(
                f"Could not reach VictoriaLogs: {exc.__class__.__name__}"
            ) from exc

        return rows, malformed


async def _read_error_detail(response: httpx.Response) -> str:
    """Read a short, bounded, single-line error body (no secrets)."""
    try:
        raw = await response.aread()
    except Exception:  # pragma: no cover - defensive
        return ""
    text = raw.decode("utf-8", errors="replace")
    return " ".join(text.split())[:300]


def normalize_logs(
    rows: Iterable[dict[str, Any]],
    *,
    sanitize,
) -> list[dict[str, Any]]:
    """Map raw VictoriaLogs rows to the normalized result shape.

    ``_time`` -> ``timestamp`` and ``_msg`` -> ``message`` when present; every
    other field is preserved under ``fields``. ``sanitize`` is injected (the
    sanitizer's ``sanitize_value``) so this helper stays free of policy.
    """
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        timestamp = raw.get("_time") or raw.get("@timestamp") or raw.get("time")
        message = raw.get("_msg")
        fields = {
            key: value for key, value in raw.items() if key not in ("_time", "_msg")
        }
        normalized.append(
            {
                "timestamp": None if timestamp is None else str(timestamp),
                "message": None if message is None else str(message),
                "fields": sanitize(fields),
            }
        )
    # Sanitize the message too (it is the most likely place for a leaked secret).
    for item in normalized:
        if item["message"] is not None:
            item["message"] = sanitize(item["message"])
    return normalized
