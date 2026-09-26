"""Safe LogsQL construction from structured ``search_logs`` parameters.

The LLM never supplies raw LogsQL. It supplies structured values
(``service`` / ``level`` / ``text_contains``) and this module builds the query
with validation and escaping so a value can never break out of its filter.

LogsQL field names are identifiers, so they are validated against a strict
pattern (configurable ``service`` field included). Values are always emitted
as double-quoted phrases with backslashes and quotes escaped.
"""
from __future__ import annotations

import re

# LogsQL field names: start with a letter / underscore / '@', then letters,
# digits, underscore, dot or hyphen. This covers the real field names seen in
# the VictoriaLogs deployment (e.g. ``service``, ``caller_class_name``,
# ``@timestamp``) while rejecting anything that could inject LogsQL syntax.
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_@][A-Za-z0-9_.@-]*$")

# The field that stores the log level. It is a documented, conventional field
# and is verified against real data during manual integration verification.
LEVEL_FIELD = "level"


class QueryBuildError(ValueError):
    """Raised when a LogsQL fragment cannot be built safely."""


def validate_field_name(name: str) -> str:
    """Return ``name`` if it is a safe LogsQL field identifier, else raise."""
    candidate = (name or "").strip()
    if not _FIELD_NAME_RE.match(candidate):
        raise QueryBuildError(f"invalid LogsQL field name: {name!r}")
    return candidate


def quote_value(value: str) -> str:
    """Emit ``value`` as a double-quoted LogsQL phrase.

    Backslashes and double quotes are escaped and CR/LF are collapsed, so the
    value cannot terminate the phrase or inject additional filters.
    """
    cleaned = str(value).replace("\r", " ").replace("\n", " ")
    escaped = cleaned.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_logsql(
    *,
    service: str,
    service_field: str = "service",
    level: str | None = None,
    text_contains: str | None = None,
) -> str:
    """Build a safe LogsQL query from structured filters.

    Example::

        service:"orders-service" AND level:"ERROR" AND "timeout"
    """
    service = (service or "").strip()
    if not service:
        raise QueryBuildError("service must not be empty")

    field = validate_field_name(service_field)
    parts = [f"{field}:{quote_value(service)}"]

    if level is not None and level.strip():
        parts.append(f"{LEVEL_FIELD}:{quote_value(level.strip())}")

    if text_contains is not None and text_contains.strip():
        # A bare quoted phrase searches the log message body.
        parts.append(quote_value(text_contains.strip()))

    return " AND ".join(parts)
