"""Basic secret sanitization for VictoriaLogs content sent to the LLM.

This is intentionally a *simple* safety net, not a DLP system. It masks the
most common ways a credential leaks into a log line:

* ``Authorization: Bearer <token>`` / a bare ``Bearer <token>``;
* ``password=...`` / ``passwd: ...``;
* ``token=...`` / ``access_token=...`` / ``refresh_token=...``;
* ``api_key=...`` / ``apikey=...`` / ``secret=...`` / ``client_secret=...``;
* ``Cookie: ...`` / ``Set-Cookie: ...``.

Matched secret values are replaced with ``[REDACTED]``. The sanitizer is
applied to the normalized ``message`` and to every string field value before
the tool result is returned to the agent/LLM.
"""
from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

# Dict keys whose value is treated as sensitive regardless of its content.
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)^(authorization|password|passwd|pwd|token|access[_-]?token|"
    r"refresh[_-]?token|id[_-]?token|api[_-]?key|apikey|secret|"
    r"client[_-]?secret|cookie|set[_-]?cookie)$"
)

# Order matters: the Authorization rule must run before the bare Bearer rule.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization: Bearer <value>  /  authorization=<value>
    (
        re.compile(r"(?i)\b(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"),
        rf"\1{REDACTED}",
    ),
    # A bare bearer token anywhere in the text.
    (
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{4,}"),
        f"Bearer {REDACTED}",
    ),
    # password / passwd / pwd
    (
        re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*[^\s,;]+"),
        rf"\1={REDACTED}",
    ),
    # token family
    (
        re.compile(
            r"(?i)\b(access[_-]?token|refresh[_-]?token|id[_-]?token|token)"
            r"\s*[:=]\s*[^\s,;]+"
        ),
        rf"\1={REDACTED}",
    ),
    # api key / secret family
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|client[_-]?secret|secret)"
            r"\s*[:=]\s*[^\s,;]+"
        ),
        rf"\1={REDACTED}",
    ),
    # Cookie / Set-Cookie headers (value may contain spaces, so mask to EOL).
    (
        re.compile(r"(?i)\b(set-cookie|cookie)\s*[:=]\s*[^\r\n]+"),
        rf"\1: {REDACTED}",
    ),
)


def sanitize_text(text: str) -> str:
    """Mask obvious secrets in a single string."""
    if not text:
        return text
    result = text
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings inside a JSON-like value."""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        cleaned: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE_KEY_RE.match(key.strip()):
                cleaned[key] = REDACTED
            else:
                cleaned[key] = sanitize_value(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value
