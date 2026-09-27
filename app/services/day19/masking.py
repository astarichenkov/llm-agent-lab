"""Day 19 — opt-in output masking ("Маскировать данные").

This is an ADDITIONAL privacy layer on top of the Day 17 secret sanitizer.
When the user checks *Маскировать данные*, the pipeline masks identifying
values before they reach the LLM, the final answer, the trace and the saved
artifacts:

* URLs (``https://...``), host:port pairs, IPv4 addresses and e-mails;
* values of identifier fields (``service``, ``container``, ``host``, ``pod``,
  ``namespace``, ``node``, ``cluster``, ``url``, ``endpoint`` ...);
* the literal service name supplied in the request and any service/container
  names discovered in the structured ``fields`` of the searched rows.

The masker is deterministic and one-way: masked values are replaced with
``[MASKED]`` / ``[URL]`` / ``[IP]`` / ``[EMAIL]`` and cannot be recovered from
the output. It is only active when explicitly requested; the default pipeline
behaviour is unchanged.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

MASKED = "[MASKED]"
URL_PLACEHOLDER = "[URL]"
IP_PLACEHOLDER = "[IP]"
EMAIL_PLACEHOLDER = "[EMAIL]"

# Fields whose value is treated as an identifier and fully masked.
IDENTIFIER_KEYS: frozenset[str] = frozenset(
    {
        "service",
        "service_name",
        "service_instance",
        "container",
        "container_name",
        "container_id",
        "host",
        "hostname",
        "host_name",
        "pod",
        "pod_name",
        "namespace",
        "node",
        "node_name",
        "cluster",
        "cluster_name",
        "deployment",
        "instance",
        "instance_id",
        "url",
        "uri",
        "base_url",
        "endpoint",
        "addr",
        "address",
        "app",
        "app_name",
        "application",
    }
)

_URL_RE = re.compile(
    r"(?i)\b(?:https?|ftp|ws|wss)://[^\s\"'<>()\[\]]*[^\s\"'<>()\[\].,;:!?]"
)
# ``host:port`` — requires either a dotted FQDN/IP-ish host or a host that
# starts with a letter. This deliberately avoids matching clock times such as
# ``10:00`` inside log messages/timestamps.
_HOST_PORT_RE = re.compile(
    r"\b(?:"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"|[A-Za-z][A-Za-z0-9-]*"
    r"):\d{2,5}\b"
)
_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Minimum length for a literal to be replaced, to avoid masking tiny words.
_MIN_LITERAL_LENGTH = 3


def _normalize_key(key: Any) -> str:
    return str(key or "").strip().lower().replace("-", "_")


class DataMasker:
    """Deterministic masking of identifiers in strings and JSON-like values."""

    def __init__(self, literals: Iterable[str] | None = None) -> None:
        self._literals: list[str] = []
        self._literal_re: re.Pattern[str] | None = None
        self.add_literals(literals or [])

    # ------------------------------------------------------------------
    # Literals (exact service / container names)
    # ------------------------------------------------------------------
    def add_literals(self, values: Iterable[Any]) -> None:
        changed = False
        for value in values:
            if not isinstance(value, str):
                continue
            text = value.strip()
            if len(text) < _MIN_LITERAL_LENGTH:
                continue
            if text not in self._literals:
                self._literals.append(text)
                changed = True
        if changed:
            self._literals.sort(key=len, reverse=True)
            joined = "|".join(re.escape(item) for item in self._literals)
            self._literal_re = re.compile(
                rf"(?<![A-Za-z0-9_])(?:{joined})(?![A-Za-z0-9_])"
            )

    def add_log_literals(self, logs: Iterable[Any]) -> None:
        """Collect service/container names found in the rows' ``fields``."""
        values: list[Any] = []
        for row in logs or []:
            fields = row.get("fields") if isinstance(row, dict) else None
            if not isinstance(fields, dict):
                continue
            for key, value in fields.items():
                if _normalize_key(key) in IDENTIFIER_KEYS and isinstance(value, str):
                    values.append(value)
        self.add_literals(values)

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    def mask_text(self, text: Any) -> Any:
        if not isinstance(text, str) or not text:
            return text
        result = _URL_RE.sub(URL_PLACEHOLDER, text)
        result = _HOST_PORT_RE.sub(MASKED, result)
        result = _IPV4_RE.sub(IP_PLACEHOLDER, result)
        result = _EMAIL_RE.sub(EMAIL_PLACEHOLDER, result)
        if self._literal_re is not None:
            result = self._literal_re.sub(MASKED, result)
        return result

    def mask_value(self, value: Any, *, key: Any = None) -> Any:
        if key is not None and _normalize_key(key) in IDENTIFIER_KEYS:
            return MASKED if value is not None else None
        if isinstance(value, str):
            return self.mask_text(value)
        if isinstance(value, dict):
            return {k: self.mask_value(v, key=k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.mask_value(item) for item in value]
        if isinstance(value, tuple):
            return [self.mask_value(item) for item in value]
        return value

    def mask_logs(self, logs: Iterable[Any]) -> list[Any]:
        masked: list[Any] = []
        for row in logs or []:
            if isinstance(row, dict):
                masked.append(self.mask_value(row))
            else:
                masked.append(row)
        return masked


__all__ = [
    "DataMasker",
    "EMAIL_PLACEHOLDER",
    "IDENTIFIER_KEYS",
    "IP_PLACEHOLDER",
    "MASKED",
    "URL_PLACEHOLDER",
]
