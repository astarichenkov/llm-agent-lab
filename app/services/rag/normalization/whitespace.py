"""Whitespace normalization shared by the citation validator and the
Telegram source-link resolver.

Kept in the neutral ``normalization`` package so neither component has to
import the other (which would create a circular import between
``grounding`` and ``sources``).
"""
from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace and line endings, then trim.

    Windows/Unix line endings, multiple spaces and leading/trailing
    whitespace are normalized. Case and characters are NOT changed, so a
    paraphrase still fails validation.
    """
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return _WHITESPACE.sub(" ", value).strip()
