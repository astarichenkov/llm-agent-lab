"""Reusable "does this chunk cover an expected source?" logic.

The Day 22 evaluation dataset describes expected sources as either:

* a manual page (``source_type=manual``, ``source``, ``page``); or
* a Telegram conversation region (``source_type=telegram``, ``source``,
  ``message_ids`` — the stored chunk may cover only some of them, so the
  match is an *overlap*).

This module is shared by the Day 23 evaluation service and the similarity
score report. It is intentionally independent of the index so it can be tested
with plain objects.
"""
from __future__ import annotations

from typing import Any, Iterable

from app.schemas.day22 import ExpectedSource


def _message_id_set(metadata: dict[str, Any]) -> set[int]:
    values = metadata.get("message_ids")
    if values is None:
        values = metadata.get("message_id")
    if values is None:
        return set()
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    result: set[int] = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _chunk_pages(metadata: dict[str, Any]) -> set[int]:
    """Collect every page number a chunk can be said to cover."""
    pages: set[int] = set()
    page = metadata.get("page")
    if page is not None:
        try:
            pages.add(int(page))
        except (TypeError, ValueError):
            pass
    start = metadata.get("page_from")
    if start is not None:
        try:
            start_i = int(start)
            end_i = int(metadata["page_to"]) if metadata.get("page_to") is not None else start_i
        except (TypeError, ValueError):
            start_i = end_i = None  # type: ignore[assignment]
        if start_i is not None and end_i is not None:
            lo, hi = min(start_i, end_i), max(start_i, end_i)
            pages.update(range(lo, hi + 1))
    return pages


def matches_expected_source(
    metadata: dict[str, Any], expected: ExpectedSource
) -> bool:
    """Return True when *metadata* provably covers *expected*."""
    if metadata.get("source_type") != expected.source_type:
        return False
    if expected.source and metadata.get("source") != expected.source:
        return False

    if expected.source_type == "telegram":
        expected_ids = {int(i) for i in (expected.message_ids or [])}
        if not expected_ids:
            return True
        return bool(_message_id_set(metadata) & expected_ids)

    if expected.page is not None:
        return int(expected.page) in _chunk_pages(metadata)
    return True


def matched_expected_indices(
    candidates: Iterable[Any],
    expected_sources: list[ExpectedSource],
) -> set[int]:
    """Return indices of expected sources covered by at least one candidate."""
    if not expected_sources:
        return set()
    matched: set[int] = set()
    for candidate in candidates:
        metadata = dict(getattr(candidate, "metadata", None) or {})
        if not metadata and isinstance(candidate, dict):
            metadata = dict(candidate.get("metadata", {}) or {})
        # RetrievedChunk keeps source_type/source as first-class fields; fall
        # back to them when a hand-built candidate's metadata is incomplete.
        metadata.setdefault("source_type", getattr(candidate, "source_type", None))
        metadata.setdefault("source", getattr(candidate, "source", None))
        for index, expected in enumerate(expected_sources):
            if index in matched:
                continue
            if matches_expected_source(metadata or {}, expected):
                matched.add(index)
    return matched


def all_expected_sources_retrieved(
    candidates: Iterable[Any],
    expected_sources: list[ExpectedSource],
) -> bool:
    """True when EVERY expected source is covered by the candidate list.

    A question with no declared expected source is treated as not retrieved
    (there is nothing to verify).
    """
    if not expected_sources:
        return False
    matched = matched_expected_indices(candidates, expected_sources)
    return len(matched) == len(expected_sources)
