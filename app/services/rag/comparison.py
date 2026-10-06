"""Day 21 chunking-strategy comparison.

This module powers the ``compare-chunking`` CLI command. It retrieves the same
query against two *already built* indexes (structural and fixed) and exposes
objective, comparable numbers about their retrieval results. It never rebuilds
or modifies an index and it never declares a "winner": cosine similarity alone
is not a quality metric (see the README).

The comparison is intentionally read-only and reuses the existing
:class:`~app.services.rag.index.sqlite_index.SqliteVectorIndex` plus the
embedding provider factory, so behaviour stays identical to ``search``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.services.rag.embeddings.base import EmbeddingError, EmbeddingProvider
from app.services.rag.index.sqlite_index import SqliteVectorIndex
from app.services.rag.models import SearchHit
from app.services.rag.service import build_query_provider


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------
class ComparisonError(Exception):
    """Base class for comparison failures that the CLI prints politely."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class IndexNotFoundError(ComparisonError):
    """Raised when one of the two index files is missing."""

    def __init__(self, label: str, path: str | Path) -> None:
        self.label = label
        self.path = str(path)
        super().__init__(f"{label} index not found: {self.path}")


class EmptyIndexError(ComparisonError):
    """Raised when an index exists but contains no chunks."""

    def __init__(self, label: str, path: str | Path) -> None:
        self.label = label
        self.path = str(path)
        super().__init__(f"{label} index is empty: {self.path}")


class IndexCompatibilityError(ComparisonError):
    """Raised when the two indexes were built with different embedders."""

    def __init__(self, structural: "IndexInfo", fixed: "IndexInfo") -> None:
        self.structural = structural
        self.fixed = fixed
        super().__init__("Indexes were created with different embedding models.")


# ----------------------------------------------------------------------
# index summary
# ----------------------------------------------------------------------
@dataclass
class IndexInfo:
    """Read-only summary of one persisted index."""

    path: str
    chunks: int = 0
    strategy: str = ""
    provider: str = ""
    model: str = ""
    dimension: int = 0
    avg_chars: float = 0.0
    min_chars: int = 0
    max_chars: int = 0
    chunks_by_source_type: dict[str, int] = field(default_factory=dict)


def read_index_info(path: str | Path) -> IndexInfo:
    """Read chunk counts, chunk sizes and metadata from one index file."""
    index = SqliteVectorIndex(path)
    try:
        stats = index.stats()
    finally:
        index.close()
    return IndexInfo(
        path=str(path),
        chunks=int(stats.get("chunks", 0) or 0),
        strategy=str(stats.get("chunking", "") or ""),
        provider=str(stats.get("embedding_provider", "") or ""),
        model=str(stats.get("embedding_model", "") or ""),
        dimension=int(stats.get("embedding_dimension", 0) or 0),
        avg_chars=float(stats.get("avg_chunk_size", 0) or 0.0),
        min_chars=int(stats.get("min_chunk_size", 0) or 0),
        max_chars=int(stats.get("max_chunk_size", 0) or 0),
        chunks_by_source_type=dict(stats.get("chunks_by_source_type", {}) or {}),
    )


def ensure_indexes_compatible(structural: IndexInfo, fixed: IndexInfo) -> None:
    """Fail loudly when the two indexes cannot be compared meaningfully.

    Provider, model and (when both are known) vector dimension must match;
    otherwise the query embedding would not be valid for both stores.
    """
    if (
        structural.provider != fixed.provider
        or structural.model != fixed.model
        or (
            structural.dimension
            and fixed.dimension
            and structural.dimension != fixed.dimension
        )
    ):
        raise IndexCompatibilityError(structural, fixed)


# ----------------------------------------------------------------------
# region overlap
# ----------------------------------------------------------------------
def _page_range(metadata: dict) -> tuple[int, int] | None:
    """Return an inclusive ``(from, to)`` page range, if the chunk has one."""
    if metadata.get("page") is not None:
        try:
            page = int(metadata["page"])
        except (TypeError, ValueError):
            page = None
        if page is not None:
            return (page, page)
    start = metadata.get("page_from")
    if start is None:
        return None
    try:
        start_i = int(start)
        end_i = int(metadata["page_to"]) if metadata.get("page_to") is not None else start_i
    except (TypeError, ValueError):
        return None
    return (min(start_i, end_i), max(start_i, end_i))


def _message_id_set(metadata: dict) -> set[int]:
    """Collect ``message_ids`` (or a single ``message_id``) as integers."""
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


def same_source_region(a: SearchHit, b: SearchHit) -> bool | None:
    """Decide whether two hits cover the same source region.

    Returns:

    * ``True``  — the regions provably overlap;
    * ``False`` — the regions provably do not overlap (different source, or
      non-overlapping page/message ranges);
    * ``None``  — the metadata is not precise enough to tell.

    Manuals are compared by inclusive page range (``page`` or
    ``page_from``/``page_to``). Telegram chunks are compared by the
    intersection of their ``message_ids``. A different ``source``/``source_type``
    is always a definite non-overlap. Chunk ids are intentionally never used:
    fixed and structural indexes assign different ids to the same source text.
    """
    if a.metadata.get("source_type") != b.metadata.get("source_type"):
        return False
    if a.metadata.get("source") != b.metadata.get("source"):
        return False

    if a.metadata.get("source_type") == "telegram":
        ids_a = _message_id_set(a.metadata)
        ids_b = _message_id_set(b.metadata)
        if ids_a and ids_b:
            return bool(ids_a & ids_b)
        return None

    range_a = _page_range(a.metadata)
    range_b = _page_range(b.metadata)
    if range_a and range_b:
        return not (range_a[1] < range_b[0] or range_b[1] < range_a[0])
    return None


@dataclass
class RegionOverlap:
    """How many hits of one result list also appear as a region in another."""

    matched: int = 0
    compared: int = 0
    available: bool = False

    @property
    def percent(self) -> float | None:
        if not self.available or not self.compared:
            return None
        return round(100.0 * self.matched / self.compared, 1)

    @property
    def label(self) -> str:
        if not self.available:
            return "unavailable"
        return f"{self.matched} / {self.compared}"


def compute_region_overlap(
    primary_hits: list[SearchHit],
    other_hits: list[SearchHit],
) -> RegionOverlap:
    """Count ``primary`` hits whose source region is covered by ``other``.

    A hit counts as matched when at least one hit in ``other_hits`` shares its
    source *and* provably overlaps its page range / Telegram message ids. If any
    hit cannot be decided from metadata, the whole metric is reported as
    ``unavailable`` rather than guessed.
    """
    compared = len(primary_hits)
    if not compared:
        return RegionOverlap(matched=0, compared=0, available=False)

    matched = 0
    unknown = False
    for primary in primary_hits:
        candidates = [other for other in other_hits if _same_source(primary, other)]
        if not candidates:
            continue
        statuses = [same_source_region(primary, other) for other in candidates]
        if any(status is True for status in statuses):
            matched += 1
        elif all(status is False for status in statuses):
            continue
        else:
            unknown = True
    return RegionOverlap(matched=matched, compared=compared, available=not unknown)


def _same_source(a: SearchHit, b: SearchHit) -> bool:
    return (
        a.metadata.get("source_type") == b.metadata.get("source_type")
        and a.metadata.get("source") == b.metadata.get("source")
    )


# ----------------------------------------------------------------------
# comparison result
# ----------------------------------------------------------------------
def _average(scores: list[float]) -> float | None:
    return round(sum(scores) / len(scores), 4) if scores else None


@dataclass
class ChunkingComparison:
    """Objective side-by-side view of one query against two indexes."""

    query: str
    top_k: int
    structural: IndexInfo
    fixed: IndexInfo
    structural_hits: list[SearchHit]
    fixed_hits: list[SearchHit]
    overlap: RegionOverlap
    fixed_overlap: RegionOverlap

    @property
    def structural_top1(self) -> float | None:
        return self.structural_hits[0].score if self.structural_hits else None

    @property
    def fixed_top1(self) -> float | None:
        return self.fixed_hits[0].score if self.fixed_hits else None

    @property
    def structural_avg(self) -> float | None:
        return _average([hit.score for hit in self.structural_hits])

    @property
    def fixed_avg(self) -> float | None:
        return _average([hit.score for hit in self.fixed_hits])


# ----------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------
def compare_chunking(
    query: str,
    *,
    settings: Settings,
    structural_path: str | Path | None = None,
    fixed_path: str | Path | None = None,
    top_k: int = 5,
    embedding_provider: EmbeddingProvider | None = None,
) -> ChunkingComparison:
    """Run one query against the structural and fixed indexes.

    The same query embedding is computed **once** (when both indexes share the
    same provider/model) and used for both searches. Raises
    :class:`ComparisonError` subclasses for missing/empty/incompatible indexes
    and :class:`~app.services.rag.embeddings.base.EmbeddingError` when the
    query embedding cannot be produced.
    """
    structural_path = Path(structural_path or settings.rag_index_path)
    fixed_path = Path(fixed_path or settings.rag_fixed_index_path)

    if top_k <= 0:
        raise ComparisonError("--top-k must be a positive integer")
    if not structural_path.exists():
        raise IndexNotFoundError("Structural", structural_path)
    if not fixed_path.exists():
        raise IndexNotFoundError("Fixed", fixed_path)

    structural = read_index_info(structural_path)
    fixed = read_index_info(fixed_path)
    if structural.chunks == 0:
        raise EmptyIndexError("Structural", structural_path)
    if fixed.chunks == 0:
        raise EmptyIndexError("Fixed", fixed_path)

    ensure_indexes_compatible(structural, fixed)

    structural_index = SqliteVectorIndex(structural_path)
    fixed_index = SqliteVectorIndex(fixed_path)
    try:
        meta = structural_index.get_meta()
        provider = build_query_provider(settings, meta, embedding_provider)
        try:
            vector = provider.embed_query(query)
        except EmbeddingError as exc:
            detail = f"provider={provider.name}, model={provider.model}"
            if provider.name == "ollama":
                endpoint = getattr(provider, "base_url", settings.rag_ollama_base_url)
                detail += f", endpoint={endpoint}"
            raise EmbeddingError(f"{exc.message} ({detail})") from exc
        except Exception as exc:  # noqa: BLE001 - surface a friendly message
            raise EmbeddingError(
                f"Cannot compute query embedding (provider={provider.name}, "
                f"model={provider.model}): {exc}"
            ) from exc

        structural_hits = structural_index.search(vector, top_k=top_k)
        fixed_hits = fixed_index.search(vector, top_k=top_k)
    finally:
        structural_index.close()
        fixed_index.close()

    return ChunkingComparison(
        query=query,
        top_k=top_k,
        structural=structural,
        fixed=fixed,
        structural_hits=structural_hits,
        fixed_hits=fixed_hits,
        overlap=compute_region_overlap(structural_hits, fixed_hits),
        fixed_overlap=compute_region_overlap(fixed_hits, structural_hits),
    )


__all__ = [
    "ChunkingComparison",
    "ComparisonError",
    "EmptyIndexError",
    "IndexCompatibilityError",
    "IndexInfo",
    "IndexNotFoundError",
    "RegionOverlap",
    "compare_chunking",
    "compute_region_overlap",
    "ensure_indexes_compatible",
    "read_index_info",
    "same_source_region",
]
