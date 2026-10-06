"""Day 21 — chunking-strategy comparison (`compare-chunking`) tests.

Everything here is deterministic and offline: indexes are tiny synthetic
SQLite files and embeddings use the dependency-free hashing provider. No
Ollama, no real Telegram export and no manual PDF is touched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from app.config import Settings
from app.services.rag import cli
from app.services.rag.comparison import (
    ChunkingComparison,
    EmptyIndexError,
    IndexCompatibilityError,
    IndexInfo,
    IndexNotFoundError,
    RegionOverlap,
    compare_chunking,
    compute_region_overlap,
    read_index_info,
    same_source_region,
)
from app.services.rag.embeddings import HashingEmbeddingProvider
from app.services.rag.index import SqliteVectorIndex
from app.services.rag.models import Chunk, SearchHit


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _settings(tmp_path: Path) -> Settings:
    return Settings(
        rag_index_path=str(tmp_path / "structural.sqlite3"),
        rag_fixed_index_path=str(tmp_path / "fixed.sqlite3"),
        rag_embedding_provider="hashing",
        rag_embedding_dimension=64,
    )


def _build_index(
    path: Path,
    records: list[tuple[str, str, dict]],
    *,
    provider: HashingEmbeddingProvider | None = None,
    provider_name: str = "hashing",
    model: str = "hashing-ngrams",
    chunking: str = "structural",
) -> None:
    provider = provider or HashingEmbeddingProvider(dimension=64)
    chunks = [
        Chunk(chunk_id=cid, text=text, metadata=meta, doc_id=meta.get("source", ""))
        for cid, text, meta in records
    ]
    index = SqliteVectorIndex(path)
    index.add_chunks(chunks, provider.embed([c.text for c in chunks]))
    index.set_meta(
        {
            "embedding_provider": provider_name,
            "embedding_model": model,
            "embedding_dimension": provider.dimension,
            "chunking": chunking,
        }
    )
    index.close()


def _hit(cid: str, metadata: dict, score: float = 0.5) -> SearchHit:
    return SearchHit(score=score, chunk_id=cid, text=f"text-{cid}", metadata=metadata, doc_id=cid)


def _manual(source: str = "m.pdf") -> dict:
    return {"source_type": "manual", "source": source}


def _telegram(source: str = "result.json") -> dict:
    return {"source_type": "telegram", "source": source}


# ----------------------------------------------------------------------
# 1. parsing compare-chunking
# ----------------------------------------------------------------------
def test_parse_compare_chunking() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["compare-chunking", "давление в шинах", "--top-k", "7"])
    assert args.command == "compare-chunking"
    assert args.query == "давление в шинах"
    assert args.top_k == 7
    assert args.full is False
    assert args.structural_index is None
    assert args.fixed_index is None
    assert args.queries_file is None
    assert args.func is cli.cmd_compare_chunking


# ----------------------------------------------------------------------
# 2. default paths come from central settings
# ----------------------------------------------------------------------
def test_default_paths_are_central(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    assert settings.rag_index_path == "data/day21/rag_index.sqlite3"
    assert settings.rag_fixed_index_path == "data/day21/rag_index_fixed.sqlite3"

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    structural, fixed = cli._compare_paths(
        argparse.Namespace(structural_index=None, fixed_index=None)
    )
    assert structural == settings.rag_index_path
    assert fixed == settings.rag_fixed_index_path


# ----------------------------------------------------------------------
# 3. custom index paths
# ----------------------------------------------------------------------
def test_parse_custom_index_paths_and_full() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "compare-chunking",
            "q",
            "--structural-index",
            "a.sqlite3",
            "--fixed-index",
            "b.sqlite3",
            "--full",
        ]
    )
    assert args.structural_index == "a.sqlite3"
    assert args.fixed_index == "b.sqlite3"
    assert args.full is True


def test_compare_chunking_uses_custom_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = HashingEmbeddingProvider(dimension=64)
    structural = tmp_path / "custom_structural.sqlite3"
    fixed = tmp_path / "custom_fixed.sqlite3"
    _build_index(structural, [("s1", "давление в шинах", _manual())], provider=provider)
    _build_index(
        fixed,
        [("f1", "давление в шинах", _manual())],
        provider=provider,
        chunking="fixed",
    )
    result = compare_chunking(
        "давление",
        settings=settings,
        structural_path=structural,
        fixed_path=fixed,
        top_k=1,
        embedding_provider=provider,
    )
    assert result.structural.chunks == 1
    assert result.fixed.chunks == 1


# ----------------------------------------------------------------------
# 4. top-K
# ----------------------------------------------------------------------
def test_compare_chunking_respects_top_k(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = HashingEmbeddingProvider(dimension=64)
    structural = tmp_path / "s.sqlite3"
    fixed = tmp_path / "f.sqlite3"
    _build_index(
        structural,
        [(f"s{i}", f"шины давление {i}", _manual()) for i in range(5)],
        provider=provider,
    )
    _build_index(
        fixed,
        [(f"f{i}", f"шины давление {i}", _manual()) for i in range(5)],
        provider=provider,
        chunking="fixed",
    )
    result = compare_chunking(
        "давление",
        settings=settings,
        structural_path=structural,
        fixed_path=fixed,
        top_k=2,
        embedding_provider=provider,
    )
    assert len(result.structural_hits) == 2
    assert len(result.fixed_hits) == 2


# ----------------------------------------------------------------------
# 5. missing structural index
# ----------------------------------------------------------------------
def test_missing_structural_index(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fixed = tmp_path / "fixed.sqlite3"
    _build_index(fixed, [("f1", "x", _manual())], chunking="fixed")
    with pytest.raises(IndexNotFoundError) as excinfo:
        compare_chunking("q", settings=settings, structural_path=tmp_path / "nope.sqlite3", fixed_path=fixed)
    assert excinfo.value.label == "Structural"
    assert "nope.sqlite3" in excinfo.value.path


# ----------------------------------------------------------------------
# 6. missing fixed index
# ----------------------------------------------------------------------
def test_missing_fixed_index(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    structural = tmp_path / "structural.sqlite3"
    _build_index(structural, [("s1", "x", _manual())])
    with pytest.raises(IndexNotFoundError) as excinfo:
        compare_chunking("q", settings=settings, structural_path=structural, fixed_path=tmp_path / "nope.sqlite3")
    assert excinfo.value.label == "Fixed"


def test_empty_index(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    structural = tmp_path / "structural.sqlite3"
    fixed = tmp_path / "fixed.sqlite3"
    SqliteVectorIndex(structural).close()
    _build_index(fixed, [("f1", "x", _manual())], chunking="fixed")
    with pytest.raises(EmptyIndexError):
        compare_chunking("q", settings=settings, structural_path=structural, fixed_path=fixed)


# ----------------------------------------------------------------------
# 7. incompatible embedding models
# ----------------------------------------------------------------------
def test_incompatible_embedding_models(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = HashingEmbeddingProvider(dimension=64)
    structural = tmp_path / "structural.sqlite3"
    fixed = tmp_path / "fixed.sqlite3"
    _build_index(structural, [("s1", "x", _manual())], provider=provider, model="bge-m3")
    _build_index(
        fixed,
        [("f1", "x", _manual())],
        provider=provider,
        model="other-model",
        chunking="fixed",
    )
    with pytest.raises(IndexCompatibilityError) as excinfo:
        compare_chunking(
            "q",
            settings=settings,
            structural_path=structural,
            fixed_path=fixed,
            embedding_provider=provider,
        )
    assert excinfo.value.structural.model == "bge-m3"
    assert excinfo.value.fixed.model == "other-model"


def test_incompatible_provider(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = HashingEmbeddingProvider(dimension=64)
    structural = tmp_path / "structural.sqlite3"
    fixed = tmp_path / "fixed.sqlite3"
    _build_index(structural, [("s1", "x", _manual())], provider=provider, provider_name="hashing")
    _build_index(
        fixed,
        [("f1", "x", _manual())],
        provider=provider,
        provider_name="ollama",
        chunking="fixed",
    )
    with pytest.raises(IndexCompatibilityError):
        compare_chunking(
            "q",
            settings=settings,
            structural_path=structural,
            fixed_path=fixed,
            embedding_provider=provider,
        )


# ----------------------------------------------------------------------
# 8. manual source-region overlap
# ----------------------------------------------------------------------
def test_manual_page_range_overlap() -> None:
    structural = [
        _hit("s1", {**_manual(), "page_from": 10, "page_to": 12}),
        _hit("s2", {**_manual(), "page": 50}),
    ]
    fixed = [
        _hit("f1", {**_manual(), "page": 12}),
        _hit("f2", {**_manual(), "page_from": 99, "page_to": 100}),
    ]
    overlap = compute_region_overlap(structural, fixed)
    assert overlap.available is True
    assert overlap.compared == 2
    assert overlap.matched == 1  # s1 overlaps f1; s2 does not
    assert overlap.percent == 50.0
    assert overlap.label == "1 / 2"


def test_manual_different_source_is_not_overlap() -> None:
    a = _hit("s1", {**_manual("a.pdf"), "page": 5})
    b = _hit("f1", {**_manual("b.pdf"), "page": 5})
    assert same_source_region(a, b) is False


def test_manual_overlap_unavailable_without_pages() -> None:
    structural = [_hit("s1", _manual())]
    fixed = [_hit("f1", {**_manual(), "page": 5})]
    overlap = compute_region_overlap(structural, fixed)
    assert overlap.available is False
    assert overlap.percent is None


# ----------------------------------------------------------------------
# 9. Telegram message_ids overlap
# ----------------------------------------------------------------------
def test_telegram_message_id_overlap() -> None:
    structural = [_hit("s1", {**_telegram(), "message_ids": [150, 151, 152, 153]})]
    fixed = [_hit("f1", {**_telegram(), "message_ids": [152, 153, 154]})]
    assert same_source_region(structural[0], fixed[0]) is True
    overlap = compute_region_overlap(structural, fixed)
    assert overlap.available is True
    assert overlap.matched == 1


def test_telegram_message_ids_do_not_overlap() -> None:
    structural = [_hit("s1", {**_telegram(), "message_ids": [1, 2]})]
    fixed = [_hit("f1", {**_telegram(), "message_ids": [5, 6]})]
    assert same_source_region(structural[0], fixed[0]) is False
    overlap = compute_region_overlap(structural, fixed)
    assert overlap.available is True
    assert overlap.matched == 0


# ----------------------------------------------------------------------
# 10. empty search results
# ----------------------------------------------------------------------
def test_empty_search_results_are_handled(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = HashingEmbeddingProvider(dimension=64)
    structural = tmp_path / "structural.sqlite3"
    fixed = tmp_path / "fixed.sqlite3"
    _build_index(structural, [("s1", "давление", _manual())], provider=provider)
    _build_index(fixed, [("f1", "давление", _manual())], provider=provider, chunking="fixed")

    # An empty query has a zero vector -> the index returns no hits.
    result = compare_chunking(
        "",
        settings=settings,
        structural_path=structural,
        fixed_path=fixed,
        top_k=5,
        embedding_provider=provider,
    )
    assert result.structural_hits == []
    assert result.fixed_hits == []
    assert result.overlap.available is False
    assert result.structural_top1 is None
    assert result.structural_avg is None


# ----------------------------------------------------------------------
# 11. comparison statistics
# ----------------------------------------------------------------------
def test_comparison_statistics() -> None:
    structural_hits = [_hit("s1", {}, score=0.8), _hit("s2", {}, score=0.6)]
    fixed_hits = [_hit("f1", {}, score=0.7)]
    comparison = ChunkingComparison(
        query="q",
        top_k=2,
        structural=IndexInfo(path="s.sqlite3", chunks=10, avg_chars=100.0, min_chars=10, max_chars=200),
        fixed=IndexInfo(path="f.sqlite3", chunks=12, avg_chars=90.0, min_chars=5, max_chars=150),
        structural_hits=structural_hits,
        fixed_hits=fixed_hits,
        overlap=RegionOverlap(matched=1, compared=2, available=True),
        fixed_overlap=RegionOverlap(matched=1, compared=1, available=True),
    )
    assert comparison.structural_top1 == 0.8
    assert comparison.fixed_top1 == 0.7
    assert comparison.structural_avg == 0.7
    assert comparison.fixed_avg == 0.7
    assert comparison.overlap.percent == 50.0
    assert comparison.structural.chunks == 10
    assert comparison.fixed.avg_chars == 90.0


def test_read_index_info_reports_sizes_and_distribution(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    provider = HashingEmbeddingProvider(dimension=64)
    _build_index(
        path,
        [
            ("m1", "x" * 100, _manual()),
            ("t1", "y" * 300, _telegram()),
        ],
        provider=provider,
        chunking="fixed",
    )
    info = read_index_info(path)
    assert info.chunks == 2
    assert info.strategy == "fixed"
    assert info.min_chars == 100
    assert info.max_chars == 300
    assert info.avg_chars == 200.0
    assert info.chunks_by_source_type == {"manual": 1, "telegram": 1}
