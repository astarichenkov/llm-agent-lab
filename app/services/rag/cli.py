"""Day 21 command-line demo.

Usage::

    python -m app.services.rag.cli sources
    python -m app.services.rag.cli index --chunking fixed
    python -m app.services.rag.cli index --chunking structural
    python -m app.services.rag.cli stats
    python -m app.services.rag.cli chunks --source telegram --limit 5
    python -m app.services.rag.cli search "вибрация на D" --top-k 5

The commands only read/build the local index; no LLM is called.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from app.config import get_settings
from app.services.rag.chunking import SUPPORTED_STRATEGIES
from app.services.rag.comparison import (
    ChunkingComparison,
    ComparisonError,
    EmptyIndexError,
    IndexCompatibilityError,
    IndexInfo,
    IndexNotFoundError,
    compare_chunking,
)
from app.services.rag.embeddings.base import EmbeddingError
from app.services.rag.index.base import VectorIndexError
from app.services.rag.models import BuildReport, Chunk, LoaderStats, SearchHit
from app.services.rag.service import RagService


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - platform dependent
        pass


def _service(args: argparse.Namespace) -> RagService:
    settings = get_settings()
    if getattr(args, "provider", None):
        settings.rag_embedding_provider = args.provider
    if getattr(args, "model", None):
        settings.rag_embedding_model = args.model
    return RagService(
        settings,
        index_path=getattr(args, "index", None),
        manual_path=getattr(args, "manuals", None),
        telegram_path=getattr(args, "telegram", None),
    )


def _print_sources(stats: Iterable[LoaderStats]) -> None:
    stats = list(stats)
    manuals = [s for s in stats if s.source_type == "manual"]
    telegram = [s for s in stats if s.source_type == "telegram"]
    other = [s for s in stats if s.source_type not in ("manual", "telegram")]

    if manuals:
        files = sum(s.files for s in manuals)
        pages = sum(s.pages for s in manuals)
        print("Manuals")
        print("-------")
        print(f"files: {files}")
        print(f"pages: {pages}")
    if telegram:
        messages = sum(s.messages for s in telegram)
        usable = sum(s.usable_messages for s in telegram)
        skipped = sum(s.skipped_messages for s in telegram)
        print("\nTelegram")
        print("--------")
        print(f"messages: {messages}")
        print(f"text messages: {usable}")
        print(f"skipped: {skipped}")
        reasons: dict[str, int] = {}
        for stat in telegram:
            for reason, count in stat.skip_reasons.items():
                reasons[reason] = reasons.get(reason, 0) + count
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
            print(f"  {reason}: {count}")
    for stat in other:
        print(f"\n{stat.source_type}\n{'-' * len(stat.source_type)}")
        print(f"files: {stat.files}")


def _print_build(report: BuildReport) -> None:
    print(f"Chunking strategy: {report.chunking}")
    print(f"Documents loaded:  {report.documents_loaded}")
    print(f"Telegram messages: {report.messages_loaded} used, {report.messages_skipped} skipped")
    print(f"Chunks created:    {report.chunks_created}")
    print(f"Chunk size:        avg={report.avg_chunk_size} min={report.min_chunk_size} max={report.max_chunk_size}")
    print("Chunks by source_type:")
    for source_type, count in sorted(report.chunks_by_source_type.items()):
        print(f"  {source_type}: {count}")
    print(f"Embeddings:        {report.embedding_provider} / {report.embedding_model} (dim={report.embedding_dimension})")
    print(f"Index:             {report.index_path}")
    print(f"Elapsed:           {report.elapsed_seconds}s")


def _print_chunk(chunk: Chunk, *, full: bool = False, max_chars: int = 600) -> None:
    print(f"chunk_id : {chunk.chunk_id}")
    print(f"doc_id   : {chunk.doc_id}")
    metadata = {k: v for k, v in chunk.metadata.items() if k != "locator"}
    print(f"metadata : {json.dumps(metadata, ensure_ascii=False)}")
    text = chunk.text if full else chunk.text[:max_chars]
    print(f"text     :\n{text}")
    if not full and len(chunk.text) > max_chars:
        print("...")
    print("-" * 60)


def _print_hit(hit: SearchHit, *, max_chars: int = 300) -> None:
    metadata = hit.metadata
    print(f"score       : {hit.score:.4f}")
    print(f"source_type : {metadata.get('source_type', '')}")
    print(f"source      : {metadata.get('source', '')}")
    print(f"chunk_id    : {hit.chunk_id}")
    if metadata.get("page") is not None:
        print(f"page        : {metadata.get('page')}")
    elif metadata.get("page_from") is not None:
        print(f"page        : {metadata.get('page_from')}-{metadata.get('page_to')}")
    if metadata.get("section"):
        print(f"section     : {metadata.get('section')}")
    if metadata.get("message_ids"):
        print(f"message_ids : {metadata.get('message_ids')}")
    print(f"text        : {hit.text[:max_chars].strip()}")
    print("-" * 60)


def cmd_sources(args: argparse.Namespace) -> int:
    service = _service(args)
    stats = service.sources()
    telegram = [s for s in stats if s.source_type == "telegram"]
    if telegram and telegram[0].files == 0:
        print(f"Telegram export not found: {service.ingestion.telegram_path}\n")
    _print_sources(stats)
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.build_index(args.chunking, rebuild=args.rebuild)
    _print_build(report)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    service = _service(args)
    stats = service.stats()
    if not stats.get("exists"):
        print(f"Index not found: {stats.get('index_path')}. Run `... index` first.")
        return 1
    print(f"Index path:        {stats.get('index_path')}")
    print(f"Chunks:            {stats.get('chunks')}")
    print(f"Chunk size:        avg={stats.get('avg_chunk_size')} min={stats.get('min_chunk_size')} max={stats.get('max_chunk_size')}")
    print(f"Chunking:          {stats.get('chunking')}")
    print(f"Embeddings:        {stats.get('embedding_provider')} / {stats.get('embedding_model')} (dim={stats.get('embedding_dimension')})")
    print(f"Created at:        {stats.get('created_at')}")
    print("Chunks by source_type:")
    for source_type, count in sorted((stats.get("chunks_by_source_type") or {}).items()):
        print(f"  {source_type}: {count}")
    return 0


def cmd_chunks(args: argparse.Namespace) -> int:
    service = _service(args)
    chunks = service.chunks(args.source, limit=args.limit, offset=args.offset)
    if not chunks:
        print("No chunks found (build the index first?).")
        return 1
    for chunk in chunks:
        _print_chunk(chunk, full=args.full)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    service = _service(args)
    hits = service.search(args.query, top_k=args.top_k, source_type=args.source)
    if not hits:
        print("No results (index missing or empty).")
        return 1
    print(f"Query: {args.query}\n")
    for hit in hits:
        _print_hit(hit)
    return 0


# ----------------------------------------------------------------------
# compare-chunking (Day 21 extension)
# ----------------------------------------------------------------------
_COMPARE_WIDTH = 60


def _score(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _print_compare_hit(rank: int, hit: SearchHit, *, full: bool, max_chars: int = 400) -> None:
    metadata = hit.metadata
    print(f"#{rank}")
    print(f"score: {hit.score:.4f}")
    print(f"source_type: {metadata.get('source_type', '')}")
    print(f"source: {metadata.get('source', '')}")
    if metadata.get("page") is not None:
        print(f"page: {metadata.get('page')}")
    elif metadata.get("page_from") is not None:
        print(f"page: {metadata.get('page_from')}-{metadata.get('page_to')}")
    if metadata.get("section"):
        print(f"section: {metadata.get('section')}")
    if metadata.get("message_ids"):
        print(f"message_ids: {metadata.get('message_ids')}")
    print(f"chunk_id: {hit.chunk_id}")
    print("text:")
    text = hit.text if full else hit.text[:max_chars].strip()
    print(text)
    if not full and len(hit.text) > max_chars:
        print("...")
    print()


def _print_index_block(label: str, info: IndexInfo) -> None:
    print(label)
    print(f"  index: {info.path}")
    print(f"  chunks: {info.chunks}")
    print(f"  strategy: {info.strategy}")
    print(f"  avg chars: {info.avg_chars}")
    print(f"  min chars: {info.min_chars}")
    print(f"  max chars: {info.max_chars}")


def _print_source_distribution(structural: IndexInfo, fixed: IndexInfo) -> None:
    source_types = sorted(
        set(structural.chunks_by_source_type) | set(fixed.chunks_by_source_type)
    )
    if not source_types:
        return
    print("Chunks by source type")
    print()
    print(f"                {'STRUCTURAL':>12}    {'FIXED':>8}")
    for source_type in source_types:
        left = structural.chunks_by_source_type.get(source_type, 0)
        right = fixed.chunks_by_source_type.get(source_type, 0)
        print(f"{source_type:<14}{left:>12}{right:>13}")


def _print_incompatible(structural: IndexInfo, fixed: IndexInfo) -> None:
    print("Cannot compare indexes.")
    print()
    print("Structural:")
    print(f"  provider: {structural.provider}")
    print(f"  model: {structural.model}")
    print()
    print("Fixed:")
    print(f"  provider: {fixed.provider}")
    print(f"  model: {fixed.model}")
    print()
    print("Indexes were created with different embedding models.")


def _print_comparison(comparison: ChunkingComparison, *, full: bool) -> None:
    structural = comparison.structural
    fixed = comparison.fixed
    provider = structural.provider or fixed.provider
    model = structural.model or fixed.model

    print("=" * _COMPARE_WIDTH)
    print("DAY 21 — CHUNKING STRATEGY COMPARISON")
    print("=" * _COMPARE_WIDTH)
    print()
    print("Query:")
    print(f"  {comparison.query}")
    print()
    print("Embedding:")
    print(f"  provider: {provider}")
    print(f"  model: {model}")
    print()

    print("-" * _COMPARE_WIDTH)
    print("INDEX SUMMARY")
    print("-" * _COMPARE_WIDTH)
    print()
    _print_index_block("STRUCTURAL", structural)
    print()
    _print_index_block("FIXED", fixed)
    print()
    _print_source_distribution(structural, fixed)
    print()

    print("-" * _COMPARE_WIDTH)
    print(f"STRUCTURAL — TOP {comparison.top_k}")
    print("-" * _COMPARE_WIDTH)
    print()
    if comparison.structural_hits:
        for rank, hit in enumerate(comparison.structural_hits, start=1):
            _print_compare_hit(rank, hit, full=full)
    else:
        print("No results.\n")

    print("-" * _COMPARE_WIDTH)
    print(f"FIXED — TOP {comparison.top_k}")
    print("-" * _COMPARE_WIDTH)
    print()
    if comparison.fixed_hits:
        for rank, hit in enumerate(comparison.fixed_hits, start=1):
            _print_compare_hit(rank, hit, full=full)
    else:
        print("No results.\n")

    print("-" * _COMPARE_WIDTH)
    print("COMPARISON")
    print("-" * _COMPARE_WIDTH)
    print()
    print(f"Top-K requested:             {comparison.top_k}")
    print(f"Structural results:          {len(comparison.structural_hits)}")
    print(f"Fixed results:               {len(comparison.fixed_hits)}")
    print()
    print(f"Same source/chunk regions:   {comparison.overlap.label}")
    if comparison.overlap.percent is not None:
        print(f"Source overlap:              {comparison.overlap.percent}%")
    else:
        print("Source overlap:              unavailable")
    print(f"Fixed regions matched:       {comparison.fixed_overlap.label}")
    print()
    print(f"Structural top-1 score:      {_score(comparison.structural_top1)}")
    print(f"Fixed top-1 score:           {_score(comparison.fixed_top1)}")
    print()
    print(f"Structural avg top-K score:  {_score(comparison.structural_avg)}")
    print(f"Fixed avg top-K score:       {_score(comparison.fixed_avg)}")
    print()
    print("No automatic winner is declared: similarity scores alone are not a")
    print("quality metric. Compare the returned regions and read the chunks.")


def _print_compact(comparison: ChunkingComparison) -> None:
    print(f"QUERY: {comparison.query}")
    print()
    print(f"                {'STRUCTURAL':>12}    {'FIXED':>8}")
    print(
        f"top1 score    {_score(comparison.structural_top1):>12}"
        f"    {_score(comparison.fixed_top1):>8}"
    )
    label = f"avg top{comparison.top_k}"
    print(
        f"{label:<14}{_score(comparison.structural_avg):>12}"
        f"    {_score(comparison.fixed_avg):>8}"
    )
    print(
        f"region overlap{comparison.overlap.label:>12}"
        f"    {comparison.fixed_overlap.label:>8}"
    )
    print()


def _read_queries(path: str) -> list[str]:
    queries: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        queries.append(stripped)
    return queries


def _compare_paths(args: argparse.Namespace) -> tuple[str, str]:
    settings = get_settings()
    structural = args.structural_index or settings.rag_index_path
    fixed = args.fixed_index or settings.rag_fixed_index_path
    return structural, fixed


def _handle_comparison_error(exc: ComparisonError) -> int:
    if isinstance(exc, IndexNotFoundError):
        print(f"{exc.label} index not found:\n{exc.path}")
    elif isinstance(exc, EmptyIndexError):
        print(f"{exc.label} index is empty:\n{exc.path}")
    elif isinstance(exc, IndexCompatibilityError):
        _print_incompatible(exc.structural, exc.fixed)
    else:
        print(exc.message)
    return 1


def cmd_compare_chunking(args: argparse.Namespace) -> int:
    if args.query and args.queries_file:
        print("Provide either QUERY or --queries-file, not both.")
        return 2
    if not args.query and not args.queries_file:
        print("Provide a QUERY or --queries-file.")
        return 2

    settings = get_settings()
    structural_path, fixed_path = _compare_paths(args)

    if args.queries_file:
        try:
            queries = _read_queries(args.queries_file)
        except OSError as exc:
            print(f"Cannot read queries file: {args.queries_file}\n{exc}")
            return 1
        if not queries:
            print(f"Queries file contains no queries: {args.queries_file}")
            return 1

        failures = 0
        for query in queries:
            try:
                comparison = compare_chunking(
                    query,
                    settings=settings,
                    structural_path=structural_path,
                    fixed_path=fixed_path,
                    top_k=args.top_k,
                )
            except ComparisonError as exc:
                failures += 1
                _handle_comparison_error(exc)
                print()
                continue
            except EmbeddingError as exc:
                failures += 1
                print(f"QUERY: {query}")
                print(f"Embedding failed: {exc.message}")
                print()
                continue
            except VectorIndexError as exc:
                failures += 1
                print(f"Cannot read index: {exc.message}")
                print()
                continue
            _print_compact(comparison)
        print("-" * _COMPARE_WIDTH)
        print(f"BATCH SUMMARY: {len(queries) - failures} / {len(queries)} queries compared")
        return 1 if failures else 0

    try:
        comparison = compare_chunking(
            args.query,
            settings=settings,
            structural_path=structural_path,
            fixed_path=fixed_path,
            top_k=args.top_k,
        )
    except ComparisonError as exc:
        return _handle_comparison_error(exc)
    except EmbeddingError as exc:
        print("Cannot compute query embedding.")
        print()
        print(f"  error: {exc.message}")
        print()
        print(f"  expected model: {settings.rag_embedding_model}")
        print(
            f"  Ollama endpoint: {settings.rag_ollama_base_url}"
        )
        return 1
    except VectorIndexError as exc:
        print(f"Cannot read index: {exc.message}")
        return 1

    _print_comparison(comparison, full=args.full)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="Day 21 — local Automotive RAG ingestion/index demo."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index", help="Override the index path.")
    common.add_argument("--manuals", help="Override the manuals directory.")
    common.add_argument("--telegram", help="Override the Telegram export path.")
    common.add_argument("--provider", help="Override the embedding provider.")
    common.add_argument("--model", help="Override the embedding model.")

    p_sources = sub.add_parser("sources", parents=[common], help="Show source statistics.")
    p_sources.set_defaults(func=cmd_sources)

    p_index = sub.add_parser("index", parents=[common], help="Build/rebuild the index.")
    p_index.add_argument("--chunking", choices=SUPPORTED_STRATEGIES, default="fixed")
    p_index.add_argument("--rebuild", action="store_true", help="Force a full rebuild.")
    p_index.set_defaults(func=cmd_index)

    p_stats = sub.add_parser("stats", parents=[common], help="Show index statistics.")
    p_stats.set_defaults(func=cmd_stats)

    p_chunks = sub.add_parser("chunks", parents=[common], help="Show chunks.")
    p_chunks.add_argument("--source", help="Filter by source_type (manual/telegram).")
    p_chunks.add_argument("--limit", type=int, default=5)
    p_chunks.add_argument("--offset", type=int, default=0)
    p_chunks.add_argument("--full", action="store_true", help="Print the full chunk text.")
    p_chunks.set_defaults(func=cmd_chunks)

    p_search = sub.add_parser("search", parents=[common], help="Debug similarity search.")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--source", help="Filter by source_type (manual/telegram).")
    p_search.set_defaults(func=cmd_search)

    p_compare = sub.add_parser(
        "compare-chunking",
        help="Compare retrieval of two indexes (structural vs fixed).",
    )
    p_compare.add_argument("query", nargs="?", help="Query string (single mode).")
    p_compare.add_argument("--top-k", type=int, default=5)
    p_compare.add_argument(
        "--structural-index",
        help="Path to the structural index (default: RAG_INDEX_PATH).",
    )
    p_compare.add_argument(
        "--fixed-index",
        help="Path to the fixed index (default: RAG_FIXED_INDEX_PATH).",
    )
    p_compare.add_argument(
        "--queries-file",
        help="Batch mode: file with one query per line (empty/# lines ignored).",
    )
    p_compare.add_argument(
        "--full", action="store_true", help="Print the full chunk text."
    )
    p_compare.set_defaults(func=cmd_compare_chunking)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
