"""SQLite-backed persistent vector index.

Chosen for Day 21 because the project already uses SQLite for every other
persistent store (agents, monitoring, ...) and it needs no extra service. It
stores, for every chunk:

* the chunk text (so it can be returned, not just an id);
* the full metadata JSON (source, page, message ids, authors, ...);
* the embedding vector as a float32 BLOB plus its L2 norm for fast cosine.

The index survives process restarts. Day 22 can add a FAISS/HNSW layer on top
without changing this contract, and incremental indexing is possible because
``chunk_id`` is deterministic and ``doc_id`` is stored alongside.
"""
from __future__ import annotations

import json
import math
import sqlite3
from array import array
from pathlib import Path
from typing import Any, Iterable

from app.services.rag.index.base import VectorIndexError
from app.services.rag.models import Chunk, SearchHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    source_type TEXT,
    source TEXT,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    embedding BLOB NOT NULL,
    norm REAL NOT NULL,
    ordinal INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _encode(vector: list[float]) -> bytes:
    return array("f", [float(value) for value in vector]).tobytes()


def _decode(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


class SqliteVectorIndex:
    """Persistent chunk + embedding store implemented on SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - filesystem dependent
            raise VectorIndexError(f"Cannot open index {self.path}: {exc}") from exc

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def set_meta(self, values: dict[str, Any]) -> None:
        rows = [(key, json.dumps(value, ensure_ascii=False)) for key, value in values.items()]
        self._conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", rows
        )
        self._conn.commit()

    def get_meta(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for row in self._conn.execute("SELECT key, value FROM meta"):
            try:
                result[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                result[row["key"]] = row["value"]
        return result

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise VectorIndexError("chunks and embeddings length mismatch")
        rows = []
        for ordinal, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    str(chunk.metadata.get("source_type", "")),
                    str(chunk.metadata.get("source", "")),
                    chunk.text,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    _encode(embedding),
                    _norm(embedding),
                    ordinal,
                )
            )
        try:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, source_type, source, text, metadata,
                     embedding, norm, ordinal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise VectorIndexError(f"Failed to write chunks: {exc}") from exc
        return len(rows)

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError:
            metadata = {}
        return Chunk(
            chunk_id=row["chunk_id"],
            text=row["text"],
            metadata=metadata,
            doc_id=row["doc_id"],
        )

    def iter_chunks(
        self,
        *,
        source_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Iterable[Chunk]:
        sql = "SELECT * FROM chunks"
        params: list[Any] = []
        if source_type:
            sql += " WHERE source_type = ?"
            params.append(source_type)
        sql += " ORDER BY rowid"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)
        for row in self._conn.execute(sql, params):
            yield self._row_to_chunk(row)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def neighbors(
        self, chunk_id: str, *, radius: int = 1, limit: int = 8
    ) -> list[Chunk]:
        """Return chunks adjacent to ``chunk_id`` inside the SAME document.

        Ordering is by insertion ``rowid`` within the ``doc_id``. This is the
        controlled context expansion used for manuals/PDFs: a strong hit on a
        title page can bring in the neighbouring fasteners/spec page without
        ever inventing content. Only real indexed chunks are returned.
        """
        if radius < 1 or limit < 1:
            return []
        anchor = self._conn.execute(
            "SELECT doc_id FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if anchor is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY rowid",
            (anchor["doc_id"],),
        ).fetchall()
        ids = [row["chunk_id"] for row in rows]
        try:
            index = ids.index(chunk_id)
        except ValueError:  # pragma: no cover - defensive
            return []
        selected = rows[max(0, index - radius) : index] + rows[
            index + 1 : index + 1 + radius
        ]
        return [self._row_to_chunk(row) for row in selected][:limit]

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[SearchHit]:
        query_norm = _norm(query_embedding)
        if query_norm == 0:
            return []
        sql = "SELECT * FROM chunks"
        params: list[Any] = []
        if source_type:
            sql += " WHERE source_type = ?"
            params.append(source_type)

        import heapq

        best: list[tuple[float, int, sqlite3.Row]] = []
        counter = 0
        for row in self._conn.execute(sql, params):
            vector = _decode(row["embedding"])
            stored_norm = row["norm"] or _norm(vector)
            if stored_norm == 0:
                continue
            dot = 0.0
            for a, b in zip(query_embedding, vector):
                dot += a * b
            score = dot / (query_norm * stored_norm)
            counter += 1
            if len(best) < top_k:
                heapq.heappush(best, (score, counter, row))
            elif score > best[0][0]:
                heapq.heapreplace(best, (score, counter, row))
        ordered = sorted(best, key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                score=score,
                chunk_id=row["chunk_id"],
                text=row["text"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                doc_id=row["doc_id"],
            )
            for score, _, row in ordered
        ]

    # ------------------------------------------------------------------
    # stats / lifecycle
    # ------------------------------------------------------------------
    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, AVG(LENGTH(text)) AS avg_len, "
            "MIN(LENGTH(text)) AS min_len, MAX(LENGTH(text)) AS max_len FROM chunks"
        ).fetchone()
        by_source = {
            r["source_type"]: r["n"]
            for r in self._conn.execute(
                "SELECT source_type, COUNT(*) AS n FROM chunks GROUP BY source_type"
            )
        }
        meta = self.get_meta()
        return {
            "chunks": int(row["n"]) if row else 0,
            "avg_chunk_size": round(float(row["avg_len"]), 1) if row and row["avg_len"] else 0.0,
            "min_chunk_size": int(row["min_len"]) if row and row["min_len"] is not None else 0,
            "max_chunk_size": int(row["max_len"]) if row and row["max_len"] is not None else 0,
            "chunks_by_source_type": by_source,
            "embedding_provider": meta.get("embedding_provider", ""),
            "embedding_model": meta.get("embedding_model", ""),
            "embedding_dimension": meta.get("embedding_dimension", 0),
            "chunking": meta.get("chunking", ""),
            "created_at": meta.get("created_at", ""),
            "index_path": str(self.path),
        }

    def reset(self) -> None:
        self._conn.executescript("DELETE FROM chunks; DELETE FROM meta;")
        self._conn.commit()

    def checkpoint(self) -> None:
        """Flush the WAL into the main database file (used before an atomic rename)."""
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:  # pragma: no cover - best effort
            pass

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover
            pass

    def __enter__(self) -> "SqliteVectorIndex":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
