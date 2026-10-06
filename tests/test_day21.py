"""Day 21 — RAG knowledge-base ingestion/index tests.

All fixtures are small synthetic documents created in ``tmp_path``. No real
Telegram export, no manual PDF and no network service is touched. Embeddings
use the deterministic offline hashing provider, so tests are fast and stable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services.rag.chunking import (
    FixedChunker,
    StructuralChunker,
    build_chunker,
)
from app.services.rag.embeddings import HashingEmbeddingProvider
from app.services.rag.index import SqliteVectorIndex
from app.services.rag.ingestion.service import IngestionService
from app.services.rag.loaders import PdfLoader, TelegramLoader, TextLoader
from app.services.rag.loaders.pdf_loader import _looks_like_heading
from app.services.rag.models import Document, Segment, make_chunk_id
from app.services.rag.normalization.telegram import (
    TelegramMessage,
    extract_text,
    group_messages,
    parse_messages,
)
from app.services.rag.service import RagService


# ----------------------------------------------------------------------
# fixtures / helpers
# ----------------------------------------------------------------------
def _telegram_export() -> dict:
    return {
        "name": "SyntheticCrew",
        "type": "public_supergroup",
        "id": 1,
        "messages": [
            {"id": 1, "type": "message", "date": "2025-04-15T10:21:00",
             "from": "User A", "from_id": "u1", "text": "У кого была вибрация на D?"},
            {"id": 2, "type": "message", "date": "2025-04-15T10:23:00",
             "from": "User B", "from_id": "u2", "text": "У меня была."},
            {"id": 3, "type": "message", "date": "2025-04-15T10:24:00",
             "from": "User A", "from_id": "u1", "text": "Что оказалось?",
             "reply_to_message_id": 2},
            {"id": 4, "type": "message", "date": "2025-04-15T10:26:00",
             "from": "User B", "from_id": "u2", "text": "Правая подушка двигателя.",
             "reply_to_message_id": 3},
            # media-only -> skipped
            {"id": 5, "type": "message", "date": "2025-04-15T10:30:00",
             "from": "User C", "from_id": "u3", "text": "",
             "photo": "(File not included.)", "photo_file_size": 123},
            # service message -> skipped
            {"id": 6, "type": "service", "date": "2025-04-15T10:31:00",
             "action": "pin_message"},
            # a big time gap -> new conversation
            {"id": 7, "type": "message", "date": "2025-04-15T14:00:00",
             "from": "User A", "from_id": "u1", "text": "Про давление в шинах кто что скажет?"},
        ],
    }


def _write_telegram(tmp_path: Path) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_telegram_export(), ensure_ascii=False), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# 1. Telegram JSON parsing
# ----------------------------------------------------------------------
def test_telegram_json_parsing(tmp_path: Path) -> None:
    path = _write_telegram(tmp_path)
    stats = TelegramLoader(path).scan()
    assert stats.files == 1
    assert stats.messages == 7
    assert stats.usable_messages == 5
    assert stats.skipped_messages == 2
    assert stats.skip_reasons["photo_without_caption"] == 1
    assert stats.skip_reasons["service_or_non_message"] == 1


# ----------------------------------------------------------------------
# 2. Telegram text extraction
# ----------------------------------------------------------------------
def test_telegram_text_extraction_plain_and_entities() -> None:
    assert extract_text("hello") == "hello"
    entities = [
        "look ", {"type": "bold", "text": "here"}, {"type": "link", "text": " link"}
    ]
    assert extract_text(entities) == "look here link"
    assert extract_text(None) == ""


def test_parse_messages_flattens_entity_text() -> None:
    raw = [{
        "id": 99, "type": "message", "date": "2025-01-01T00:00:00",
        "from": "A", "from_id": "uA",
        "text": ["Важно: ", {"type": "bold", "text": "тормоза"}],
    }]
    messages, stats = parse_messages(raw)
    assert stats.usable == 1
    assert messages[0].text == "Важно: тормоза"


# ----------------------------------------------------------------------
# 3. Telegram conversation grouping
# ----------------------------------------------------------------------
def test_telegram_conversation_grouping() -> None:
    messages = [
        TelegramMessage(1, "2025-04-15T10:00:00", "A", "one"),
        TelegramMessage(2, "2025-04-15T10:05:00", "B", "two"),
        TelegramMessage(3, "2025-04-15T12:00:00", "A", "three"),  # gap > 30m
        TelegramMessage(4, "2025-04-15T12:05:00", "B", "four"),
    ]
    groups = group_messages(messages, gap_minutes=30, max_group_messages=10)
    assert [[m.message_id for m in g] for g in groups] == [[1, 2], [3, 4]]


def test_telegram_grouping_respects_size_limit() -> None:
    messages = [
        TelegramMessage(i, f"2025-04-15T10:{i:02d}:00", "A", f"m{i}")
        for i in range(10)
    ]
    groups = group_messages(messages, gap_minutes=60, max_group_messages=4)
    assert [len(g) for g in groups] == [4, 4, 2]


def test_telegram_loader_groups_context_with_message_ids(tmp_path: Path) -> None:
    path = _write_telegram(tmp_path)
    docs = list(TelegramLoader(path).load())
    assert len(docs) == 2  # first conversation + the message after the gap
    first = docs[0]
    assert first.metadata["message_ids"] == [1, 2, 3, 4]
    assert "Правая подушка двигателя." in first.text
    assert "У кого была вибрация на D?" in first.text
    assert first.metadata["source_type"] == "telegram"
    assert first.metadata["authors"] == ["User A", "User B"]
    # original message ids survive on the segments
    seg_ids = [s.metadata["message_id"] for s in first.segments]
    assert seg_ids == [1, 2, 3, 4]


# ----------------------------------------------------------------------
# 4. fixed chunking
# ----------------------------------------------------------------------
def _long_document() -> Document:
    text = " ".join(f"token{i}" for i in range(400))
    segments = [Segment(text=text, metadata={"page": 1})]
    return Document(text=text, metadata={"source_type": "manual", "source": "x.pdf"}, segments=segments)


def test_fixed_chunking_sizes() -> None:
    chunker = FixedChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk(_long_document())
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.metadata["source_type"] == "manual" for c in chunks)
    assert all(c.metadata["page"] == 1 for c in chunks)


def test_fixed_chunking_overlap() -> None:
    chunker = FixedChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk(_long_document())
    first, second = chunks[0], chunks[1]
    # The windows advance by 150 chars and are 200 long -> 50 chars shared.
    assert first.metadata["start_char"] == 0
    assert second.metadata["start_char"] == 150
    assert first.text[-50:] == second.text[:50]


def test_fixed_chunking_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=100, overlap=100)


# ----------------------------------------------------------------------
# 5/6. structural chunking
# ----------------------------------------------------------------------
def test_structural_chunking_respects_sections() -> None:
    segments = [
        Segment("Введение", {"section": "Введение", "is_heading": True, "page": 1}),
        Segment("Текст " * 30, {"section": "Введение", "page": 1}),
        Segment("Тормоза", {"section": "Тормоза", "is_heading": True, "page": 2}),
        Segment("Описание тормозной системы " * 20, {"section": "Тормоза", "page": 2}),
    ]
    document = Document(
        text="\n\n".join(s.text for s in segments),
        metadata={"source_type": "manual", "source": "x.pdf"},
        segments=segments,
    )
    chunks = StructuralChunker(max_chars=1000, min_chars=50).chunk(document)
    assert len(chunks) >= 2
    # The heading stays with the paragraph that belongs to it.
    assert chunks[0].text.startswith("Введение")
    assert "Тормоза" not in chunks[0].text
    assert any("Тормоза" in c.text for c in chunks)
    # Pages are preserved in the metadata.
    assert chunks[0].metadata["page_from"] == 1


def test_structural_splits_only_on_segment_boundaries() -> None:
    segments = [Segment(f"message-{i} " * 20, {"message_id": i}) for i in range(1, 5)]
    document = Document(
        text=" ".join(s.text for s in segments),
        metadata={"source_type": "telegram", "source": "result.json"},
        segments=segments,
    )
    chunks = StructuralChunker(max_chars=250, min_chars=0).chunk(document)
    assert len(chunks) > 1
    for chunk in chunks:
        # Every chunk consists only of whole message segments.
        tokens = chunk.text.split()
        assert tokens
        assert all(token.startswith("message-") for token in tokens)


# ----------------------------------------------------------------------
# 7. metadata preservation
# ----------------------------------------------------------------------
def test_metadata_preserved_across_chunking() -> None:
    segments = [
        Segment("страница один", {"page": 1, "section": "S"}),
        Segment("страница два", {"page": 2, "section": "S"}),
    ]
    document = Document(
        text="\n\n".join(s.text for s in segments),
        metadata={"source_type": "manual", "source": "manual.pdf", "title": "Manual"},
        segments=segments,
    )
    chunk = StructuralChunker(max_chars=10000, min_chars=0).chunk(document)[0]
    assert chunk.metadata["source_type"] == "manual"
    assert chunk.metadata["source"] == "manual.pdf"
    assert chunk.metadata["title"] == "Manual"
    assert chunk.metadata["page_from"] == 1
    assert chunk.metadata["page_to"] == 2
    assert chunk.metadata["section"] == "S"


def test_telegram_metadata_on_chunks(tmp_path: Path) -> None:
    docs = list(TelegramLoader(_write_telegram(tmp_path)).load())
    chunks = FixedChunker(chunk_size=500, overlap=50).chunk(docs[0])
    chunk = chunks[0]
    assert chunk.metadata["source_type"] == "telegram"
    assert chunk.metadata["message_ids"]
    assert chunk.metadata["author"]
    assert chunk.metadata["date_from"]


# ----------------------------------------------------------------------
# 8. deterministic chunk_id
# ----------------------------------------------------------------------
def test_chunk_ids_are_deterministic() -> None:
    document = _long_document()
    first = FixedChunker(chunk_size=200, overlap=50).chunk(document)
    second = FixedChunker(chunk_size=200, overlap=50).chunk(document)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    structural = StructuralChunker(max_chars=500, min_chars=0).chunk(document)
    assert all(c.chunk_id not in {f.chunk_id for f in first} for c in structural)


def test_make_chunk_id_changes_with_text() -> None:
    assert make_chunk_id("doc", "fixed", 0, "abc") != make_chunk_id("doc", "fixed", 0, "abcd")


# ----------------------------------------------------------------------
# PDF loader (synthetic, no real file)
# ----------------------------------------------------------------------
class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, pages: list[str]) -> None:
        self.pages = [_FakePage(p) for p in pages]


def test_pdf_loader_blocks_and_pages(tmp_path: Path) -> None:
    reader = _FakeReader([
        "Введение\n\nПервый абзац текста.",
        "Тормозная система\n\nОписание тормозов автомобиля.",
    ])
    loader = PdfLoader(tmp_path)
    document, stats = loader._build_document(Path("manual.pdf"), reader)
    assert stats.pages == 2
    assert document is not None
    assert document.metadata["source_type"] == "manual"
    pages = {s.metadata.get("page") for s in document.segments}
    assert pages == {1, 2}


def test_pdf_heading_heuristic() -> None:
    assert _looks_like_heading("Тормозная система")
    assert _looks_like_heading("1.2 Общая информация")
    assert not _looks_like_heading("Это длинное предложение заканчивается точкой.")
    assert not _looks_like_heading("текст " * 40)


# ----------------------------------------------------------------------
# Text loader
# ----------------------------------------------------------------------
def test_text_loader_markdown_sections(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text(
        "# Раздел\n\nПервый абзац.\n\n## Подраздел\n\nВторой абзац.",
        encoding="utf-8",
    )
    docs = list(TextLoader(tmp_path).load())
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Раздел"
    sections = [s.metadata.get("section") for s in docs[0].segments]
    assert "Раздел" in sections
    assert "Подраздел" in sections


# ----------------------------------------------------------------------
# 9/10. embedding/index persistence + retrieval after restart
# ----------------------------------------------------------------------
def test_index_persistence_and_retrieval(tmp_path: Path) -> None:
    provider = HashingEmbeddingProvider(dimension=128)
    path = tmp_path / "index.sqlite3"
    chunks = [
        Segment("правая подушка двигателя вибрация", {}),
        Segment("давление в шинах рекомендации", {}),
    ]
    from app.services.rag.models import Chunk

    chunk_objs = [
        Chunk("c1", chunks[0].text, {"source_type": "telegram"}, "d1"),
        Chunk("c2", chunks[1].text, {"source_type": "telegram"}, "d1"),
    ]
    index = SqliteVectorIndex(path)
    index.add_chunks(chunk_objs, provider.embed([c.text for c in chunk_objs]))
    index.set_meta({"embedding_provider": "hashing", "embedding_dimension": 128})
    index.close()

    # Reopen as a new object: simulates a restart.
    reopened = SqliteVectorIndex(path)
    assert reopened.count() == 2
    assert reopened.get_chunk("c1") is not None
    hits = reopened.search(provider.embed_query("вибрация подушка"), top_k=1)
    assert hits and hits[0].chunk_id == "c1"
    assert hits[0].metadata["source_type"] == "telegram"
    reopened.close()


# ----------------------------------------------------------------------
# full ingestion service (synthetic sources) + atomic build
# ----------------------------------------------------------------------
def _settings(tmp_path: Path) -> Settings:
    manuals = tmp_path / "manuals"
    manuals.mkdir()
    (manuals / "guide.md").write_text(
        "# Двигатель\n\nМасло вариатора следует менять по регламенту.\n\n"
        "# Шины\n\nДавление в шинах проверяйте регулярно.",
        encoding="utf-8",
    )
    telegram = _write_telegram(tmp_path)
    return Settings(
        rag_manual_path=str(manuals),
        rag_telegram_export_path=str(telegram),
        rag_index_path=str(tmp_path / "rag.sqlite3"),
        rag_embedding_provider="hashing",
        rag_embedding_dimension=128,
    )


def test_ingestion_build_is_atomic_and_searchable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = HashingEmbeddingProvider(dimension=128)
    service = RagService(settings, embedding_provider=provider)

    report = service.build_index("structural")
    assert report.chunks_created > 0
    assert report.chunks_by_source_type.get("manual", 0) > 0
    assert report.chunks_by_source_type.get("telegram", 0) > 0
    assert report.embedding_dimension == 128

    # No partial build file left behind.
    assert not list(tmp_path.glob("*.building*"))
    assert Path(settings.rag_index_path).exists()

    hits = service.search("давление в шинах", top_k=3)
    assert hits
    assert any("шин" in hit.text.lower() for hit in hits)

    stats = service.stats()
    assert stats["exists"] is True
    assert stats["chunks"] == report.chunks_created
    assert stats["embedding_provider"] == "hashing"


def test_ingestion_build_failure_leaves_no_partial_index(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class BoomProvider(HashingEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimension=32)
            self.calls = 0

        def embed(self, texts):  # type: ignore[override]
            raise RuntimeError("boom")

    service = IngestionService(settings, embedding_provider=BoomProvider())
    with pytest.raises(RuntimeError):
        service.build("fixed")
    assert not Path(settings.rag_index_path).exists()
    assert not list(tmp_path.glob("*.building*"))


# ----------------------------------------------------------------------
# chunker factory
# ----------------------------------------------------------------------
def test_build_chunker_from_settings() -> None:
    settings = Settings(rag_chunk_size=500, rag_chunk_overlap=100, rag_structural_max_chars=900)
    fixed = build_chunker("fixed", settings=settings)
    assert isinstance(fixed, FixedChunker)
    assert fixed.chunk_size == 500
    assert fixed.overlap == 100
    structural = build_chunker("structural", settings=settings)
    assert isinstance(structural, StructuralChunker)
    assert structural.max_chars == 900
    with pytest.raises(ValueError):
        build_chunker("nope", settings=settings)
