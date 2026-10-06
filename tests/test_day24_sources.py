"""Day 24 clickable-source tests.

Covers the SourceLinkResolver (manual + Telegram), the secure manual file
endpoint (path traversal), quote -> message mapping, and the UI render
contract. No test requires a real Telegram export, Ollama or the network.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings, get_settings
from app.schemas.day24 import GroundedCitation
from app.services.rag.sources.links import SourceLinkResolver
from app.services.rag.sources.manual import (
    is_safe_manual_filename,
    manual_source_url,
    resolve_manual_path,
)
from app.services.rag.sources.telegram import (
    TelegramSourceIndex,
    internal_chat_id,
    is_valid_username,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAY24_JS = (PROJECT_ROOT / "app/static/js/day24.js").read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _citation(
    *,
    chunk_id="chunk-1",
    quote="quote text",
    source_type="manual",
    source="20_XPANDER_RU1.pdf",
    page=None,
    page_from=None,
    page_to=None,
    message_ids=None,
    topic_id=None,
    quote_valid=True,
):
    return GroundedCitation(
        chunk_id=chunk_id,
        quote=quote,
        quote_valid=quote_valid,
        source_type=source_type,
        source=source,
        page=page,
        page_from=page_from,
        page_to=page_to,
        message_ids=list(message_ids or []),
        topic_id=topic_id,
    )


def _manual_dir(tmp_path):
    directory = tmp_path / "manual"
    directory.mkdir()
    (directory / "20_XPANDER_RU1.pdf").write_bytes(b"%PDF-1.4 dummy")
    return directory


def _export(
    tmp_path,
    *,
    chat_id=1780128600,
    chat_type="public_supergroup",
    name="Xpander Club",
    username="",
    messages=None,
):
    messages = messages if messages is not None else [
        {"id": 83159, "type": "message", "date": "2026-09-22T12:08:00",
         "from": "Owner A", "text": "Цитируемый текст про пластик."},
        {"id": 83165, "type": "message", "date": "2026-09-22T12:20:00",
         "from": "Owner B", "text": "Другое сообщение обсуждения."},
    ]
    path = tmp_path / f"export_{chat_id}_{bool(username)}.json"
    path.write_text(
        json.dumps(
            {"id": chat_id, "name": name, "type": chat_type,
             "username": username, "messages": messages},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _resolver(settings, tmp_path, *, manual_root=None, export=None,
              base=None, index=None):
    return SourceLinkResolver(
        settings,
        manual_root=manual_root or _manual_dir(tmp_path),
        telegram_path=export,
        telegram_link_base=base,
        telegram_index=index,
    )


# ----------------------------------------------------------------------
# 1-4. Manual links
# ----------------------------------------------------------------------
def test_manual_source_gets_url(settings, tmp_path) -> None:
    resolver = _resolver(settings, tmp_path)
    link = resolver.resolve(_citation())
    assert link.url == "/api/rag/sources/manual/20_XPANDER_RU1.pdf"
    assert link.link_kind == "document"


def test_manual_page_added_to_url(settings, tmp_path) -> None:
    resolver = _resolver(settings, tmp_path)
    link = resolver.resolve(_citation(page=123))
    assert link.url == "/api/rag/sources/manual/20_XPANDER_RU1.pdf#page=123"


def test_manual_missing_page_has_no_fragment(settings, tmp_path) -> None:
    resolver = _resolver(settings, tmp_path)
    link = resolver.resolve(_citation(page=None, page_from=None))
    assert link.url is not None
    assert "#page=" not in link.url


def test_manual_page_from_used_when_page_missing(settings, tmp_path) -> None:
    resolver = _resolver(settings, tmp_path)
    link = resolver.resolve(_citation(page=None, page_from=238, page_to=239))
    assert link.url.endswith("#page=238")


def test_manual_url_does_not_expose_windows_path(settings, tmp_path) -> None:
    resolver = _resolver(settings, tmp_path)
    url = resolver.resolve(_citation(page=12)).url
    assert url is not None
    assert "C:" not in url
    assert "\\" not in url
    assert str(tmp_path) not in url
    assert url.startswith("/api/rag/sources/manual/")


def test_manual_unknown_file_has_no_url(settings, tmp_path) -> None:
    resolver = _resolver(settings, tmp_path)
    link = resolver.resolve(_citation(source="missing.pdf"))
    assert link.url is None


# ----------------------------------------------------------------------
# 5. Path traversal blocked
# ----------------------------------------------------------------------
def test_safe_filename_rules() -> None:
    assert is_safe_manual_filename("20_XPANDER_RU1.pdf")
    assert not is_safe_manual_filename("..")
    assert not is_safe_manual_filename("../.env")
    assert not is_safe_manual_filename("..\\..\\.env")
    assert not is_safe_manual_filename("subdir/doc.pdf")
    assert not is_safe_manual_filename("subdir\\doc.pdf")
    assert not is_safe_manual_filename(".hidden.pdf")
    assert not is_safe_manual_filename("")


def test_resolve_manual_path_rejects_traversal(tmp_path) -> None:
    root = _manual_dir(tmp_path)
    assert resolve_manual_path(root, "20_XPANDER_RU1.pdf") is not None
    assert resolve_manual_path(root, "../.env") is None
    assert resolve_manual_path(root, "..\\..\\.env") is None
    assert resolve_manual_path(root, "/etc/passwd") is None
    assert resolve_manual_path(root, "missing.pdf") is None
    # A non-PDF file inside the root must not be served either.
    (root / "notes.txt").write_text("secret", encoding="utf-8")
    assert resolve_manual_path(root, "notes.txt") is None


def test_manual_endpoint_serves_pdf(client, tmp_path) -> None:
    root = _manual_dir(tmp_path)
    client.app.dependency_overrides[get_settings] = lambda: Settings(
        rag_manual_path=str(root)
    )
    response = client.get("/api/rag/sources/manual/20_XPANDER_RU1.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


def test_manual_endpoint_blocks_traversal(client, tmp_path) -> None:
    root = _manual_dir(tmp_path)
    # Put a sensitive-ish file next to the manual root to prove it is not read.
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    client.app.dependency_overrides[get_settings] = lambda: Settings(
        rag_manual_path=str(root)
    )
    for path in (
        "/api/rag/sources/manual/..%5C..%5C.env",
        "/api/rag/sources/manual/%2e%2e%2f%2e%2e%2f.env",
        "/api/rag/sources/manual/notes.txt",
        "/api/rag/sources/manual/missing.pdf",
    ):
        response = client.get(path)
        assert response.status_code == 404, path


# ----------------------------------------------------------------------
# 6-9. Telegram URL strategies
# ----------------------------------------------------------------------
def test_telegram_configured_base_has_priority(settings, tmp_path) -> None:
    export = _export(tmp_path, username="publicname")
    resolver = _resolver(settings, tmp_path, export=export,
                         base="https://t.me/xpanderclub")
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json", message_ids=[83159],
        quote="Цитируемый текст про пластик.",
    ))
    assert link.url == "https://t.me/xpanderclub/83159"


def test_telegram_configured_base_forum_topic(settings, tmp_path) -> None:
    export = _export(tmp_path)
    resolver = _resolver(settings, tmp_path, export=export,
                         base="https://t.me/xpanderRU")
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json", message_ids=[83159],
        quote="Цитируемый текст про пластик.", topic_id=12943,
    ))
    assert link.url == "https://t.me/xpanderRU/12943/83159"


def test_telegram_xpanderru_base_cited_message(settings, tmp_path) -> None:
    """Real-world case: message_ids [83467, 83468], cite 83468."""
    export = _export(tmp_path, messages=[
        {"id": 83467, "type": "message", "date": "2026-10-02T19:13:00",
         "from": "Owner A", "text": "Не холодно ли в Xpander зимой?"},
        {"id": 83468, "type": "message", "date": "2026-10-02T19:43:00",
         "from": "Owner B", "text": "Тут люди разделятся на два фронта."},
    ])
    resolver = _resolver(settings, tmp_path, export=export,
                         base="https://t.me/xpanderRU")
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json",
        message_ids=[83467, 83468],
        quote="Тут люди разделятся на два фронта.",
    ))
    assert link.cited_message_id == 83468
    assert link.url == "https://t.me/xpanderRU/83468"
    cited = [item for item in link.context_links if item.is_cited]
    assert [item.message_id for item in cited] == [83468]
    assert {item.message_id for item in link.context_links} == {83467, 83468}


def test_telegram_public_username(settings, tmp_path) -> None:
    export = _export(tmp_path, username="publicname")
    resolver = _resolver(settings, tmp_path, export=export, base="")
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json", message_ids=[83159],
        quote="Цитируемый текст про пластик.",
    ))
    assert link.url == "https://t.me/publicname/83159"


def test_telegram_private_c_form(settings, tmp_path) -> None:
    # Telegram supergroup ids are often exported as -100<internal>.
    export = _export(tmp_path, chat_id=-1001780128600,
                     chat_type="private_supergroup", username="")
    resolver = _resolver(settings, tmp_path, export=export, base="")
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json", message_ids=[83159],
        quote="Цитируемый текст про пластик.",
    ))
    assert link.url == "https://t.me/c/1780128600/83159"
    assert link.note  # private access note


def test_telegram_unknown_chat_id_has_no_url(settings, tmp_path) -> None:
    export = _export(tmp_path, chat_id=None, chat_type="private_group")
    resolver = _resolver(settings, tmp_path, export=export, base="")
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json", message_ids=[83159],
        quote="Цитируемый текст про пластик.",
    ))
    assert link.url is None


def test_telegram_invalid_base_falls_back(settings, tmp_path) -> None:
    # A non-http base must be ignored (no javascript: injection).
    export = _export(tmp_path, username="publicname")
    resolver = _resolver(settings, tmp_path, export=export,
                         base="javascript:alert(1)")
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json", message_ids=[83159],
        quote="Цитируемый текст про пластик.",
    ))
    assert link.url == "https://t.me/publicname/83159"


def test_internal_chat_id_rules() -> None:
    assert internal_chat_id(1780128600, "public_supergroup") == "1780128600"
    assert internal_chat_id(-1001780128600, "supergroup") == "1780128600"
    assert internal_chat_id(-12345, "supergroup") is None
    assert internal_chat_id(1780128600, "private_group") is None
    assert internal_chat_id(None, "supergroup") is None
    assert is_valid_username("xpanderclub")
    assert not is_valid_username("Xpander RU")
    assert not is_valid_username("ab")


# ----------------------------------------------------------------------
# 10-14. cited_message_id and quote -> message mapping
# ----------------------------------------------------------------------
def test_telegram_url_uses_cited_message(settings, tmp_path) -> None:
    export = _export(tmp_path, username="publicname")
    resolver = _resolver(settings, tmp_path, export=export, base="")
    # The quote belongs to message 83165, not to the first message 83159.
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json",
        message_ids=[83159, 83165],
        quote="Другое сообщение обсуждения.",
    ))
    assert link.cited_message_id == 83165
    assert link.url == "https://t.me/publicname/83165"
    assert link.link_kind == "message"


def test_cited_message_is_within_chunk_message_ids(settings, tmp_path) -> None:
    export = _export(tmp_path, username="publicname")
    resolver = _resolver(settings, tmp_path, export=export, base="")
    # Message 83165 exists in the export but is NOT part of this chunk.
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json", message_ids=[83159],
        quote="Другое сообщение обсуждения.",
    ))
    assert link.cited_message_id is None
    assert link.link_kind == "conversation"
    assert link.url == "https://t.me/publicname/83159"


def test_quote_to_message_mapping(settings, tmp_path) -> None:
    export = _export(tmp_path)
    index = TelegramSourceIndex(export)
    assert index.find_message_for_quote([83159], "Цитируемый текст про пластик.") == 83159
    assert index.find_message_for_quote([83159, 83165], "Другое сообщение обсуждения.") == 83165
    assert index.find_message_for_quote([83159], "текста нет") is None


def test_invalid_message_quote_uses_conversation_anchor(settings, tmp_path) -> None:
    export = _export(tmp_path, username="publicname")
    resolver = _resolver(settings, tmp_path, export=export, base="")
    link = resolver.resolve(_citation(
        source_type="telegram", source="result.json",
        message_ids=[83159, 83165], quote="совсем другой текст",
    ))
    assert link.cited_message_id is None
    assert link.link_kind == "conversation"
    assert link.url is not None
    assert "83159" in (link.note or link.url)


# ----------------------------------------------------------------------
# 15. Invalid citation never gets a trusted link
# ----------------------------------------------------------------------
def test_invalid_citation_gets_no_link(settings, tmp_path) -> None:
    resolver = _resolver(settings, tmp_path)
    link = resolver.resolve(_citation(quote_valid=False))
    assert link.url is None


# ----------------------------------------------------------------------
# 16. Grounded API returns source URL
# ----------------------------------------------------------------------
def test_api_grounded_ask_returns_source_url(client) -> None:
    response = client.post(
        "/api/week5/day24/ask",
        json={"question": "Давление в шинах?", "similarity_threshold": 0.5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["sources"], "expected at least one source"
    source = body["sources"][0]
    assert source["url"] is not None
    assert source["url"].startswith("/api/rag/sources/manual/")
    # The citation carries the same resolved URL.
    assert body["citations"][0]["url"] == source["url"]


def test_api_telegram_source_url(client, day24_generation) -> None:
    day24_generation.payload = {
        "status": "answered",
        "answer": "Опыт владельцев.",
        "evidence": [
            {
                "chunk_id": "telegram-chunk-1",
                "quote": "Владелец связал вспучивание пластика с пеной для мойки.",
            }
        ],
    }
    response = client.post(
        "/api/week5/day24/ask",
        json={"question": "Про пластик?", "similarity_threshold": 0.5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    telegram = [s for s in body["sources"] if s["source_type"] == "telegram"]
    assert telegram, body["sources"]
    assert telegram[0]["url"] is not None
    assert telegram[0]["url"].startswith("https://t.me/")
    assert telegram[0]["cited_message_id"] == 83159


# ----------------------------------------------------------------------
# 17-19. Web UI render contract
# ----------------------------------------------------------------------
def test_ui_renders_clickable_manual_source() -> None:
    assert "d24-source-link" in DAY24_JS
    assert 'target = "_blank"' in DAY24_JS
    assert 'rel = "noopener noreferrer"' in DAY24_JS
    assert "renderCompactSources" in DAY24_JS
    # The source title itself is the clickable element.
    assert "sourceTitleNode" in DAY24_JS


def test_ui_renders_clickable_telegram_source() -> None:
    assert "Telegram" in DAY24_JS
    assert "sourceIcon" in DAY24_JS
    assert "d24-compact-icon" in DAY24_JS
    assert "conversation" in DAY24_JS
    assert "message " in DAY24_JS
    # Main cited message + context links.
    assert "renderContextLinks" in DAY24_JS
    assert "context_links" in DAY24_JS
    assert "(cited)" in DAY24_JS


def test_ui_source_without_url_has_no_not_available_marker() -> None:
    # A source without a URL stays readable (plain span); the old
    # "source URL: not available" marker is gone and the title is the link.
    assert "d24-source-plain" in DAY24_JS
    assert "source URL: not available" not in DAY24_JS
    assert "sourceTitleNode" in DAY24_JS


def test_manual_source_url_helper_encodes_cyrillic() -> None:
    url = manual_source_url("Инструкция_по_установке_защиты_шериф.pdf", page=3)
    assert url.startswith("/api/rag/sources/manual/")
    assert "#page=3" in url
    assert "%" in url  # percent-encoded, no raw path
