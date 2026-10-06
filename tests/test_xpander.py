"""Tests for the standalone ``/xpander`` product page and its API facade.

The page must look like a finished automotive assistant and must NOT expose
the laboratory structure (Week 5 / Day 21-25 / Task State / RAG internals).
The API reuses the Day 25 chat service, so these tests run on the same fakes.
"""
import re


LAB_MARKERS = [
    "Week 5", "Day 21", "Day 22", "Day 23", "Day 24", "Day 25",
    "RAG Indexing", "Improved RAG", "Grounded RAG",
    "Task State", "GOAL", "VEHICLE", "Technical details",
    "contextual query", "rewritten query", "retrieval_top_k",
    "chunk_id", "similarity", "grounding",
]


def test_xpander_page_renders(client):
    response = client.get("/xpander")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "Mitsubishi Xpander Assistant" in html
    assert "Чаты, опыт владельцев и база знаний по Xpander" in html
    assert "Популярные темы" in html
    assert "Ваши чаты" in html
    assert "Текущая тема" in html
    assert "Похожие темы" in html
    assert "Задайте вопрос об Mitsubishi Xpander..." in html


def test_xpander_page_hides_lab_structure(client):
    html = client.get("/xpander").text
    for marker in LAB_MARKERS:
        assert marker not in html, f"lab marker leaked into /xpander: {marker!r}"


def test_xpander_page_uses_facade_api_not_lab_api(client):
    html = client.get("/xpander").text
    assert "/api/xpander" not in html  # base URL lives in the JS file
    assert "/api/week5" not in html
    js = client.get("/static/js/xpander.js").text
    assert '"/api/xpander"' in js
    assert "/api/week5" not in js
    for marker in ["Task State", "GOAL", "VEHICLE", "Technical details", "grounding"]:
        assert marker not in js, f"lab marker leaked into xpander.js: {marker!r}"


def test_xpander_page_does_not_embed_api_key(client, settings):
    assert settings.deepseek_api_key not in client.get("/xpander").text
    assert not re.search(r"sk-[A-Za-z0-9]{16,}", client.get("/xpander").text)


def test_xpander_static_assets_served(client):
    css = client.get("/static/css/xpander.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    js = client.get("/static/js/xpander.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    svg = client.get("/static/assets/xpander/hero-fallback.svg")
    assert svg.status_code == 200


def test_xpander_references_prepared_assets(client):
    html = client.get("/xpander").text
    assert "xpander_main_banner.png" in html
    assert "xpander/hero-fallback.svg" in html


def test_xpander_banner_asset_is_served(client):
    response = client.get("/static/assets/xpander/xpander_main_banner.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


# ----------------------------------------------------------------------
# API facade (reuses the Day 25 service + fakes)
# ----------------------------------------------------------------------
def test_xpander_api_create_and_list(client):
    created = client.post("/api/xpander/sessions", json={"title": "Вибрация на D"})
    assert created.status_code == 201
    body = created.json()
    assert body["title"] == "Вибрация на D"

    listed = client.get("/api/xpander/sessions")
    assert listed.status_code == 200
    ids = [s["id"] for s in listed.json()["sessions"]]
    assert body["id"] in ids


def test_xpander_api_send_message_and_state(client):
    created = client.post("/api/xpander/sessions", json={}).json()
    response = client.post(
        f"/api/xpander/sessions/{created['id']}/messages",
        json={"content": "Пробег 82 тысячи."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["role"] == "assistant"
    # Task-state summary is available to the UI, but the page never renders
    # the raw JSON.
    state = client.get(f"/api/xpander/sessions/{created['id']}/state")
    assert state.status_code == 200
    assert state.json()["state"]["vehicle"]["mileage_km"] == 82000


def test_xpander_api_grounded_answer_has_sources(client):
    created = client.post("/api/xpander/sessions", json={}).json()
    response = client.post(
        f"/api/xpander/sessions/{created['id']}/messages",
        json={"content": "Какое давление в шинах 205/55R16?"},
    )
    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["status"] == "answered"
    assert assistant["evidence"], "grounded answer must carry evidence"


def test_xpander_api_unknown_session(client):
    assert client.get("/api/xpander/sessions/nope").status_code == 404


def test_xpander_sessions_persist(client, settings):
    from app.services.day25 import ChatRepository

    created = client.post(
        "/api/xpander/sessions", json={"title": "Persist check"}
    ).json()
    client.post(
        f"/api/xpander/sessions/{created['id']}/messages",
        json={"content": "Пробег 92 тысячи."},
    )
    reopened = ChatRepository(settings.day25_chat_db_path)
    assert reopened.get_session(created["id"]).title == "Persist check"
    assert reopened.get_state(created["id"]).vehicle.mileage_km == 92000
