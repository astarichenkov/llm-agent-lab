"""Tests for the homepage and static frontend assets."""


def test_homepage_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    # Russian educational comparison UI
    assert "DeepSeek API — управление ответом модели" in html
    assert "Один и тот же запрос с разным уровнем контроля ответа через API" in html
    # default borscht prompt is pre-filled
    assert "Напиши список основных продуктов для приготовления борща на 4 порции." in html
    assert "Настройки ответа с ограничениями" in html
    assert "Режим формата ответа" in html
    assert "Требуемая структура JSON-ответа" in html
    assert "Максимальная длина ответа" in html
    assert "Условие завершения (stop sequence)" in html
    assert "Параметры контролируемого API-запроса" in html
    assert "Инструкция по структуре JSON" in html
    # element ids
    for el_id in [
        "compare-form", "message", "json-structure", "response-format",
        "max-tokens", "stop-sequence", "reset-json", "compare-button",
        "loading", "api-preview", "structure-instruction",
        "card-unrestricted", "card-controlled", "applied-settings",
        "applied-heading", "structure-used", "summary-section", "summary",
    ]:
        assert f'id="{el_id}"' in html
    assert "Без ограничений" in html
    assert "С ограничениями" in html
    assert "Что изменилось?" in html
    assert "Одно сравнение выполняет 2 запроса к DeepSeek API" in html
    # Tooltips + real API names
    assert "info-btn" in html
    assert "tooltip" in html
    assert '"type": "json_object"' in html
    assert "response_format" in html
    assert "max_tokens" in html
    assert "stop" in html
    assert "finish_reason" in html
    # default structure fields shown
    assert '"products"' in html and '"name"' in html and '"unit"' in html


def test_homepage_does_not_embed_api_key(client, settings):
    response = client.get("/")
    assert settings.deepseek_api_key not in response.text


def test_homepage_does_not_leak_credentials_patterns(client):
    html = client.get("/").text
    assert "Authorization" not in html
    assert "sk-" not in html
    assert "DEEPSEEK_API_KEY" not in html


def test_static_css_served(client):
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_static_js_served(client):
    response = client.get("/static/js/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_all_js_dom_references_exist_in_homepage(client):
    """Every element id app.js requires must exist in the rendered homepage."""
    import re
    from pathlib import Path

    js_path = Path(__file__).resolve().parents[1] / "app" / "static" / "js" / "app.js"
    referenced = set(re.findall(r'getElementById\("([^"]+)"\)', js_path.read_text(encoding="utf-8")))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"app.js references missing ids: {missing}"


def test_js_is_wrapped_for_dom_ready_and_missing_element_guard(client):
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "app" / "static" / "js" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "DOMContentLoaded" in js
    assert "requiredIds" in js
    assert "missing DOM elements" in js


def test_homepage_has_both_tabs(client):
    html = client.get("/").text
    assert 'id="tab-day2"' in html
    assert 'id="tab-day3"' in html
    assert 'id="panel-day2"' in html
    assert 'id="panel-day3"' in html
    assert "День 2 — Управление ответом" in html
    assert "День 3 — Способы рассуждения" in html
    assert 'src="/static/js/day3.js"' in html


def test_day3_required_elements_and_default_task(client):
    html = client.get("/").text
    # Day 2 controls must remain
    assert 'id="compare-form"' in html
    # default Day 3 task present
    assert "Анна, Борис и Виктор выступают с докладами" in html
    assert "Определи, в какой день выступает каждый." in html
    for el in [
        "d3-task", "d3-max-tokens", "d3-stop", "d3-ref-note",
        "d3-compare-section", "d3-compare-tbody",
        "d3-m1-run", "d3-m1-prompt", "d3-m1-answer", "d3-m1-finish",
        "d3-m2-run", "d3-m3-gen", "d3-m3-prompt", "d3-m3-use", "d3-m3-sent",
        "d3-m4-run", "d3-m4-prompt", "d3-m4-answer", "d3-conclusion",
    ]:
        assert f'id="{el}"' in html, f"missing Day3 id {el}"
    assert "5 API-запросов к DeepSeek" in html or "5 API-запросов" in html


def test_no_duplicate_ids_in_homepage(client):
    import re
    import collections

    ids = re.findall(r'id="([^"]+)"', client.get("/").text)
    dups = [k for k, v in collections.Counter(ids).items() if v > 1]
    assert not dups, f"duplicate ids: {dups}"


def test_day3_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set()
    for name in ("app.js", "day3.js", "day4.js"):
        src = (base / name).read_text(encoding="utf-8")
        referenced |= set(re.findall(r'getElementById\("([^"]+)"\)', src))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"referenced but missing ids: {missing}"


def test_homepage_has_day4_tab(client):
    html = client.get("/").text
    assert 'id="tab-day4"' in html
    assert 'id="panel-day4"' in html
    assert "День 4 — Температура" in html
    assert 'src="/static/js/day4.js"' in html


def test_day4_structure(client):
    html = client.get("/").text
    # common prompt once
    assert html.count("Объясни школьнику, что такое искусственный интеллект") == 1
    # sub-tabs
    for sid in ("d4-sub-0", "d4-sub-07", "d4-sub-12", "d4-sub-out"):
        assert f'id="{sid}"' in html
    # default temperatures editable
    for el, val in (("d4-0-temp", 'value="0"'), ("d4-07-temp", 'value="0.7"'), ("d4-12-temp", 'value="1.2"')):
        assert f'id="{el}"' in html and val in html
    # fair indicator + conclusions + rating controls
    assert 'id="d4-fair-indicator"' in html
    assert 'id="d4-conclusion"' in html
    assert 'id="d4-out-t0use"' in html and 'id="d4-out-t07use"' in html and 'id="d4-out-t12use"' in html
    for k in ("0", "07", "12"):
        for r in ("acc", "crea", "dive"):
            assert f'id="d4-r-{k}-{r}"' in html
    # Day 4 has no response_format / JSON-mode control
    assert 'response_format' not in html or "День 2" in html


def test_day4_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set()
    for name in ("day4.js",):
        src = (base / name).read_text(encoding="utf-8")
        referenced |= set(re.findall(r'getElementById\("([^"]+)"\)', src))
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day4.js references missing ids: {missing}"


def test_homepage_has_day5_tab(client):
    html = client.get("/").text
    assert 'id="tab-day5"' in html
    assert 'id="panel-day5"' in html
    assert "День 5 — Версии моделей" in html
    assert 'src="/static/js/day5.js"' in html
    # previous day tabs remain
    for t in ("tab-day2", "tab-day3", "tab-day4"):
        assert f'id="{t}"' in html


def test_day5_structure(client):
    html = client.get("/").text
    assert html.count("бинарный поиск работает быстрее линейного") == 1
    for sub in ("d5-sub-weak", "d5-sub-medium", "d5-sub-strong", "d5-sub-out"):
        assert f'id="{sub}"' in html
    for k in ("weak", "medium", "strong"):
        assert f'id="d5-{k}-model"' in html
        assert f'id="d5-{k}-metrics"' in html
        assert f'id="d5-{k}-answer"' in html
        for r in ("qual", "acc", "util"):
            assert f'id="d5-r-{k}-{r}"' in html
    assert 'id="d5-fair"' in html
    assert 'id="d5-compare-tbody"' in html
    assert 'id="d5-links"' in html
    assert 'id="d5-conclusion"' in html
    assert 'id="d5-run-all"' in html


def test_day5_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set()
    for name in ("app.js", "day3.js", "day4.js", "day5.js"):
        referenced |= set(re.findall(r'getElementById\("([^"]+)"\)', (base / name).read_text(encoding="utf-8")))
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"referenced missing ids: {missing}"
