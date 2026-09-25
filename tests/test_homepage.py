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
    import re

    html = client.get("/").text
    assert "Authorization" not in html
    assert "DEEPSEEK_API_KEY" not in html
    # A real DeepSeek key looks like sk-<long token>; the bare substring "sk-"
    # can legitimately appear inside identifiers (e.g. "d9-ask-control").
    assert not re.search(r"sk-[A-Za-z0-9]{16,}", html), "possible API key leak"


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


def test_homepage_has_day7_tab(client):
    html = client.get("/").text
    assert 'id="tab-day7"' in html
    assert 'id="panel-day7"' in html
    assert "День 7 — Сохранение контекста" in html
    assert 'src="/static/js/day7.js"' in html
    # all previous day tabs remain
    for t in ("tab-day2", "tab-day3", "tab-day4", "tab-day5", "tab-day6"):
        assert f'id="{t}"' in html


def test_homepage_has_day6_tab(client):
    html = client.get("/").text
    assert 'id="tab-day6"' in html
    assert 'id="panel-day6"' in html
    assert "День 6 — Первый агент" in html
    assert 'src="/static/js/day6.js"' in html
    # tab order: Day 6 before Day 7
    assert html.index('id="tab-day6"') < html.index('id="tab-day7"')


def test_day6_structure(client):
    html = client.get("/").text
    for el in (
        "d6-agent-name", "d6-agent-provider", "d6-agent-model",
        "d6-agent-max-tokens", "d6-agent-thinking",
        "d6-input", "d6-send", "d6-loading", "d6-error",
        "d6-answer", "d6-meta",
    ):
        assert f'id="{el}"' in html, f"missing Day6 id {el}"
    # stateless message must be visible to the student
    assert "Контекст между" in html


def test_day6_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day6.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day6.js references missing ids: {missing}"


def test_day7_structure(client):
    html = client.get("/").text
    for el in (
        "d7-agent-name", "d7-agent-provider", "d7-agent-model",
        "d7-agent-max-tokens", "d7-agent-thinking",
        "d7-context-count", "d7-messages", "d7-input", "d7-send",
        "d7-clear", "d7-loading", "d7-error",
    ):
        assert f'id="{el}"' in html, f"missing Day7 id {el}"


def test_day7_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day7.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day7.js references missing ids: {missing}"


def test_homepage_has_day8_tab(client):
    html = client.get("/").text
    assert 'id="tab-day8"' in html
    assert 'id="panel-day8"' in html
    assert "День 8 — Работа с токенами" in html
    assert 'src="/static/js/day8.js"' in html
    # tab order: Day 7 before Day 8
    assert html.index('id="tab-day7"') < html.index('id="tab-day8"')


def test_day8_structure(client):
    html = client.get("/").text
    for el in (
        "d8-model", "d8-context-window", "d8-max-output",
        "d8-scenario-short", "d8-scenario-long", "d8-fill-context", "d8-clear",
        "d8-messages", "d8-input", "d8-send", "d8-loading", "d8-error",
        "d8-sys", "d8-hist", "d8-cur", "d8-est-input",
        "d8-api-input", "d8-api-output", "d8-api-total", "d8-api-cache", "d8-cost",
        "d8-cum-requests", "d8-cum-messages", "d8-cum-history",
        "d8-cum-input", "d8-cum-output", "d8-cum-total", "d8-cum-cost",
        "d8-rows", "d8-totals", "d8-chart",
        "d8-ov-estimate", "d8-ov-limit", "d8-ov-excess", "d8-ov-status",
        "d8-overflow-send", "d8-ov-provider",
    ):
        assert f'id="{el}"' in html, f"missing Day8 id {el}"
    # both overflow concepts are explained on the tab
    assert "context window" in html.lower()
    assert "finish_reason" in html or "output" in html.lower()


def test_day8_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day8.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day8.js references missing ids: {missing}"


def test_homepage_has_day9_tab(client):
    html = client.get("/").text
    assert 'id="tab-day9"' in html
    assert 'id="panel-day9"' in html
    assert "День 9 — Сжатие истории" in html
    assert 'src="/static/js/day9.js"' in html
    # tab order: Day 8 before Day 9
    assert html.index('id="tab-day8"') < html.index('id="tab-day9"')


def test_day9_structure(client):
    html = client.get("/").text
    for el in (
        "d9-model", "d9-recent", "d9-batch", "d9-cycles",
        "d9-mode-full", "d9-mode-compressed", "d9-recent-input", "d9-batch-input",
        "d9-seed", "d9-ask-control", "d9-clear", "d9-loading",
        "d9-messages", "d9-input", "d9-send", "d9-error", "d9-compression-note",
        "d9-full-count", "d9-summarized-count", "d9-recent-count", "d9-pending-count",
        "d9-full-tokens", "d9-summary-tokens", "d9-recent-tokens",
        "d9-compressed-tokens", "d9-saved", "d9-saved-percent", "d9-cycles-diag",
        "d9-summary-details", "d9-summary-text",
        "d9-rows", "d9-chart",
        "d9-answer-full", "d9-answer-compressed",
    ):
        assert f'id="{el}"' in html, f"missing Day9 id {el}"


def test_day9_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day9.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day9.js references missing ids: {missing}"


def test_no_duplicate_ids_after_day9(client):
    import re
    import collections

    ids = re.findall(r'id="([^"]+)"', client.get("/").text)
    dups = [k for k, v in collections.Counter(ids).items() if v > 1]
    assert not dups, f"duplicate ids: {dups}"


def test_homepage_has_day10_tab(client):
    html = client.get("/").text
    assert 'id="tab-day10"' in html
    assert 'id="panel-day10"' in html
    assert "День 10 — Управление контекстом" in html
    assert 'src="/static/js/day10.js"' in html
    # tab order: Day 9 before Day 10
    assert html.index('id="tab-day9"') < html.index('id="tab-day10"')


def test_day10_structure(client):
    html = client.get("/").text
    for el in (
        "d10-s-sliding", "d10-s-sticky", "d10-s-branching",
        "d10-strategy-name", "d10-model", "d10-history-count", "d10-active-branch",
        "d10-window-input", "d10-recent-input", "d10-update-facts",
        "d10-seed-branches", "d10-demo-run", "d10-compare-run", "d10-clear",
        "d10-loading", "d10-progress", "d10-note",
        "d10-branch-panel", "d10-checkpoint", "d10-branch-list",
        "d10-branch-name", "d10-branch-create",
        "d10-messages", "d10-input", "d10-send", "d10-error",
        "d10-sys", "d10-facts-tokens", "d10-recent-tokens", "d10-cur",
        "d10-est-input", "d10-api-input", "d10-api-output", "d10-api-total",
        "d10-facts-extract", "d10-cost",
        "d10-cum-requests", "d10-cum-extractions", "d10-cum-input",
        "d10-cum-output", "d10-cum-total", "d10-cum-extract",
        "d10-sliding-panel", "d10-sw-summary", "d10-full-count",
        "d10-sent-count", "d10-dropped-count", "d10-dropped-list",
        "d10-facts-panel", "d10-facts-count", "d10-facts-recent",
        "d10-facts-block", "d10-facts-json", "d10-facts-error",
        "d10-compare-rows",
    ):
        assert f'id="{el}"' in html, f"missing Day10 id {el}"
    # the three strategies are all visible
    assert "Sliding Window" in html
    assert "Sticky Facts" in html
    assert "Branching" in html


def test_day10_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day10.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day10.js references missing ids: {missing}"


def test_days_1_to_5_are_hidden_from_navigation(client):
    """Days 1-5 stay in the DOM/API but must not be visible in the UI."""
    html = client.get("/").text
    for day in ("day2", "day3", "day4", "day5"):
        assert f'id="tab-{day}"' in html, f"{day} tab must still exist in markup"
    # the hidden tabs carry the day-hidden class and the hidden attribute
    assert 'class="tab-btn day-hidden" id="tab-day2"' in html
    assert 'class="tab-btn day-hidden" id="tab-day5"' in html
    # the Day 2 main panel is hidden too
    assert 'class="container hidden" id="panel-day2"' in html

    from pathlib import Path

    app_js = (
        Path(__file__).resolve().parents[1] / "app" / "static" / "js" / "app.js"
    ).read_text(encoding="utf-8")
    main_tabs = app_js.split("MAIN_TABS =")[1].split(";")[0]
    # Legacy tabs stay switchable in code (so their panels are still hidden),
    # but the active module is now Week 4 / Day 16.
    for hidden in ("day2", "day3", "day4", "day5"):
        assert f'"{hidden}"' in main_tabs
    assert '"day16"' in main_tabs
    # default landing tab is the current module (Day 16)
    assert 'switchMainTab("day16")' in app_js


def test_days_6_to_10_are_hidden_from_navigation(client):
    """Days 6-15 are removed from the visible UI (Week 4 is shown instead)."""
    html = client.get("/").text
    import re

    for day in (
        "day6", "day7", "day8", "day9", "day10",
        "day11", "day12", "day13", "day14", "day15",
    ):
        match = re.search(
            r'<button[^>]*id="tab-' + day + r'"[^>]*>', html
        )
        assert match, f"missing tab button for {day}"
        assert "day-hidden" in match.group(0), f"{day} must be hidden"
        assert "hidden" in match.group(0), f"{day} must carry the hidden attribute"


def test_no_duplicate_ids_after_day10(client):
    import re
    import collections

    ids = re.findall(r'id="([^"]+)"', client.get("/").text)
    dups = [k for k, v in collections.Counter(ids).items() if v > 1]
    assert not dups, f"duplicate ids: {dups}"


def test_homepage_has_day11_tab(client):
    html = client.get("/").text
    assert 'id="tab-day11"' in html
    assert 'id="panel-day11"' in html
    assert "День 11 — Модель памяти агента" in html
    assert 'src="/static/js/day11.js"' in html
    # tab order: Day 10 before Day 11
    assert html.index('id="tab-day10"') < html.index('id="tab-day11"')


def test_day11_structure(client):
    html = client.get("/").text
    for el in (
        "d11-strategy", "d11-window-input", "d11-use-memory", "d11-classify",
        "d11-include-lt", "d11-include-working",
        "d11-model", "d11-session-id",
        "d11-demo-run", "d11-seed", "d11-new-session",
        "d11-clear-st", "d11-clear-working", "d11-clear-lt",
        "d11-loading", "d11-progress", "d11-note",
        "d11-messages", "d11-input", "d11-send", "d11-error",
        "d11-decision-status", "d11-decision-list", "d11-decision-error",
        "d11-save-st", "d11-save-working", "d11-save-lt", "d11-save-summary",
        "d11-st-count", "d11-st-list",
        "d11-working-count", "d11-working-block", "d11-working-json",
        "d11-lt-count", "d11-lt-block", "d11-lt-json",
        "d11-ctx-summary", "d11-ctx-layers", "d11-ctx-total", "d11-ctx-sent",
        "d11-ctx-dropped", "d11-ctx-dropped-list", "d11-ctx-blocks",
    ):
        assert f'id="{el}"' in html, f"missing Day11 id {el}"
    # the three memory layers are all visible
    assert "SHORT-TERM MEMORY" in html
    assert "WORKING MEMORY" in html
    assert "LONG-TERM MEMORY" in html
    assert "Решение о сохранении" in html


def test_day11_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day11.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day11.js references missing ids: {missing}"


def test_no_duplicate_ids_after_day11(client):
    import re
    import collections

    ids = re.findall(r'id="([^"]+)"', client.get("/").text)
    dups = [k for k, v in collections.Counter(ids).items() if v > 1]
    assert not dups, f"duplicate ids: {dups}"


def test_homepage_has_day12_tab(client):
    html = client.get("/").text
    assert 'id="tab-day12"' in html
    assert 'id="panel-day12"' in html
    assert "День 12 — Персонализация ассистента" in html
    assert 'src="/static/js/day12.js"' in html
    # tab order: Day 11 before Day 12
    assert html.index('id="tab-day11"') < html.index('id="tab-day12"')


def test_day12_structure(client):
    html = client.get("/").text
    for el in (
        "d12-active-style", "d12-active-badges", "d12-presets",
        "d12-name", "d12-style",
        "d12-f-structured", "d12-f-lists", "d12-f-examples", "d12-f-code",
        "d12-c-no-emoji", "d12-c-no-intro", "d12-c-no-lists", "d12-lang",
        "d12-save", "d12-save-status", "d12-instructions",
        "d12-use-memory", "d12-classify",
        "d12-messages", "d12-input", "d12-send", "d12-loading", "d12-error",
        "d12-note", "d12-context-summary", "d12-context-layers",
        "d12-memory-summary", "d12-memory-list",
        "d12-demo-prompt", "d12-compare-run", "d12-compare-results",
    ):
        assert f'id="{el}"' in html, f"missing Day12 id {el}"
    # the three preference groups are visible
    assert "Стиль ответа" in html
    assert "Формат ответа" in html
    assert "Ограничения" in html
    assert "Профиль пользователя" in html


def test_day12_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day12.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day12.js references missing ids: {missing}"


def test_no_duplicate_ids_after_day12(client):
    import re
    import collections

    ids = re.findall(r'id="([^"]+)"', client.get("/").text)
    dups = [k for k, v in collections.Counter(ids).items() if v > 1]
    assert not dups, f"duplicate ids: {dups}"


def test_homepage_has_day13_tab(client):
    html = client.get("/").text
    assert 'id="tab-day13"' in html
    assert 'id="panel-day13"' in html
    assert "День 13 — Состояние задачи" in html
    assert 'src="/static/js/day13.js"' in html
    # tab order: Day 12 before Day 13
    assert html.index('id="tab-day12"') < html.index('id="tab-day13"')


def test_day13_structure(client):
    html = client.get("/").text
    for el in (
        "d13-goal", "d13-create", "d13-reset",
        "d13-loading", "d13-error", "d13-note",
        "d13-fsm",
        "d13-stage-planning", "d13-stage-execution",
        "d13-stage-validation", "d13-stage-done",
        "d13-stage", "d13-step", "d13-expected", "d13-status",
        "d13-pause", "d13-resume", "d13-allowed",
        "d13-plan-list", "d13-completed-list", "d13-raw",
        "d13-messages", "d13-input", "d13-send", "d13-context-block",
    ):
        assert f'id="{el}"' in html, f"missing Day13 id {el}"
    # the FSM stages and the pause/resume actions are visible
    assert "Planning" in html
    assert "Execution" in html
    assert "Validation" in html
    assert "Done" in html
    assert "Pause" in html
    assert "Resume" in html
    assert "Show raw state" not in html  # Russian UI label is used
    assert "Показать raw state" in html


def test_day13_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day13.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day13.js references missing ids: {missing}"


def test_no_duplicate_ids_after_day13(client):
    import re
    import collections

    ids = re.findall(r'id="([^"]+)"', client.get("/").text)
    dups = [k for k, v in collections.Counter(ids).items() if v > 1]
    assert not dups, f"duplicate ids: {dups}"


def test_homepage_has_day14_tab(client):
    html = client.get("/").text
    assert 'id="tab-day14"' in html
    assert 'id="panel-day14"' in html
    assert "День 14 — Инварианты" in html
    assert 'src="/static/js/day14.js"' in html
    # tab order: Day 13 before Day 14
    assert html.index('id="tab-day13"') < html.index('id="tab-day14"')


def test_day14_structure(client):
    html = client.get("/").text
    for el in (
        "d14-invariants",
        "d14-result", "d14-result-status", "d14-result-text", "d14-conflicts",
        "d14-messages", "d14-input", "d14-send", "d14-clear",
        "d14-loading", "d14-error", "d14-note", "d14-context-block", "d14-raw",
    ):
        assert f'id="{el}"' in html, f"missing Day14 id {el}"
    # the invariants block is visually separated from the chat
    assert "ACTIVE INVARIANTS" in html
    assert "Диалог" in html
    assert "Результат проверки запроса" in html


def test_day14_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day14.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day14.js references missing ids: {missing}"


def test_no_duplicate_ids_after_day14(client):
    import re
    import collections

    ids = re.findall(r'id="([^"]+)"', client.get("/").text)
    dups = [k for k, v in collections.Counter(ids).items() if v > 1]
    assert not dups, f"duplicate ids: {dups}"


def test_homepage_has_day15_tab(client):
    html = client.get("/").text
    assert 'id="tab-day15"' in html
    assert 'id="panel-day15"' in html
    assert "День 15 — Контроль переходов" in html
    assert 'src="/static/js/day15.js"' in html
    # tab order: Day 14 before Day 15
    assert html.index('id="tab-day14"') < html.index('id="tab-day15"')


def test_day15_structure(client):
    html = client.get("/").text
    for el in (
        "d15-current-state", "d15-current-step", "d15-expected-action",
        "d15-plan-approved", "d15-validation-passed", "d15-paused",
        "d15-lifecycle",
        "d15-state-planning", "d15-state-plan_approval",
        "d15-state-execution", "d15-state-validation", "d15-state-done",
        "d15-allowed-title", "d15-allowed-list",
        "d15-goal", "d15-create", "d15-prepare", "d15-approve",
        "d15-start-execution", "d15-run-validation", "d15-back-execution",
        "d15-pass", "d15-fail",
        "d15-complete", "d15-pause", "d15-resume", "d15-reset", "d15-loading",
        "d15-skip-approval", "d15-skip-execution", "d15-skip-done",
        "d15-decision", "d15-decision-status", "d15-decision-text",
        "d15-decision-reason", "d15-error", "d15-note",
        "d15-scenario-happy", "d15-scenario-skip-planning",
        "d15-scenario-skip-validation", "d15-scenario-failed-validation",
        "d15-scenario-pause-resume", "d15-log",
        "d15-messages", "d15-input", "d15-propose", "d15-send", "d15-raw",
    ):
        assert f'id="{el}"' in html, f"missing Day15 id {el}"
    assert "Current Task State" in html
    assert "Lifecycle" in html
    assert "Allowed transitions" in html
    assert "Transition Log" in html
    assert "PLAN APPROVAL" in html


def test_day15_js_dom_references_exist(client):
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
    referenced = set(re.findall(
        r'\$\("([^"]+)"\)',
        (base / "day15.js").read_text(encoding="utf-8"),
    ))
    assert referenced
    present = set(re.findall(r'id="([^"]+)"', client.get("/").text))
    missing = sorted(referenced - present)
    assert not missing, f"day15.js references missing ids: {missing}"


def test_no_duplicate_ids_after_day15(client):
    import re
    import collections

    ids = re.findall(r'id="([^"]+)"', client.get("/").text)
    dups = [k for k, v in collections.Counter(ids).items() if v > 1]
    assert not dups, f"duplicate ids: {dups}"
