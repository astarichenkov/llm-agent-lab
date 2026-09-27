# Week 4 / Day 19 — Композиция MCP-инструментов

## Goal

Day 19 показывает, как из **нескольких MCP-инструментов** собрать один
автоматический pipeline:

```
пользовательский запрос
        │
        ▼
   search_logs     — получает данные из VictoriaLogs
        │
        ▼
   analyze_logs    — структурированный анализ (LLM)
        │
        ▼
   save_report     — детерминированно сохраняет артефакты
        │
        ▼
   финальный ответ
```

Пользователь **не указывает порядок tools вручную** и не вызывает их по
одному. Backend сам выполняет шаги в правильном порядке и передаёт результат
каждого tool следующему.

## Отличие Day 19 от Day 17

| Day 17 | Day 19 |
| --- | --- |
| Один MCP tool `search_logs`. | Три MCP tools в одной цепочке. |
| LLM сам вызывает tool через function calling. | Backend детерминированно ведёт цепочку `search → analyze → save`. |
| Результат — текстовый ответ. | Результат — структурированный анализ + файловые артефакты. |
| Нет LLM-анализа найденных логов. | `analyze_logs` делает structured LLM-анализ. |

Day 17 остаётся полностью рабочим и не изменяется.

## Отличие Day 19 от Day 18

| Day 18 | Day 19 |
| --- | --- |
| Долгоживущий scheduler, периодические запуски. | Один явный запуск pipeline по запросу. |
| Хранит **только агрегаты** в SQLite. | Хранит **артефакты** `raw.jsonl` / `analysis.md` / `metadata.json`. |
| LLM в scheduled-запусках не вызывается. | LLM вызывается в `analyze_logs` и для финального ответа. |
| Мониторинг как фон. | Разовый анализ проблемы. |

Day 19 **не использует** Day 18 monitoring DB и работает независимо.

## Architecture / files

| File | Responsibility |
| --- | --- |
| `app/schemas/day19.py` | Pydantic-модели, лимиты анализа, имена артефактов, ответы API/MCP. |
| `app/services/day19/tools.py` | `Day19Toolset`: реальная реализация трёх tools. |
| `app/services/day19/analysis.py` | Ограничения, prompt, парсинг и валидация structured analysis. |
| `app/services/day19/artifacts.py` | Безопасное хранилище артефактов + deterministic Markdown renderer. |
| `app/services/day19/runner.py` | `Day19PipelineService`: оркестратор цепочки и финальный ответ. |
| `app/services/mcp/pipeline/server.py` | FastMCP server `Pipeline MCP`: `search_logs`, `analyze_logs`, `save_report`. |
| `app/services/mcp/pipeline/client.py` | MCP-клиент к Pipeline MCP через in-memory transport. |
| `app/config.py` | `day19_artifact_root`. |
| `app/main.py` | Конструирование `Day19PipelineService`. |
| `app/api/routes.py` | `POST /api/week4/day19/pipeline`, безопасная выдача артефактов. |
| `app/templates/_day19.html`, `app/static/js/day19.js` | UI Day 19. |
| `tests/test_day19.py` | Unit/API/MCP/artifact/security/UI тесты Day 19. |

MCP SDK остаётся на **1.x** (`mcp[cli]>=1.0,<2.0`). Миграции на 2.x нет.

## Почему отдельный Pipeline MCP server

Day 19 — это отдельный логический MCP-сервер (`Pipeline MCP`), а не набор
tools поверх Day 17/18:

* композиция инструментов видна как отдельная граница;
* Day 20 (несколько независимых MCP servers, multi-server router) можно
  добавить, не трогая Day 17/18;
* состояние pipeline и файловая запись живут в долгоживущем процессе
  приложения, поэтому используется **in-memory transport** (тот же приём,
  что в Day 18): настоящий MCP `initialize → tools/list → tools/call`, но без
  short-lived subprocess.

`search_logs` внутри Day 19 **не дублирует HTTP-код VictoriaLogs**: он
переиспользует:

```
Day 17 VictoriaLogsClient      (app/services/mcp/victorialogs/client.py)
Day 17 build_logsql            (app/services/mcp/victorialogs/query_builder.py)
Day 17 sanitizer               (app/services/mcp/victorialogs/sanitizer.py)
Day 17 SearchLogsInput/schemas (app/services/mcp/victorialogs/schemas.py)
```

## Три MCP tools и их схемы

### `search_logs` (получает данные)

| Field | Type | Required | Constraints |
| --- | --- | --- | --- |
| `service` | string | yes | 1–200 |
| `since_minutes` | integer | no (30) | 1–360 |
| `level` | string \| null | no | ≤ 32 |
| `text_contains` | string \| null | no | ≤ 200 |
| `limit` | integer | no (100) | 1–500 |

Возвращает нормализованные и уже санитизированные строки:

```json
{
  "service": "application",
  "since_minutes": 30,
  "count": 27,
  "limit": 100,
  "truncated": false,
  "malformed_lines": 0,
  "logs": [
    {"timestamp": "...", "message": "...", "fields": {"level": "ERROR"}}
  ]
}
```

### `analyze_logs` (обрабатывает данные)

| Field | Type | Required |
| --- | --- | --- |
| `service` | string | yes |
| `logs` | NormalizedLog[] | yes (минимум пустой список) |
| `question` | string | yes |
| `since_minutes` / `level` / `start` / `end` / `logs_received` | metadata | no |

Tool **не ходит повторно в VictoriaLogs** — анализирует ровно переданные
строки. Возвращает структурированный результат:

```json
{
  "summary": "За период наблюдались...",
  "error_groups": [
    {"pattern": "connection timeout", "count": 12, "examples": ["..."]}
  ],
  "possible_causes": ["..."],
  "notable_patterns": ["..."],
  "recommended_checks": ["..."],
  "logs_analyzed": 27,
  "analysis_truncated": false
}
```

### `save_report` (сохраняет данные и анализ)

Детерминированный tool, LLM не используется. Принимает `run_id`, metadata,
`logs` и `analysis`; пишет артефакты и возвращает их имена.

## Pipeline sequence и data handoff

`Day19PipelineService.run()` детерминированно выполняет три реальных MCP
`tools/call`:

```
PipelineMCPClient.call_tool("search_logs", filters)
        │  search payload (sanitized logs)
        ▼
PipelineMCPClient.call_tool("analyze_logs", {service, question, logs: <search logs>})
        │  structured analysis
        ▼
PipelineMCPClient.call_tool("save_report", {run_id, logs: <search logs>, analysis: <analysis>})
```

Результат первого tool **реально** подставляется в аргументы второго, а
результат второго — в аргументы третьего. Это проверяется тестами
(`test_pipeline_passes_exact_logs_and_analysis`, `test_pipeline_trace_shows_data_handoff`).

### Выбранная orchestration strategy (Вариант B)

Используется **детерминированная backend-оркестрация** (Вариант B) при том,
что LLM участвует в `analyze_logs` и в финальном ответе. Причины:

1. Главное требование задания — реальная передача данных между
   инструментами и строгий порядок `search → analyze → save`. Большие массивы
   санитизированных логов невозможно надёжно прогонять через аргументы
   tool-call LLM, не потеряв данные и не раздув контекст.
2. Надёжность важнее «магии»: детерминированная цепочка даёт байт-точную
   передачу, явное прерывание при ошибке и достоверный trace.
3. LLM при этом реально используется: `analyze_logs` делает структурированный
   анализ с учётом вопроса пользователя, а финальный компактный LLM-вызов
   формирует естественный ответ (с детерминированным fallback).

Порядок гарантируется кодом оркестратора, а не поведением модели. При ошибке
шага следующие шаги не выполняются.

## Structured analysis: prompt и валидация

`analyze_logs` использует отдельный system prompt
(`app/services/day19/analysis.py`). Он требует:

* анализировать только предоставленные строки;
* не придумывать отсутствующие факты;
* разделять observations и hypotheses;
* `possible_causes` формулировать как предположения;
* не выводить секреты;
* вернуть строго заданную JSON-схему.

Ответ модели **не доверяется напрямую**: он парсится
(`app/services/structured_json.py`) и валидируется Pydantic-моделью
`LogAnalysis`. Любая ошибка (невалидный JSON, нарушение схемы, пустой ответ,
timeout, API error) превращается в контролируемую `AnalysisError` →
`{"error": ..., "error_kind": "analysis"}`. После ошибки `save_report` не
выполняется.

## Analysis limits

| Limit | Value | Где |
| --- | --- | --- |
| `MAX_ANALYSIS_LOGS` | 100 | `app/schemas/day19.py` |
| `MAX_LOG_MESSAGE_CHARS` | 2000 | `app/schemas/day19.py` |
| `MAX_ANALYSIS_INPUT_CHARS` | 200 000 | `app/schemas/day19.py` |
| `MAX_ERROR_GROUPS` | 25 | `app/schemas/day19.py` |
| `MAX_ANALYSIS_ITEMS` | 25 | `app/schemas/day19.py` |

Если `search_logs` вернул больше строк, анализируются первые
`MAX_ANALYSIS_LOGS` (характерное для VictoriaLogs упорядочение — от новых к
старым), каждая строка обрезается до `MAX_LOG_MESSAGE_CHARS`. Флаг
`analysis_truncated` попадает и в ответ, и в `metadata.json`. При этом
`raw.jsonl` содержит **все** санитизированные строки, полученные `search_logs`
(в пределах `limit`).

## Артефакты

```
data/day19/runs/<run_id>/
    raw.jsonl       # по одному нормализованному sanitized JSON-объекту на строку
    analysis.md     # deterministic Markdown renderer из structured analysis
    metadata.json   # machine-readable метаданные запуска
```

`data/` — gitignored, поэтому stage-артефакты не попадают в git.

`analysis.md` формирует backend-renderer, а не LLM: модель возвращает только
JSON, а формат файла стабилен и пригоден для diff.

Пример `metadata.json`:

```json
{
  "run_id": "run_20260928_120530_ab12cd",
  "created_at": "2026-09-28T12:05:31Z",
  "pipeline_id": "run_20260928_120530_ab12cd",
  "service": "application",
  "since_minutes": 30,
  "level": "ERROR",
  "logs_received": 27,
  "logs_analyzed": 27,
  "truncated": false,
  "analysis_truncated": false,
  "model": "deepseek-v4-flash",
  "duration_ms": 4321,
  "status": "completed",
  "artifacts": {
    "raw": "raw.jsonl",
    "analysis": "analysis.md",
    "metadata": "metadata.json"
  }
}
```

## Security / privacy

* `search_logs` возвращает уже санитизированные строки (Day 17 sanitizer).
* `save_report` дополнительно санитизирует `logs` и `metadata` перед записью
  (защита в глубину, даже для прямых вызовов tool).
* В артефактах не должно быть `Authorization`, bearer-токенов, cookies,
  API-ключей, `password`, `token`, VictoriaLogs URL и credentials. Это
  проверяется тестом `test_secrets_are_sanitized_before_analyze_and_save`.
* LLM не получает неограниченный объём: действуют лимиты анализа.

## Маскирование данных (галка «Маскировать данные»)

Дополнительный слой приватности поверх sanitizer-а. При включённой галке
pipeline маскирует идентификаторы **до** отправки в LLM, в финальный ответ,
в trace и в артефакты:

* URL (`https://...`) → `[URL]`;
* `host:port`, IPv4-адреса и e-mail → `[MASKED]` / `[IP]` / `[EMAIL]`;
* значения полей-идентификаторов (`service`, `container`, `host`, `pod`,
  `namespace`, `node`, `cluster`, `url`, `endpoint`, ...) → `[MASKED]`;
* literal-имена сервиса/контейнеров, собранные из запроса и из `fields`
  найденных строк.

Маскирование одностороннее: восстановить исходные значения из вывода
нельзя. По умолчанию галка выключена, поведение pipeline не меняется.
Маскированные строки — это то, что реально уходит в `analyze_logs` и
`save_report`, поэтому исходные идентификаторы не попадают ни в
`raw.jsonl`, ни в `analysis.md`, ни в `metadata.json`.

## Path security

Путь к отчёту **не** определяется пользовательским вводом:

* `run_id` генерирует backend (`run_YYYYmmdd_HHMMSS_<6hex>`);
* `run_id` валидируется по `[A-Za-z0-9_-]{1,64}` — это отсекает `../`,
  абсолютные пути, drive letters и произвольные имена;
* итоговый resolved path дополнительно проверяется на нахождение внутри
  `data/day19/runs/`;
* endpoint артефактов отдаёт только три известных имени и только для
  существующего `run_id`.

Тесты: `test_validate_run_id_rejects_unsafe_values`,
`test_artifact_store_rejects_traversal`.

## Failure behaviour

| Failure | Поведение |
| --- | --- |
| `search_logs` упал | `analyze_logs` и `save_report` не вызываются, в trace `skipped`. |
| `analyze_logs` упал | `save_report` не вызывается, в trace `skipped`. |
| `save_report` упал | pipeline `failed`; артефакты могут не создаться. |
| `search_logs` вернул **0 строк** | pipeline остаётся `completed`, но `analyze_logs` возвращает детерминированный «логи не найдены» **без вызова LLM**; `raw.jsonl` пуст (это корректно), `analysis.md`/`metadata.json` создаются. |

Все ошибки контролируемые и не содержат stack trace.

## API

```
POST /api/week4/day19/pipeline
{
  "service": "application",
  "level": "ERROR",
  "since_minutes": 30,
  "limit": 100,
  "question": "Найди основные проблемы и возможные причины"
}
```

Ответ содержит `pipeline_id`, `status`, `answer`, `search`, `analysis`,
`artifacts`, `trace`, `error`.

Безопасная выдача артефактов:

```
GET /api/week4/day19/runs/{run_id}/artifacts/{raw.jsonl|analysis.md|metadata.json}
```

## UI

* Week 4 → Day 19 — форма: Service / Level / Period / Limit / Analysis
  question (+ optional text) и галка **Маскировать данные**.
* Кнопка **Run pipeline**.
* Визуализация шагов: `✓ search_logs (N logs received)`,
  `✓ analyze_logs (K groups found)`, `✓ save_report (3 artifacts created)`.
* При ошибке: `✗ analyze_logs`, `- save_report skipped`.
* Список артефактов и preview `analysis.md` / `metadata.json`.
* `raw.jsonl` доступен как файл, но не отображается по умолчанию.
* Полный pipeline trace (backend-generated, не выдуманный frontend).

Day 16, Day 17, Day 18 остаются активными. Day 20 остаётся disabled.

## Manual verification

Безопасный smoke против реального stage:

```bash
pytest -m integration   # опциональный real-API smoke (если сконфигурирован)
```

Ручной прогон (не выводить сырые логи в отчёт):

```
service = application
level = ERROR
since = 15 минут
limit = 10–20
```

Pipeline должен выполнить `search → analyze → save`. В отчёт выводить только:
search count, analysis group count, имена артефактов, статус pipeline.

В окружении разработки реальный stage может быть недоступен (отсутствует
internal CA bundle / нет сетевого доступа). В этом случае все unit-тесты
работают на моках и не зависят от stage.

## Video scenario

1. Открыть Week 4 → Day 19.
2. Выбрать `service = application`, `level = ERROR`, period = 1/15/30 min
   (короткий период 1 min удобен для быстрой проверки).
3. Ввести `Найди основные типы ошибок и возможные причины`.
4. Run pipeline.
5. Показать `search_logs ✓` и число logs.
6. Показать `analyze_logs ✓` и число error groups.
7. Показать `save_report ✓` и пути артефактов.
8. Показать pipeline trace.
9. Открыть `analysis.md`.
10. Открыть `metadata.json`.
11. Показать наличие `raw.jsonl`, не раскрывая чувствительные логи.
12. В коде показать три tool registrations.
13. Показать orchestration sequence.

Проговорить:

> Day 17 вызывал один MCP tool. Day 19 автоматически композирует несколько
> MCP tools: первый получает данные, второй их анализирует, третий сохраняет
> результат.
