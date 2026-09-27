# Week 4 / Day 18 — Планировщик и фоновые задачи

## Goal

Превратить разовый поиск Day 17 (`search_logs`) в **постоянно работающий
агент**: пользователь создаёт monitoring job для stage-логов VictoriaLogs,
долгоживущий **scheduler** периодически выполняет детерминированный сбор
данных (без LLM), результаты-агрегаты сохраняются в **SQLite**, а приложение
показывает историю запусков и сводку.

```
Browser / Agent
      │
      ▼
MCP tool: start_monitoring            (app/services/mcp/monitoring/server.py)
      │
      ▼
MonitoringService                     (app/services/day18/service.py)
      │
      ├──► SQLite repository          (app/services/day18/repository.py)
      │
      └──► MonitoringScheduler        (app/services/day18/scheduler.py)
                   │  every N seconds
                   ▼
          Day 17 VictoriaLogs search  (VictoriaLogsClient + build_logsql)
                   │
                   ▼
             monitor_runs (SQLite)
```

Ключевой принцип: **каждый scheduled run — детерминированный**:
`scheduler → VictoriaLogs → aggregate → SQLite`. LLM внутри запуска
**не вызывается**.

## Architecture / files

| File | Responsibility |
| --- | --- |
| `app/schemas/day18.py` | Pydantic-модели, whitelist интервалов и lookback, статусы, ответы API/MCP. |
| `app/services/day18/repository.py` | SQLite-шлюз: `monitor_jobs`, `monitor_runs`, UTC-метки, новая connection на операцию. |
| `app/services/day18/scheduler.py` | Тонкая обёртка над APScheduler `AsyncIOScheduler` (register/unregister/lifecycle). |
| `app/services/day18/service.py` | `MonitoringService`: создание job, первый inline-run, scheduled run, агрегация, stop, restart recovery. |
| `app/services/day18/agent.py` | `Day18MonitoringAgentService`: агентный tool-calling loop только с monitoring-инструментами. |
| `app/services/mcp/monitoring/server.py` | FastMCP monitoring server: `start_monitoring`, `get_monitoring_status`, `get_monitoring_summary`, `stop_monitoring`. |
| `app/services/mcp/monitoring/client.py` | MCP-клиент к monitoring server через in-memory transport. |
| `app/main.py` | Конструирование сервисов + FastAPI `lifespan` (start/stop scheduler, restore jobs). |
| `app/api/routes.py` | REST-dashboard + `POST /api/week4/day18/chat`. |
| `app/templates/_day18.html`, `app/static/js/day18.js` | UI Day 18. |
| `tests/test_day18.py` | Unit/API/MCP/agent/UI тесты Day 18. |

MCP SDK остаётся на **1.x** (`mcp[cli]>=1.0,<2.0`). Миграции на 2.x нет.

## Scheduler library и lifecycle

* Библиотека: **APScheduler 3.x** (`AsyncIOScheduler`), добавлена в
  `requirements.txt`.
* Scheduler создаётся в `create_app()` вместе с `MonitoringService`.
* **Запускается и останавливается** в FastAPI `lifespan`
  (`app/main.py`):

```python
@asynccontextmanager
async def lifespan(application):
    service = application.state.monitoring_service
    service.start_scheduler()          # startup
    service.restore_active_jobs()      # restart recovery
    try:
        yield
    finally:
        service.shutdown_scheduler()   # shutdown
```

* Настройки job по умолчанию: `max_instances=1`, `coalesce=True`,
  `misfire_grace_time=5s`. Это защищает от overlapping-запусков (особенно
  важно при 10-секундном demo-интервале).
* Scheduler **не живёт внутри MCP-подпроцесса**: он всегда в процессе
  FastAPI, поэтому переживает отдельные MCP-вызовы и корректно
  останавливается вместе с приложением.

## MCP tools Day 18

Monitoring-инструменты реализованы как **отдельный MCP server**
(`Monitoring MCP`) и подключаются через **in-memory transport** MCP SDK
(`create_connected_server_and_client_session`). Это по-прежнему настоящий MCP:
`initialize` → `tools/list` → JSON-schema validation → `tools/call`.

Почему не stdio-подпроцесс, как Day 16/17:

* scheduler и SQLite обязаны жить в долгоживущем процессе приложения;
* короткоживущий subprocess не может владеть этим lifecycle;
* мост из subprocess в приложение потребовал бы HTTP/socket-хопа.

In-memory transport даёт настоящий MCP-протокол без лишнего процесса и порта.

### `start_monitoring`

| Field | Type | Required | Constraints |
| --- | --- | --- | --- |
| `service` | string | yes | 1–200 |
| `level` | string \| null | no | ≤ 32 |
| `interval_seconds` | integer | no (default 30) | **whitelist** `10,30,60,300,600,1800,3600` |
| `lookback_minutes` | integer | no (default 5) | 1–360 |
| `text_contains` | string \| null | no | ≤ 200 |
| `limit` | integer | no (default 100) | 1–500 |

Валидация whitelist объявлена через `Literal[...]` (публикуется в JSON-schema
как `enum`) **и** повторно проверяется в `MonitoringService` — backend не
полагается на frontend.

### Остальные tools

* `get_monitoring_status(job_id)` → статус, service, interval, lookback,
  last/next run, `runs_count`;
* `get_monitoring_summary(job_id, last_runs=10)` → агрегат по последним
  запускам (`runs`, success/failed, total/avg/min/max, last, trend);
* `stop_monitoring(job_id)` → idempotent stop, история сохраняется.

Неизвестный `job_id` возвращает контролируемый `{"error": ..., "error_kind": "not_found"}`.

## Interval presets и demo intervals

Поддерживаются только фиксированные пресеты:

| internal (сек) | UI label |
| --- | --- |
| 10 | 10 sec |
| 30 | 30 sec |
| 60 | 1 min |
| 300 | 5 min |
| 600 | 10 min |
| 1800 | 30 min |
| 3600 | 1 hour |

Произвольный интервал через UI задать нельзя; backend отклоняет всё, что не
входит в whitelist (`422`).

> **10 sec / 30 sec / 1 min существуют для демонстрации.** В реальном
> мониторинге обычно выбираются более длинные интервалы (5–60 минут).
> Отдельного скрытого demo-mode нет — это просто нижняя часть того же
> whitelist.

## Interval vs Lookback

Это **независимые** параметры:

* **Interval** — как часто выполняется проверка;
* **Lookback** — за какой период VictoriaLogs ищет записи при каждом запуске.

Например `interval = 30 sec`, `lookback = 5 min` допустим для demo. Окна
соседних запусков при этом **перекрываются по времени**. Для Day 18 это
нормально, потому что мы сохраняем результаты запусков (агрегаты), а не
уникальные raw events. Дедупликация событий не выполняется.

UI: отдельные `<select>` для interval (default `30 sec`) и lookback
(default `5 min`).

## First-run behaviour

Выбран вариант: **первый run выполняется сразу** (inline внутри
`start_monitoring`), а последующие — по расписанию каждые `interval_seconds`.

Причина: dashboard сразу показывает результат, а пользователю/зрителю demo не
нужно ждать 10–30 секунд. Если первый запуск упал (нет URL, timeout, HTTP
error), он сохраняется как failed run, а job **остаётся `active`** и
продолжает попытки по расписанию.

## Scheduled run flow

```
MonitoringService.run_job(job_id)
  1. load job из SQLite; если статус != active → выход
  2. overlap-guard: если run уже идёт → skip
  3. build_logsql(service, level, text_contains)   # Day 17 builder
  4. VictoriaLogsClient.search(start=now-lookback, end=now, limit)
  5. записать monitor_runs (logs_count, malformed_count, status, duration_ms)
  6. обновить job.last_run_at, job.next_run_at = finish + interval
```

Никакого LLM внутри `run_job`.

## Error behaviour

| Ситуация | Поведение |
| --- | --- |
| `VICTORIA_LOGS_BASE_URL` не задан | failed run `"not configured"`, job остаётся active |
| Timeout / network / HTTP 4xx-5xx | failed run с контролируемым сообщением |
| Неверный `job_id` | API `404`; MCP `error_kind: not_found` |
| Повторный stop | idempotent, `status: stopped` |
| Неверный interval | `422` (backend whitelist) |
| Неверный lookback | `422` (bounds 1..360) |
| SQLite / scheduler error | логируется; старт job при registration failure переводит job в `error` |
| Failed scheduled run | **не останавливает** scheduler: следующий run выполняется |

## Overlap protection

Двойная защита:

1. APScheduler `max_instances=1` + `coalesce=True` — второй запуск того же
   job не стартует, пока идёт первый;
2. in-process guard `MonitoringService._running` — даже прямой вызов
   `run_job()` для уже выполняющегося job возвращает `None` и не пишет run.

## Restart recovery

На startup `restore_active_jobs()`:

```
load active jobs из SQLite
  → зарегистрировать каждый в scheduler
```

`SQLite` — источник правды, `replace_existing` + предварительный
`unregister` не создают дубликатов. Если приложение было выключено дольше
`next_run_at`, следующий запуск ставится «как можно скорее» (`now`).
Stopped jobs не восстанавливаются.

## SQLite schema

БД: `data/day18/monitoring.db` (путь настраивается `DAY18_MONITORING_DB_PATH`).
Каталог `data/` уже в `.gitignore` и bind-mount-ится в Docker.

```sql
CREATE TABLE monitor_jobs (
    id               TEXT PRIMARY KEY,
    service          TEXT NOT NULL,
    level            TEXT,
    text_contains    TEXT,
    interval_seconds INTEGER NOT NULL,
    lookback_minutes INTEGER NOT NULL,
    limit_count      INTEGER NOT NULL,   -- API поле limit (SQL keyword)
    status           TEXT NOT NULL,      -- active | stopped | error
    created_at       TEXT NOT NULL,      -- UTC ISO-8601
    updated_at       TEXT NOT NULL,
    last_run_at      TEXT,
    next_run_at      TEXT
);

CREATE TABLE monitor_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT NOT NULL,
    logs_count      INTEGER NOT NULL DEFAULT 0,
    malformed_count INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL,       -- success | error
    error_message   TEXT,
    duration_ms     INTEGER NOT NULL DEFAULT 0
);
```

SQLite-доступ: **новая connection на операцию** (не шарятся между
threads/tasks), ленивое создание схемы, таймауты. Scheduler callback —
async, поэтому мост sync SQLite ↔ async не требует вложенных event loop.

## Aggregation

`get_monitoring_summary(job_id, last_runs=10, max=100)` считает агрегат на
backend по последним запускам:

```json
{
  "job_id": "mon_ab12cd",
  "service": "application",
  "runs": 3,
  "successful_runs": 3,
  "failed_runs": 0,
  "total_logs": 6,
  "average_logs_per_run": 2.0,
  "min_logs_per_run": 2,
  "max_logs_per_run": 2,
  "last_run_logs": 2,
  "first_run_at": "...",
  "last_run_at": "...",
  "trend": "stable"
}
```

`trend` (`rising` / `falling` / `stable`) детерминированно сравнивает
последний run со средним окна. Frontend агрегаты не считает.

## Docker persistence

`docker-compose.yml` уже монтирует `./data:/app/data`, поэтому
`data/day18/monitoring.db` переживает `restart` и пересоздание контейнера.
Добавлена только переменная `DAY18_MONITORING_DB_PATH` (default
`data/day18/monitoring.db`). Дополнительные порты не публикуются, MCP
остаётся локальным, VictoriaLogs — внешний HTTP API через
`VICTORIA_LOGS_BASE_URL`.

## Security / privacy

В SQLite **не сохраняются**:

* VictoriaLogs URL, токены, auth headers;
* raw log-сообщения и поля отдельных событий;
* секреты.

Хранятся только параметры job и агрегаты (`logs_count`, `malformed_count`,
`status`, `duration_ms`). Фильтр `text_contains` хранится, поскольку он
является частью определения job; он может быть потенциально чувствительным и
не должен использоваться для секретов.

## Manual verification

Проверено вручную против реального stage VictoriaLogs (без вывода raw-логов):

* job `mon_706379`, service `application`, level `ERROR`,
  `interval=10s`, `lookback=5min`, `limit=20`;
* 3 scheduled runs подряд, все `success`, `logs_count = 2`, duration
  ~260–320 ms;
* `summary`: `runs=3`, `total_logs=6`, `average=2.0`, `max=2`, `last=2`;
* `stop` → `stopped`, `next_run_at = null`.

Restart-проверка:

* создан job `mon_68725c`, выполнен 1 run, scheduler остановлен (эмуляция
  рестарта);
* новый `MonitoringService` над той же БД: `restore_active_jobs() = 1`,
  job зарегистрирован, следующий run выполнился (`runs` 1 → 2), затем stop.

В отчёте выводятся только `run count`, `logs_count`, timestamps и `status`.

## Video scenario

1. Открыть Week 4 → **Day 18**.
2. Выбрать service `application`.
3. Level = `ERROR`.
4. Interval = `10 sec` (или `30 sec`).
5. Lookback = `5 min`.
6. Нажать **Start monitoring**.
7. Показать job id и статус `Active`.
8. Показать первый run (выполнился сразу).
9. Дождаться ещё 1–2 запусков (таблица Recent runs обновляется polling'ом).
10. Показать run history.
11. Показать aggregated Summary.
12. Показать SQLite-записи **без raw logs**.
13. Показать код scheduler (`app/services/day18/scheduler.py`).
14. Показать MCP tool `start_monitoring`
    (`app/services/mcp/monitoring/server.py`).
15. Нажать **Stop**.
16. Показать `Stopped`.
17. (Bonus) создать job, перезапустить приложение/контейнер и показать, что
    мониторинг продолжился.

## Known limitations

* Lookback-окна соседних запусков могут перекрываться — дедупликации
  событий нет by design (храним агрегаты, а не уникальные события).
* `text_contains` хранится в БД (часть определения job).
* Демо-интервалы 10/30/60 сек не предназначены для production-мониторинга.
* Один процесс приложения — SQLite-файл не рассчитан на несколько
  параллельных writer-процессов (для учебного проекта достаточно).
* Day 19–20 (raw.log, analysis.md, `save_report`, pipeline
  search → analyze → save, GitLab MCP, multi-server orchestration) **не
  реализованы** — в UI остаются `Coming next`.
