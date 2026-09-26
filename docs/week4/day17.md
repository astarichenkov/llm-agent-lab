# Week 4 / Day 17 — VictoriaLogs MCP Tool

## Goal

Build a **real MCP server around a real API** (VictoriaLogs), register a
useful tool (`search_logs`), and let the existing LLM agent call it. Day 17 is
the first day where the model itself decides to invoke an MCP tool; the tool
result is fed back to the model and used in the final answer.

```
Browser
   │  POST /api/week4/day17/chat
   ▼
FastAPI  →  Day17LogsService (agent tool-calling loop)
   │            │
   │            ├─ MCP initialize + tools/list (stdio)
   │            ├─ LLM function calling  → model selects search_logs
   │            └─ MCP tools/call (stdio subprocess)
   │                     │
   │                     ▼
   │             VictoriaLogs MCP server
   │                     │  HTTPS
   │                     ▼
   │             VictoriaLogs /select/logsql/query   (JSON Lines)
   ▼
final answer + real tool call + real trace
```

**Day 17 contains a genuine MCP `tools/call`.** The backend never calls the
VictoriaLogs HTTP API directly from a route/controller: the only HTTP path to
VictoriaLogs is inside the MCP server subprocess.

## Architecture / files

| File | Responsibility |
| --- | --- |
| `app/services/mcp/victorialogs/client.py` | `VictoriaLogsClient`: builds the `/select/logsql/query` URL, bounded request, streams/parses JSON Lines, classifies errors. |
| `app/services/mcp/victorialogs/query_builder.py` | Safe LogsQL construction from structured filters (validated field names + escaped values). |
| `app/services/mcp/victorialogs/sanitizer.py` | Masks obvious secrets before content reaches the LLM. |
| `app/services/mcp/victorialogs/schemas.py` | `SearchLogsInput` / `NormalizedLog` / `SearchLogsResult` (typed limits). |
| `app/services/mcp/victorialogs/server.py` | FastMCP server (stdio) registering `search_logs`. |
| `app/services/mcp/client.py` | Existing MCP client extended with `call_tool()` and subprocess `env` forwarding. |
| `app/services/day17/service.py` | The agent tool-calling loop + trace + iteration cap. |
| `app/schemas/day17.py` | API request/response models. |
| `app/api/routes.py` | `POST /api/week4/day17/chat` + `get_day17_service`. |
| `app/templates/_day17.html`, `app/static/js/day17.js` | Day 17 UI. |
| `tests/test_day17.py` | Server, query, client, sanitizer, real-subprocess, agent and UI tests. |

The MCP SDK stays at **1.x (`mcp[cli]>=1.0,<2.0`)** — no migration to 2.x.

## MCP server & transport

`app/services/mcp/victorialogs/server.py` is run as a local subprocess:

```
command = sys.executable
args    = ["-m", "app.services.mcp.victorialogs.server"]
cwd     = <project root>
transport = stdio
```

The server binds **no port** and is not exposed to the network. The MCP client
forwards `VICTORIA_LOGS_*` variables explicitly because the MCP stdio transport
only inherits a small safe allow-list of environment variables.

## MCP tool: `search_logs`

Description (published to the LLM via `tools/list`):

> Search recent stage logs stored in VictoriaLogs using safe structured
> filters (service, time window, level and optional message text). Returns
> normalized log rows (timestamp, message, fields) plus the generated query
> and a count. Use this when the user asks about service logs, errors or
> incidents. The time window is limited and results are capped.

### Input schema

| Field | Type | Required | Constraints |
| --- | --- | --- | --- |
| `service` | string | yes | 1–200 chars |
| `since_minutes` | integer | no (default 15) | 1–360; ignored when `start`+`end` are set |
| `level` | string \| null | no | ≤ 32 chars |
| `text_contains` | string \| null | no | ≤ 200 chars |
| `limit` | integer | no (default 100) | 1–500 |
| `start` | datetime \| null | no | ISO8601 UTC; used together with `end` |
| `end` | datetime \| null | no | ISO8601 UTC; window ≤ 24 h |

Raw LogsQL is **not** accepted from the caller — only structured filters. The
same constraints are declared with `pydantic.Field` (so the model sees them)
and re-validated through `SearchLogsInput`.

### Output shape

```json
{
  "service": "orders-service",
  "since_minutes": 30,
  "level": "ERROR",
  "text_contains": null,
  "query": "service:\"orders-service\" AND level:\"ERROR\"",
  "count": 24,
  "limit": 100,
  "truncated": false,
  "malformed_lines": 0,
  "logs": [
    {
      "timestamp": "2026-09-26T10:00:00Z",
      "message": "...",
      "fields": {"level": "ERROR", "service": "orders-service"}
    }
  ]
}
```

`_time` is normalized to `timestamp`, `_msg` to `message`; every other field is
kept under `fields` (sanitized).

## Query construction (LogsQL)

`build_logsql()` emits a query such as:

```text
service:"orders-service" AND level:"ERROR" AND "timeout"
```

* the service field name is validated as a LogsQL identifier;
* values are emitted as double-quoted phrases with `\` and `"` escaped, so an
  injected quote cannot break out of the filter;
* `text_contains` becomes a bare quoted phrase (searches the message body).

### Service field assumption (IMPORTANT)

The field that stores the service name is **deployment-specific**. Day 17 uses
a configurable field name:

```text
VICTORIA_LOGS_SERVICE_FIELD   (default: service)
```

The default `service` is an assumption that must be verified against real data.
If the deployment uses a different field, set `VICTORIA_LOGS_SERVICE_FIELD`
accordingly (no code change needed). Manual integration verification confirmed
the staged deployment exposes a `service` field, but this may differ per
environment.

## Time range & limit

* By default `start = now - since_minutes`, `end = now` (UTC, RFC3339).
* The UI interval picker sends an **explicit `start`/`end`** window, which then
  takes precedence. The window is stated in the LLM context and **enforced** on
  the actual `search_logs` call, so the model cannot drift to another range.
* `since_minutes` is capped at **360** (6 h); an explicit window is capped at
  **24 h**.
* a **server-side `limit`** is always sent (default **100**, max **500**); the
  client also applies a defensive cap while parsing.

### UI interval picker

The Day 17 panel offers:

* presets: **last 30 minutes, 1 hour, 4 hours, 8 hours, 24 hours**;
* explicit `datetime-local` **start**/**end** inputs;
* **−1 hour / +1 hour** paging (shifts both bounds) and a **Now** button.

## Query timeout

Both layers enforce a timeout:

* client-side `httpx` timeout: `VICTORIA_LOGS_TIMEOUT_SECONDS` (default 5s);
* server-side LogsQL `timeout=5s` argument.

A timeout is mapped to a controlled `VictoriaLogsTimeoutError`, never a hang.

## JSON Lines handling

`/select/logsql/query` returns **JSON Lines**, not a JSON array. The client
streams the response (`client.stream(...)` + `aiter_lines()`), parses each line
with `json.loads`, ignores blank lines, skips malformed lines (counted in
`malformed_lines`) and stops at `limit`. It never buffers an unbounded body.

## Sanitization

Before any log content is passed to the LLM, `sanitize_value` / `sanitize_text`
mask obvious secrets:

* `Authorization: Bearer …` and bare `Bearer …`;
* `password=` / `passwd:` / `pwd=`;
* `token=` / `access_token=` / `refresh_token=` / `id_token=`;
* `api_key=` / `apikey=` / `secret=` / `client_secret=`;
* `Cookie:` / `Set-Cookie:`;
* dictionary keys named like the above.

Matched values become `[REDACTED]`. This is a simple safety net, not a DLP.

## Configuration

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `VICTORIA_LOGS_BASE_URL` | only when Day 17 is used | — | e.g. `https://vlogs.example.com` (no trailing-slash assumptions). |
| `VICTORIA_LOGS_SERVICE_FIELD` | no | `service` | Verify against real data. |
| `VICTORIA_LOGS_TIMEOUT_SECONDS` | no | `5` | HTTP timeout. |
| `VICTORIA_LOGS_VERIFY_SSL` | no | `true` | `false` only for an internal CA. |
| `VICTORIA_LOGS_CA_BUNDLE` | no | — | Path to a CA bundle (PEM). Preferred over `VERIFY_SSL=false`; wins when set. |
| `VICTORIA_LOGS_CA_DIR` | no | `./certs` | docker-compose only: host dir mounted read-only at `/certs`. |

Docker example (secure internal CA): put the CA in a host directory (e.g.
`<HOST_CA_DIR>`), copy `.env.example` to `.env` (git-ignored), then set
`VICTORIA_LOGS_CA_DIR=<HOST_CA_DIR>`,
`VICTORIA_LOGS_CA_BUNDLE=/certs/<CA_FILE>` and `VICTORIA_LOGS_VERIFY_SSL=true`.
Never commit the CA: `certs/*.crt|*.pem|*.cer|*.key` and `.env` are git-ignored.

### TLS tip (httpx maps TLS failures to `ConnectError`)

If the tool reports `Could not reach VictoriaLogs: ConnectError`, the cause is
usually **certificate verification**, not the network: `httpx` wraps
`SSLCertVerificationError` in `ConnectError`. Verify connectivity first
(`/select/logsql/query?query=*&limit=1`), then either mount the internal CA and
set `VICTORIA_LOGS_CA_BUNDLE`, or (less secure) set
`VICTORIA_LOGS_VERIFY_SSL=false` for an internal-only endpoint.

A missing `VICTORIA_LOGS_BASE_URL` **does not break startup or Day 16**:
`VictoriaLogsClient` validates it lazily and the tool returns a controlled
`VICTORIA_LOGS_BASE_URL is not configured` error.

## Agent tool-call flow

1. `discover_tools()` — spawn MCP server (stdio), `initialize`, `tools/list`.
2. The discovered tools are converted to OpenAI function tools (schema passed
   through unchanged).
3. First LLM call with the tools. The model chooses `search_logs` and provides
   structured arguments.
4. `MCPClient.call_tool("search_logs", arguments)` — real MCP `tools/call`.
5. The normalized result is appended as a `role="tool"` message.
6. Second LLM call → final natural-language answer based on the tool result.

The loop is capped at `MAX_TOOL_ITERATIONS = 3`; exceeding it ends the turn
with a controlled error.

## Error handling

Handled without ever showing a Python traceback to the user:

* DNS/connect (`VictoriaLogsNetworkError`), timeout (`VictoriaLogsTimeoutError`),
  HTTP 4xx/5xx (`VictoriaLogsHTTPError`);
* invalid JSONL lines (skipped + counted), empty result (valid, `count=0`);
* missing configuration (`VictoriaLogsConfigError`);
* MCP server unavailable / protocol failure (controlled `error`);
* provider errors (`DeepSeekError` subclasses).

Tracebacks go to the backend log only.

## How to run

```bash
# 1) configure (do NOT commit real values)
cp .env.example .env
# set VICTORIA_LOGS_BASE_URL=... (and VICTORIA_LOGS_SERVICE_FIELD if needed)

# 2) run the app
.venv/Scripts/uvicorn app.main:app --reload

# 3) call the Day 17 endpoint
curl -X POST http://127.0.0.1:8000/api/week4/day17/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Покажи ERROR для orders-service за последние 30 минут"}'
```

The MCP server can also be started standalone (it speaks MCP/JSON-RPC over
stdio and will wait for input):

```bash
.venv/Scripts/python -m app.services.mcp.victorialogs.server
```

## Manual integration verification

Only if `VICTORIA_LOGS_BASE_URL` is available on the machine:

1. Start with a **small** request: 15-minute window, `limit=3`, narrow service.
2. Verify the HTTP status, the number of rows and the detected field names.
3. If the real `service` field differs, set `VICTORIA_LOGS_SERVICE_FIELD`.
4. Do **not** commit real logs or the internal URL. Temporary files (if any)
   must live in a git-ignored location and be removed afterwards.

## Tests

```bash
.venv/Scripts/python -m pytest tests/test_day17.py -q
node tests/js/frontend_harness.js
```

Coverage: tool registration/description/schema; input validation; safe query
building; HTTP client endpoint/params/JSONL/errors; sanitizer; real MCP
subprocess against a **local mock** VictoriaLogs; agent flow (tool result
reaches the second LLM call); iteration cap; API endpoint; UI state. No test
contacts the real stage.

## Video scenario

1. Open **Week 4 → Day 17**.
2. Show that **Day 16 still works** (switch to it and reconnect).
3. Enter: `Покажи ERROR для <service> за последние 15 минут` and press **Send**.
4. Show `LLM selected MCP tool(s): search_logs`.
5. Show the tool arguments: `service` / `since_minutes` / `level` / `limit`.
6. Show `VictoriaLogs returned N logs`.
7. Show the final LLM answer.
8. Show the **MCP Trace** (discover → select → call_tool → VictoriaLogs → LLM).
9. Open `server.py` and show the `search_logs` registration.
10. Open `client.py` and show the `/select/logsql/query` request.

## Out of scope (Days 18–20)

Not implemented yet: scheduler/periodic monitoring, persistence (SQLite),
background jobs, saved reports, `analyze_logs` / `save_report` tools, GitLab
MCP and multi-server orchestration. Days 18–20 stay disabled "Coming next".
