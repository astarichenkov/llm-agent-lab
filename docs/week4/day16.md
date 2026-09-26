# Week 4 / Day 16 — MCP Connection & Tool Discovery

## Goal

Connect the existing backend to a **local MCP server**, complete the MCP
initialization handshake and call `tools/list`, then show the discovered tools
in the UI. Day 16 is about the **MCP protocol**, not about LLM tool calling:
DeepSeek is not called at all.

```
Browser  →  FastAPI  →  MCP client  →  stdio  →  Demo MCP server
                                           (initialize → tools/list)
```

## Dependency / version note (MCP SDK 1.x is intentional)

Day 16 is implemented against **MCP Python SDK 1.x**. The dependency is
intentionally pinned in `requirements.txt`:

```
mcp[cli]>=1.0,<2.0
```

The Day 16 code uses the v1 API surface:

- `from mcp.server.fastmcp import FastMCP` — in SDK **2.x** `FastMCP` was
  renamed/replaced by `MCPServer` (`mcp.server.mcpserver`); importing
  `mcp.server.fastmcp` under 2.x raises `ModuleNotFoundError`.
- `ClientSession(..., read_timeout_seconds=timedelta(...))` — 2.x changes this
  parameter to a `float` number of seconds.

Because of these breaking changes, the SDK's own error text recommends
`pin 'mcp<2' to keep running v1 code`. The `<2.0` constraint is therefore
**intentional and technically necessary**; without it pip would install 2.x
and the Demo MCP server would not start.

The project environment (`.venv`) resolves the pin to the latest 1.x release,
**mcp 1.30.0**. The `mcp version` -> `2.1.1` shown by the `mcp` command on
Windows comes from a **separate, global** Python installation
(`C:\Users\Anton\AppData\Local\Programs\Python\Python314\...`), not from the
project virtualenv, and is not used by the application or tests.

If the project later migrates to MCP SDK 2.x, the change requires:
`FastMCP` -> `MCPServer`, and `read_timeout_seconds` -> a float; only then
should the pin be raised to `mcp[cli]>=2,<3`.

## Concepts

- **MCP server** — a process that exposes capabilities (here: tools) over the
  Model Context Protocol. The one used here is a minimal local demo
  (`echo`, `get_server_info`).
- **MCP client** — the code that connects to a server, performs the
  `initialize` handshake, then sends requests such as `tools/list`.
- **Transport** — how client and server exchange JSON-RPC messages. Day 16
  uses **stdio**.

## Why stdio?

- Local only: the server is spawned as a subprocess; no port is opened and no
  public endpoint is created.
- No credentials, no network, no Docker networking, no HTTPS.
- Deterministic and easy to demonstrate: one process talking to another over
  stdin/stdout.

## Implementation files

| File | Responsibility |
| --- | --- |
| `app/services/mcp/demo_server.py` | FastMCP demo server with `echo` + `get_server_info`. |
| `app/services/mcp/client.py` | `MCPClient`: spawns the server over stdio, initializes the session, calls `tools/list`, maps the result, closes everything. |
| `app/schemas/day16.py` | API response models (`MCPStatusResponse`, `MCPToolInfo`, `MCPTraceStep`, `MCPServerInfo`). |
| `app/api/routes.py` | `GET /api/week4/day16/mcp/status` endpoint + `get_mcp_client` dependency. |
| `app/main.py` | Creates `app.state.mcp_client`. |
| `app/templates/_day16.html` | Day 16 UI (status, tools, trace). |
| `app/static/js/day16.js` | Renders whatever the backend returns; no hardcoded tools. |
| `tests/test_day16.py` | Server, client, endpoint, failure and UI tests. |
| `tests/js/frontend_harness.js` | Day 16 DOM rendering checks. |

## How the server is launched

`MCPClient` uses `mcp.client.stdio.stdio_client` with:

```text
command = sys.executable
args    = ["-m", "app.services.mcp.demo_server"]
cwd     = <project root>
```

The fresh subprocess is created on every call, so the UI's
**Reconnect / Refresh tools** button genuinely repeats the whole handshake.
Inside Docker the same code works: `app/` is copied to `/app` and the `mcp`
dependency is installed from `requirements.txt`.

## How `list_tools()` is called

`MCPClient.discover_tools()`:

1. starts the subprocess and records the `start_server` trace step;
2. opens the stdio transport (`connect`);
3. opens a `ClientSession` with a read timeout and runs `session.initialize()`;
4. calls `session.list_tools()`;
5. maps each MCP tool (`name`, `description`, `inputSchema`) to `MCPToolInfo`;
6. records the `closed` step and returns `MCPStatusResponse`.

Connection/protocol failures never leak a stack trace: they are returned as
`connected=false` with a human-readable `error` and an error trace step.

## Manual verification

```bash
# 1) Sanity-check the server in isolation (optional; it speaks MCP over stdio)
.venv/Scripts/python -m app.services.mcp.demo_server

# 2) Run the app
.venv/Scripts/uvicorn app.main:app --reload

# 3) Call the Day 16 endpoint
curl http://127.0.0.1:8000/api/week4/day16/mcp/status
```

Then open `http://127.0.0.1:8000/`, select **Week 4 → Day 16**, and press
**Reconnect / Refresh tools**.

Expected response shape:

```json
{
  "connected": true,
  "server": {"name": "Week 4 Demo MCP", "transport": "stdio"},
  "tools_count": 2,
  "tools": [
    {"name": "echo", "description": "Returns supplied text.", "input_schema": {...}},
    {"name": "get_server_info", "description": "Returns demo MCP server information.", "input_schema": {...}}
  ],
  "trace": [
    {"step": "connect", "status": "ok", "message": "Connected via stdio"},
    {"step": "initialize", "status": "ok", "message": "MCP session initialized: Week 4 Demo MCP ..."},
    {"step": "list_tools", "status": "ok", "message": "Received 2 tools"}
  ],
  "error": null
}
```

## Tests

```bash
.venv/Scripts/python -m pytest tests/test_day16.py -q
```

Covers: server tool registration; real client connect/initialize/list_tools;
the API endpoint (`connected`, `tools_count >= 2`, `echo` present); controlled
failure handling for a broken server; and that `day16.js` does not hardcode the
tool list.

## Video demo scenario

1. Open **Week 4 → Day 16**.
2. Show the status: `Connecting…` → `Connected`, transport `stdio`, server
   `Week 4 Demo MCP`.
3. Press **Reconnect / Refresh tools** to show the handshake runs again.
4. Show the **MCP Trace**: start server → connect → initialize → tools/list →
   connection closed.
5. Show the tool cards really returned by the server: `echo` and
   `get_server_info` with their input schemas.
6. Briefly show the backend `list_tools()` call in
   `app/services/mcp/client.py`.
7. (Optional) point at `app/services/mcp/demo_server.py` to show where the
   tools are registered.

## Out of scope (Days 18–20)

Day 17 (VictoriaLogs MCP server + agent tool calling) is implemented in
`docs/week4/day17.md`. Still not implemented: scheduler/periodic monitoring,
SQLite persistence, background jobs, reports, `analyze_logs` / `save_report`,
GitLab MCP and multi-server orchestration. Days 18–20 are shown as disabled
"Coming next" tabs on purpose.
