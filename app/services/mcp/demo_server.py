"""Day 16 — minimal local Demo MCP server (stdio transport).

Run it directly for a manual protocol check::

    python -m app.services.mcp.demo_server

It is NOT an HTTP service: it speaks the MCP JSON-RPC protocol over stdin /
stdout and is meant to be spawned as a subprocess by ``MCPClient``. It binds no
port and needs no credentials. The tool set is intentionally tiny and safe:
``echo`` and ``get_server_info`` — Day 16 only needs a real ``tools/list``
response, not real business logic.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

SERVER_NAME = "Week 4 Demo MCP"
SERVER_VERSION = "1.0"


def build_server() -> FastMCP:
    """Create the Demo MCP server with its two deterministic tools.

    Kept as a factory so tests can introspect the registered tools without
    starting a subprocess.
    """
    server = FastMCP(SERVER_NAME)

    @server.tool()
    def echo(text: str) -> str:
        """Returns supplied text."""
        return text

    @server.tool()
    def get_server_info() -> dict:
        """Returns demo MCP server information."""
        return {"name": SERVER_NAME, "version": SERVER_VERSION, "day": 16}

    return server


mcp = build_server()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    # stdio is the default transport; explicitly stated for clarity.
    mcp.run(transport="stdio")
