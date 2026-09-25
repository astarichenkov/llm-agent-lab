"""Day 16 — local MCP (Model Context Protocol) integration.

This package contains the two sides of the Day 16 demonstration:

* ``demo_server`` — a minimal LOCAL MCP server (stdio transport) exposing the
  ``echo`` and ``get_server_info`` tools;
* ``client`` — the MCP client used by the backend. It spawns the demo server
  as a subprocess, performs the MCP initialization handshake and calls
  ``tools/list`` to discover the available tools.

Transport is ``stdio`` only: no public endpoint, no network binding, no
credentials. Days 17-20 will build on this foundation.
"""
# NOTE: do NOT import ``demo_server`` here. It is executed as
# ``python -m app.services.mcp.demo_server``; importing it from the package
# __init__ would put it in sys.modules before runpy executes it as __main__
# and emit a RuntimeWarning in the server subprocess.
from app.services.mcp.client import (
    MCPClient,
    MCPServerConfig,
    default_server_config,
)

__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "default_server_config",
]
