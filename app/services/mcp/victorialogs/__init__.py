"""Day 17 — VictoriaLogs MCP server + reusable VictoriaLogs HTTP client.

``server`` is intentionally NOT imported here: it is executed as
``python -m app.services.mcp.victorialogs.server`` and importing it from the
package ``__init__`` would place it in ``sys.modules`` before runpy runs it as
``__main__`` (the same caveat documented for the Day 16 demo server).
"""
from app.services.mcp.victorialogs.client import (
    VictoriaLogsClient,
    VictoriaLogsConfigError,
    VictoriaLogsError,
)
from app.services.mcp.victorialogs.query_builder import build_logsql
from app.services.mcp.victorialogs.sanitizer import sanitize_text, sanitize_value
from app.services.mcp.victorialogs.schemas import SearchLogsInput

__all__ = [
    "VictoriaLogsClient",
    "VictoriaLogsConfigError",
    "VictoriaLogsError",
    "build_logsql",
    "sanitize_text",
    "sanitize_value",
    "SearchLogsInput",
]
