"""Day 20 — multi-server MCP registry and tool router.

This module is the heart of the Day 20 orchestration and is deliberately
transport-agnostic. A registry is given ONE client per MCP server; every
client implements the same small surface used across Week 4::

    await client.discover_tools() -> MCPStatusResponse
    await client.call_tool(name, args) -> MCPToolCallResult

``discover_all()`` performs ``tools/list`` on every registered server and
builds a single mapping ``tool_name -> server_name``. Routing is then a plain
dictionary lookup — there is no ``if/elif`` chain anywhere:

    search_logs       -> victorialogs
    recent_commits    -> gitea
    get_commit        -> gitea
    get_commit_diff   -> gitea
    save_report       -> reports

If two servers publish the SAME tool name the registry does NOT pick one at
random: the collision is reported as an explicit configuration error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.day16 import MCPStatusResponse, MCPToolInfo
from app.services.mcp.client import MCPToolCallResult


class RegistryError(Exception):
    """Base class for controlled registry failures."""


class ToolCollisionError(RegistryError):
    """Two MCP servers published the same tool name."""

    def __init__(self, collisions: list[tuple[str, str, str]]) -> None:
        self.collisions = collisions
        detail = "; ".join(
            f"{tool}: {first} vs {second}" for tool, first, second in collisions
        )
        super().__init__(f"duplicate MCP tool name(s): {detail}")


class UnknownToolError(RegistryError):
    """A tool was requested that no connected server exposes."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"unknown MCP tool: {tool_name}")
        self.tool_name = tool_name


@dataclass
class RegisteredServer:
    """One registered server plus its latest discovery outcome."""

    name: str
    client: Any
    label: str = ""
    transport: str = "stdio"
    connected: bool = False
    tools: list[MCPToolInfo] = field(default_factory=list)
    error: str | None = None

    @property
    def tools_count(self) -> int:
        return len(self.tools)


@dataclass
class DiscoveryResult:
    """Aggregated result of ``tools/list`` across every server."""

    servers: list[RegisteredServer] = field(default_factory=list)
    tool_map: dict[str, str] = field(default_factory=dict)
    tools: list[MCPToolInfo] = field(default_factory=list)
    collisions: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def connected_any(self) -> bool:
        return any(server.connected for server in self.servers)


class MCPServerRegistry:
    """Registers MCP servers and routes tool calls to the right one."""

    def __init__(self) -> None:
        self._servers: dict[str, RegisteredServer] = {}
        self._tool_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        client: Any,
        *,
        label: str = "",
        transport: str = "stdio",
    ) -> RegisteredServer:
        server = RegisteredServer(
            name=name, client=client, label=label or name, transport=transport
        )
        self._servers[name] = server
        return server

    @property
    def server_names(self) -> list[str]:
        return list(self._servers)

    def get_client(self, name: str) -> Any | None:
        server = self._servers.get(name)
        return server.client if server else None

    def get_server(self, name: str) -> RegisteredServer | None:
        return self._servers.get(name)

    def servers(self) -> list[RegisteredServer]:
        return list(self._servers.values())

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    async def discover_all(self) -> DiscoveryResult:
        """Run ``tools/list`` on every server and build the tool -> server map.

        A failure on ONE server never aborts discovery of the others: the
        failure is recorded on that server and the remaining tools stay
        available (partial-failure handling).
        """
        result = DiscoveryResult()
        result.tool_map = {}
        seen: dict[str, str] = {}

        for server in self._servers.values():
            try:
                status = await server.client.discover_tools()
            except Exception as exc:  # noqa: BLE001 - isolation between servers
                server.connected = False
                server.tools = []
                server.error = str(exc) or exc.__class__.__name__
                result.servers.append(server)
                continue

            if not isinstance(status, MCPStatusResponse):
                server.connected = False
                server.tools = []
                server.error = "invalid discovery response"
                result.servers.append(server)
                continue

            server.connected = bool(status.connected)
            server.tools = list(status.tools)
            server.error = status.error
            if status.server and status.server.name and not server.label:
                server.label = status.server.name
            if status.server and status.server.transport:
                server.transport = status.server.transport
            result.servers.append(server)

            if not server.connected:
                continue
            for tool in server.tools:
                if tool.name in seen:
                    result.collisions.append(
                        (tool.name, seen[tool.name], server.name)
                    )
                else:
                    seen[tool.name] = server.name
                    result.tools.append(tool)

        # Colliding tool names are excluded from the map so a lookup can never
        # silently resolve to an arbitrary server.
        colliding = {tool for tool, _, _ in result.collisions}
        result.tool_map = {
            tool: owner
            for tool, owner in seen.items()
            if tool not in colliding
        }
        self._tool_map = result.tool_map
        return result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    @property
    def tool_map(self) -> dict[str, str]:
        return dict(self._tool_map)

    def resolve(self, tool_name: str) -> str:
        """Return the server name that owns ``tool_name`` or raise."""
        try:
            return self._tool_map[tool_name]
        except KeyError as exc:
            raise UnknownToolError(tool_name) from exc

    def routing(self, result: DiscoveryResult | None = None) -> list[dict[str, str]]:
        """Return the tool -> server routing table for the UI/API."""
        rows: list[dict[str, str]] = []
        if result is not None:
            descriptions = {
                tool.name: tool.description for tool in result.tools
            }
            labels = {server.name: server.label for server in result.servers}
            for tool, server_name in result.tool_map.items():
                rows.append(
                    {
                        "tool": tool,
                        "server": server_name,
                        "label": labels.get(server_name, server_name),
                        "description": descriptions.get(tool, ""),
                    }
                )
        else:
            for tool, server_name in self._tool_map.items():
                server = self._servers.get(server_name)
                rows.append(
                    {
                        "tool": tool,
                        "server": server_name,
                        "label": server.label if server else server_name,
                        "description": "",
                    }
                )
        return sorted(rows, key=lambda row: (row["server"], row["tool"]))

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolCallResult:
        """Resolve the owning server and perform one real MCP ``tools/call``."""
        server_name = self.resolve(tool_name)
        server = self._servers[server_name]
        return await server.client.call_tool(tool_name, arguments)

    def server_for_tool(self, tool_name: str) -> RegisteredServer:
        """Return the registered server that owns ``tool_name`` or raise."""
        server_name = self.resolve(tool_name)
        return self._servers[server_name]


__all__ = [
    "DiscoveryResult",
    "MCPServerRegistry",
    "RegisteredServer",
    "RegistryError",
    "ToolCollisionError",
    "UnknownToolError",
]
