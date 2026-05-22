"""MCP Manager for handling Model Context Protocol servers."""

import json
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ai.config import BotConfig

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages MCP server connections and tool loading."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._client: MultiServerMCPClient | None = None
        self._tools: list[BaseTool] = []
        self._connected = False

    async def connect(self) -> None:
        """Initialize MCP client and connect to configured servers."""
        if self._connected:
            return

        if not self.config.mcp_servers_config:
            logger.info("No MCP servers configured")
            return

        try:
            # Parse server configuration
            servers = self._parse_server_config(self.config.mcp_servers_config)

            if not servers:
                logger.warning("MCP enabled but no valid servers configured")
                return

            # Initialize MultiServerMCPClient
            self._client = MultiServerMCPClient(servers)

            # Load tools from all configured servers
            self._tools = await self._client.get_tools()

            self._connected = True
            logger.info(f"MCP connected: {len(servers)} server(s), {len(self._tools)} tool(s) loaded")

        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}", exc_info=True)
            raise

    async def disconnect(self) -> None:
        """Close all MCP server connections."""
        if self._client:
            try:
                await self._client.close()
                logger.info("MCP connections closed")
            except Exception as e:
                logger.warning(f"Error closing MCP connections: {e}")

        self._connected = False
        self._tools = []

    async def get_tools(self) -> list[BaseTool]:
        """Get all loaded MCP tools."""
        if not self._connected:
            logger.warning("MCP not connected, returning empty tool list")
            return []

        return self._tools

    @property
    def is_connected(self) -> bool:
        """Check if MCP is connected."""
        return self._connected

    def _parse_server_config(self, config: dict) -> dict:
        """Parse MCP server configuration from JSON string."""
        try:
            if not isinstance(config, dict):
                raise ValueError("MCP_SERVERS_CONFIG must be a JSON object")

            servers = config.get("mcpServers", config)

            # Validate basic structure
            if not isinstance(servers, dict):
                raise ValueError("MCP_SERVERS_CONFIG must be a JSON object")

            # Validate each server configuration
            for name, server_config in servers.items():
                if not isinstance(server_config, dict):
                    raise ValueError(f"Server {name!r} configuration must be an object")

                transport = server_config.get("transport", "stdio")
                if transport not in ["stdio", "streamable_http", "sse"]:
                    raise ValueError(f"Server {name!r} has invalid transport: {transport!r}")

                if transport == "stdio":
                    if "command" not in server_config:
                        raise ValueError(f"Server {name!r} missing 'command' for stdio transport")
                elif transport in {"streamable_http", "sse"}:
                    if "url" not in server_config:
                        raise ValueError(f"Server {name!r} missing 'url' for {transport} transport")

            return servers

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MCP_SERVERS_CONFIG: {e}")
            raise ValueError(f"MCP_SERVERS_CONFIG contains invalid JSON: {e}") from e
