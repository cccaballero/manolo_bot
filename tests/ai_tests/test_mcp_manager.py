import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ai.mcp_manager import MCPManager
from config import Config


class TestMCPManager(unittest.IsolatedAsyncioTestCase):
    """Tests for MCPManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Set required environment variables for Config
        os.environ.setdefault("TELEGRAM_BOT_NAME", "TestBot")
        os.environ.setdefault("TELEGRAM_BOT_USERNAME", "test_bot")
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test_token")
        os.environ.setdefault("GOOGLE_API_KEY", "test_key")

        self.config = Config()
        self.config.enable_mcp = True

    async def test_mcp_manager_no_config(self):
        """Test MCP manager with no configuration."""
        self.config.mcp_servers_config = "{}"
        manager = MCPManager(self.config)

        await manager.connect()

        self.assertFalse(manager.is_connected)
        tools = await manager.get_tools()
        self.assertEqual(len(tools), 0)

    async def test_mcp_manager_invalid_json(self):
        """Test MCP manager with invalid JSON configuration."""
        self.config.mcp_servers_config = "invalid json"
        manager = MCPManager(self.config)

        with self.assertRaises(ValueError):
            await manager.connect()

    @patch("ai.mcp_manager.MultiServerMCPClient")
    async def test_mcp_manager_valid_config(self, mock_client_class):
        """Test MCP manager with valid configuration."""
        # Setup mock
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(
            return_value=[
                MagicMock(name="test_tool_1"),
                MagicMock(name="test_tool_2"),
            ]
        )
        mock_client_class.return_value = mock_client

        # Configure manager
        server_config = {"test_server": {"command": "python", "args": ["/path/to/server.py"], "transport": "stdio"}}
        self.config.mcp_servers_config = json.dumps(server_config)

        manager = MCPManager(self.config)
        await manager.connect()

        # Verify
        self.assertTrue(manager.is_connected)
        tools = await manager.get_tools()
        self.assertEqual(len(tools), 2)

        # Cleanup
        await manager.disconnect()
        mock_client.close.assert_called_once()

    async def test_mcp_manager_disconnect_before_connect(self):
        """Test disconnecting MCP manager before connecting."""
        manager = MCPManager(self.config)

        # Should not raise exception
        await manager.disconnect()

        self.assertFalse(manager.is_connected)

    async def test_mcp_manager_invalid_transport(self):
        """Test MCP manager with invalid transport type."""
        server_config = {"test_server": {"command": "python", "transport": "invalid_transport"}}
        self.config.mcp_servers_config = json.dumps(server_config)

        manager = MCPManager(self.config)

        with self.assertRaises(ValueError) as context:
            await manager.connect()

        self.assertIn("invalid transport", str(context.exception))

    async def test_mcp_manager_missing_command_for_stdio(self):
        """Test MCP manager with missing command for stdio transport."""
        server_config = {"test_server": {"transport": "stdio"}}
        self.config.mcp_servers_config = json.dumps(server_config)

        manager = MCPManager(self.config)

        with self.assertRaises(ValueError) as context:
            await manager.connect()

        self.assertIn("missing 'command'", str(context.exception))

    async def test_mcp_manager_missing_url_for_http(self):
        """Test MCP manager with missing URL for streamable_http transport."""
        server_config = {"test_server": {"transport": "streamable_http"}}
        self.config.mcp_servers_config = json.dumps(server_config)

        manager = MCPManager(self.config)

        with self.assertRaises(ValueError) as context:
            await manager.connect()

        self.assertIn("missing 'url'", str(context.exception))

    @patch("ai.mcp_manager.MultiServerMCPClient")
    async def test_mcp_manager_connect_twice(self, mock_client_class):
        """Test connecting MCP manager twice doesn't reinitialize."""
        # Setup mock
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[MagicMock(name="test_tool")])
        mock_client_class.return_value = mock_client

        # Configure manager
        server_config = {"test_server": {"command": "python", "args": ["server.py"], "transport": "stdio"}}
        self.config.mcp_servers_config = json.dumps(server_config)

        manager = MCPManager(self.config)
        await manager.connect()
        await manager.connect()  # Second call should be no-op

        # Client should only be initialized once
        mock_client_class.assert_called_once()
        self.assertTrue(manager.is_connected)
