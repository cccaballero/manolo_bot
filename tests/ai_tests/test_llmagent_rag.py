"""Agent-level RAG tests: `rag_backend=` param and MCP-wins policy (no network)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.config import Config
from manolo_bot.rag.inmemory_backend import InMemoryRAGBackend


@tool
def search_stub(query: str) -> str:
    """Stub search tool."""
    return "stub"


@tool
def rag_search(query: str) -> str:
    """MCP-provided rag_search tool (shadows the backend one)."""
    return "mcp"


async def _make_indexed_backend(tmp: str) -> InMemoryRAGBackend:
    backend = InMemoryRAGBackend(
        embeddings=FakeEmbeddings(size=32),
        bot_uuid="test-bot",
        store_path=str(Path(tmp) / "store"),
    )
    await backend.build_or_load()
    src = Path(tmp) / "handbook.md"
    src.write_text("Manolo handbook content about deployments.\n", encoding="utf-8")
    await backend.ingest([str(src)])
    return backend


class TestLLMAgentRAGBackend(unittest.IsolatedAsyncioTestCase):
    def _make_agent(self, rag_backend=None) -> LLMAgent:
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        mock_llm.get_num_tokens_from_messages = MagicMock(return_value=10)
        mock_config = MagicMock(spec=Config)
        mock_config.enable_mcp = False
        mock_config.rag_sources = ["docs/handbook.md"]
        mock_messages_storage = MagicMock()
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        return LLMAgent(mock_llm, mock_config, system_instructions, mock_messages_storage, rag_backend=rag_backend)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_rag_backend_appends_rag_search_tool(self, mock_create_agent, mock_get_all_tools):
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            backend = await _make_indexed_backend(tmp)
            mock_get_all_tools.return_value = [search_stub]
            captured: dict = {}

            def _create(model, tools):
                captured["tools"] = tools
                return MagicMock()

            mock_create_agent.side_effect = _create

            # Act
            agent = self._make_agent(rag_backend=backend)
            await agent.initialize_async_resources()

            # Assert
            names = [t.name for t in captured["tools"]]
            self.assertIn("search_stub", names)
            self.assertIn("rag_search", names)
            self.assertEqual(len(captured["tools"]), 2)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_existing_rag_search_tool_wins_over_backend(self, mock_create_agent, mock_get_all_tools):
        # Arrange: MCP already provides a tool named rag_search.
        with tempfile.TemporaryDirectory() as tmp:
            backend = await _make_indexed_backend(tmp)
            mock_get_all_tools.return_value = [search_stub, rag_search]
            captured: dict = {}

            def _create(model, tools):
                captured["tools"] = tools
                return MagicMock()

            mock_create_agent.side_effect = _create

            # Act
            agent = self._make_agent(rag_backend=backend)
            await agent.initialize_async_resources()

            # Assert: backend tool skipped, MCP one kept (single rag_search).
            tools = captured["tools"]
            self.assertEqual([t.name for t in tools].count("rag_search"), 1)
            self.assertTrue(any(t is rag_search for t in tools))

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_no_rag_backend_leaves_tools_unchanged(self, mock_create_agent, mock_get_all_tools):
        # Arrange
        mock_get_all_tools.return_value = [search_stub]
        captured: dict = {}

        def _create(model, tools):
            captured["tools"] = tools
            return MagicMock()

        mock_create_agent.side_effect = _create

        # Act
        agent = self._make_agent(rag_backend=None)
        await agent.initialize_async_resources()

        # Assert
        self.assertEqual(captured["tools"], [search_stub])


if __name__ == "__main__":
    unittest.main()
