import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import SystemMessage

from manolo_bot.ai.llmdeepagent import LLMDeepAgent
from manolo_bot.config import Config


class TestLLMDeepAgent(unittest.IsolatedAsyncioTestCase):
    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_initialization_uses_create_deep_agent(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        # Arrange
        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.use_tools = True
        mock_config.context_max_tokens = 4000
        mock_config.web_content_request_timeout = 30
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = {}
        mock_config.can_use_tavily_search = False
        mock_config.max_document_size = 10 * 1024 * 1024
        mock_config.max_voice_size = 10 * 1024 * 1024

        mock_messages_storage = MagicMock()

        mock_tools = ["tool1", "tool2"]
        mock_get_all_tools.return_value = mock_tools

        mock_agent = MagicMock()
        mock_create_deep_agent.return_value = mock_agent

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        agent = LLMDeepAgent(mock_llm, mock_config, system_instructions, mock_messages_storage)
        await agent.initialize_async_resources()

        # Assert
        mock_get_all_tools.assert_called_once()
        mock_create_deep_agent.assert_called_once()
        call_kwargs = mock_create_deep_agent.call_args.kwargs
        self.assertEqual(call_kwargs["model"], mock_llm)
        self.assertEqual(call_kwargs["subagents"], [])
        middleware = call_kwargs["middleware"]
        middleware_names = [type(m).__name__ for m in middleware]
        self.assertIn("FilesystemMiddleware", middleware_names)
        self.assertIn("TodoListMiddleware", middleware_names)
        self.assertEqual(agent.agent, mock_agent)

    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_deep_agent_can_receive_custom_backend(self, mock_create_agent):
        """Verify backend injection works."""
        mock_backend = MagicMock()
        mock_backend.backend = MagicMock()  # the actual BackendProtocol
        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.use_tools = False
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = {}
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        system_instructions = [SystemMessage(content="Test")]

        agent = LLMDeepAgent(mock_llm, mock_config, system_instructions, mock_messages_storage, backend=mock_backend)

        self.assertIs(agent._backend, mock_backend.backend)
        self.assertIs(agent._backend_wrapper, mock_backend)

    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_deep_agent_inherits_answer_message_from_llm_agent(self, mock_create_agent):
        """LLMDeepAgent should reuse LLMAgent's answer_message via self.agent.ainvoke."""
        mock_ai_message = MagicMock()
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_ai_message]})

        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.use_tools = True
        mock_config.context_max_tokens = 4000
        mock_config.web_content_request_timeout = 30
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = {}
        mock_config.can_use_tavily_search = False
        mock_config.max_document_size = 10 * 1024 * 1024
        mock_config.max_voice_size = 10 * 1024 * 1024

        mock_messages_storage = MagicMock()
        mock_messages_storage.messages = []

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        agent = LLMDeepAgent(mock_llm, mock_config, system_instructions, mock_messages_storage)
        agent.agent = mock_agent

        response = await agent.answer_message(1, "Hello")

        self.assertEqual(response, mock_ai_message)
        mock_agent.ainvoke.assert_awaited_once()

    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_deep_agent_defaults_to_state_backend(self, mock_create_agent):
        """When no backend is provided, defaults to StateBackend."""
        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.use_tools = False
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = {}
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        system_instructions = [SystemMessage(content="Test")]

        agent = LLMDeepAgent(mock_llm, mock_config, system_instructions, mock_messages_storage)

        self.assertIsNone(agent._backend_wrapper)
        from deepagents.backends import StateBackend

        self.assertIsInstance(agent._backend, StateBackend)

    async def test_deep_agent_clean_context_clears_backend_wrapper(self):
        """clean_context should also clear the backend wrapper."""
        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.use_tools = False
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = {}
        mock_messages_storage = MagicMock()
        mock_messages_storage.chat_id = 123
        mock_messages_storage.clear_messages = AsyncMock()
        mock_llm = MagicMock()
        system_instructions = [SystemMessage(content="Test")]

        mock_backend_wrapper = MagicMock()
        mock_backend_wrapper.backend = MagicMock()
        mock_backend_wrapper.clear = AsyncMock()

        agent = LLMDeepAgent(
            mock_llm, mock_config, system_instructions, mock_messages_storage, backend=mock_backend_wrapper
        )
        agent.documents_storage = None

        await agent.clean_context()

        mock_backend_wrapper.clear.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
