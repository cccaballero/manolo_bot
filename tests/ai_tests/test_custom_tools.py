import unittest
from unittest.mock import MagicMock

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.ai.llmbot import LLMBot


class TestCustomTools(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_llm.bind_tools.return_value = self.mock_llm
        self.mock_config = MagicMock()
        self.mock_config.use_tools = True
        self.mock_config.enable_mcp = False
        self.mock_storage = MagicMock()
        self.system_instructions = [SystemMessage(content="test")]

    @tool
    def my_custom_tool(query: str) -> str:
        """My custom tool."""
        return "custom result"

    def test_llm_bot_uses_custom_tools(self):
        # Arrange
        custom_tools = [self.my_custom_tool]

        # Act
        bot = LLMBot(
            llm=self.mock_llm,
            bot_config=self.mock_config,
            system_instructions=self.system_instructions,
            messages_storage=self.mock_storage,
            tools=custom_tools,
        )

        # Assert
        self.mock_llm.bind_tools.assert_called_with(custom_tools)
        self.assertEqual(bot.tools, custom_tools)

    async def test_llm_agent_uses_custom_tools(self):
        # Arrange
        custom_tools = [self.my_custom_tool]

        # We need to mock create_agent which is used inside initialize_async_resources
        with unittest.mock.patch("manolo_bot.ai.llmagent.create_agent") as mock_create_agent:
            agent = LLMAgent(
                llm=self.mock_llm,
                bot_config=self.mock_config,
                system_instructions=self.system_instructions,
                messages_storage=self.mock_storage,
                tools=custom_tools,
            )

            # Act
            await agent.initialize_async_resources()

            # Assert
            mock_create_agent.assert_called()
            # Check that the tools passed to create_agent include our custom tool
            args, kwargs = mock_create_agent.call_args
            passed_tools = kwargs.get("tools")
            self.assertIn(self.my_custom_tool, passed_tools)
