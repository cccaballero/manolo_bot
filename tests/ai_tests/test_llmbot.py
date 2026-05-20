import unittest
import unittest.mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai.llmbot import LLMBot
from config import Config


class TestLlmBot(unittest.IsolatedAsyncioTestCase):
    def get_basic_llm_bot(self):
        mock_llm = unittest.mock.MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_config = unittest.mock.MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.google_api_key = None
        mock_config.openai_api_key = None
        mock_config.openai_api_base_url = None
        mock_config.context_max_tokens = 4000
        # MCP configuration
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = {}
        mock_config.use_tools = False
        mock_config.can_use_tavily_search = False
        mock_messages_storage = unittest.mock.MagicMock()
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        llm_bot = LLMBot(mock_llm, mock_config, system_instructions, mock_messages_storage)  # Initialize the bot
        llm_bot.chats = {1: {"messages": []}}
        return llm_bot

    def test_llm_bot__init_with_ollama_ai(self):
        # Arrange
        mock_llm = unittest.mock.MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_bot_config = unittest.mock.MagicMock()
        mock_bot_config.enable_mcp = False
        mock_bot_config.use_tools = False
        mock_bot_config.can_use_tavily_search = False
        mock_messages_storage = unittest.mock.MagicMock()

        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        bot = LLMBot(mock_llm, mock_bot_config, system_instructions, mock_messages_storage)

        # Assert
        self.assertEqual(bot.llm, mock_llm)
        self.assertEqual(bot.bot_config, mock_bot_config)
        self.assertEqual(bot.system_instructions, system_instructions)
        self.assertEqual(bot.messages_storage, mock_messages_storage)

    def test_llm_bot__init_with_google_ai(self):
        # Arrange
        mock_llm = unittest.mock.MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_bot_config = unittest.mock.MagicMock()
        mock_bot_config.enable_mcp = False
        mock_bot_config.use_tools = False
        mock_bot_config.can_use_tavily_search = False
        mock_messages_storage = unittest.mock.MagicMock()

        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        bot = LLMBot(mock_llm, mock_bot_config, system_instructions, mock_messages_storage)

        # Assert
        self.assertEqual(bot.llm, mock_llm)
        self.assertEqual(bot.bot_config, mock_bot_config)
        self.assertEqual(bot.system_instructions, system_instructions)
        self.assertEqual(bot.messages_storage, mock_messages_storage)

    def test_llm_bot__init_with_openai_ai(self):
        # Arrange
        mock_llm = unittest.mock.MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_bot_config = unittest.mock.MagicMock()
        mock_bot_config.enable_mcp = False
        mock_bot_config.use_tools = False
        mock_bot_config.can_use_tavily_search = False
        mock_messages_storage = unittest.mock.MagicMock()

        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        bot = LLMBot(mock_llm, mock_bot_config, system_instructions, mock_messages_storage)

        # Assert
        self.assertEqual(bot.llm, mock_llm)
        self.assertEqual(bot.bot_config, mock_bot_config)
        self.assertEqual(bot.system_instructions, system_instructions)
        self.assertEqual(bot.messages_storage, mock_messages_storage)

    def test_extract_url__extract_valid_url(self):
        # Arrange
        bot = self.get_basic_llm_bot()
        text = "Check out this website: https://example.com/page?param=value"

        # Act
        result = bot._extract_url(text)

        # Assert
        self.assertEqual(result, "https://example.com/page?param=value")

    def test_extract_url__return_none_when_no_url(self):
        # Arrange
        bot = self.get_basic_llm_bot()
        text = "This is a text without any URL in it"

        # Act
        result = bot._extract_url(text)

        # Assert
        self.assertIsNone(result)

    def test_remove_url__removes_http_urls_from_text(self):
        # Arrange
        bot = self.get_basic_llm_bot()
        text_with_urls = "Check this link https://example.com and this one http://test.org/page?param=1"

        # Act
        result = bot._remove_urls(text_with_urls)

        # Assert
        self.assertEqual("Check this link  and this one ", result)

    def test_remove_url__handles_empty_string_input(self):
        # Arrange
        bot = self.get_basic_llm_bot()
        empty_text = ""

        # Act
        result = bot._remove_urls(empty_text)

        # Assert
        self.assertEqual("", result)

    def test_count_tokens__with_string_content(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        messages = [HumanMessage(content="Hello"), HumanMessage(content="World")]
        llm_bot.llm = unittest.mock.MagicMock()
        llm_bot.llm.get_num_tokens = unittest.mock.MagicMock()
        llm_bot.llm.get_num_tokens.return_value = 2

        # Act
        result = llm_bot.count_tokens(messages)

        # Assert
        self.assertEqual(result, 2)
        llm_bot.llm.get_num_tokens.assert_called_once_with("\n Hello\n World")

    def test_count_tokens__with_list_content(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content=[
                    {"type": "text", "text": "Test"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ]
            ),
        ]
        mock_llm = unittest.mock.MagicMock()
        llm_bot.llm = mock_llm
        mock_llm.get_num_tokens.return_value = 3

        # Act
        result = llm_bot.count_tokens(messages)

        # Assert
        self.assertEqual(result, 258 + 3)
        mock_llm.get_num_tokens.assert_called_once_with("\n Hello\n Test")

    async def test_generate_feedback_message__success_message(self):
        # Arrange
        mock_config = unittest.mock.MagicMock(spec=Config)
        mock_config.preferred_language = "Spanish"
        # MCP configuration
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = "{}"
        mock_config.use_tools = False

        llm_bot = self.get_basic_llm_bot()
        llm_bot.config = mock_config

        expected_response = AIMessage(content="¡Contexto de chat borrado con éxito!")
        llm_bot.llm = unittest.mock.Mock()
        llm_bot.llm.ainvoke = unittest.mock.AsyncMock(return_value=expected_response)

        # Act
        result = await llm_bot.generate_feedback_message("some prompt")

        # Assert
        self.assertEqual(result, "¡Contexto de chat borrado con éxito!")
        llm_bot.llm.ainvoke.assert_called_once()

    async def test_generate_feedback_message__truncates_long_messages(self):
        # Arrange
        mock_config = unittest.mock.MagicMock(spec=Config)
        mock_config.preferred_language = "English"
        # MCP configuration
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = "{}"
        mock_config.use_tools = False

        llm_bot = self.get_basic_llm_bot()
        llm_bot.config = mock_config

        # Create a response that's longer than 200 characters
        long_message = "This is a very long message that exceeds the 200 character limit. " * 5
        expected_response = AIMessage(content=long_message)
        llm_bot.llm = unittest.mock.Mock()
        llm_bot.llm.ainvoke = unittest.mock.AsyncMock(return_value=expected_response)

        # Act
        result = await llm_bot.generate_feedback_message("success")

        # Assert
        self.assertEqual(len(result), 200)  # 197 chars + 3 for "..."
        self.assertTrue(result.endswith("..."))
        self.assertEqual(result, long_message[:197] + "...")

    # TODO: find a way to test process_message_buffer logic, probably the function needs a refactor to make it more
    #  testable


if __name__ == "__main__":
    unittest.main()
