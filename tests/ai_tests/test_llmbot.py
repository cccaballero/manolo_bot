import base64
import unittest
import unittest.mock
from unittest.mock import AsyncMock, MagicMock

import aiohttp
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

    async def test_llmbot_answer_voice_message_success(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        llm_bot.llm = MagicMock()
        llm_bot.llm.ainvoke = AsyncMock()
        llm_bot.llm.get_num_tokens = MagicMock(return_value=10)

        chat_id = 1
        text = "Check this audio"
        audio_url = "http://example.com/audio.ogg"
        fake_audio_data = b"fake_audio_data"

        # Mock aiohttp
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=fake_audio_data)
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_context_manager)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        # Mock LLM response
        mock_ai_message = AIMessage(content="Heard it!")
        llm_bot.llm.ainvoke.return_value = mock_ai_message

        with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_session_cm):
            # Act
            response = await llm_bot.answer_voice_message(chat_id, text, audio_url)

        # Assert
        self.assertEqual(response, mock_ai_message)
        self.assertEqual(len(llm_bot.messages_storage.add_message.call_args_list), 1)

        # In this test we use mock_messages_storage, so we check call args
        call_args = llm_bot.messages_storage.add_message.call_args[0][0]
        self.assertIsInstance(call_args, HumanMessage)
        content = call_args.content
        self.assertEqual(content[0]["text"], text)
        self.assertEqual(content[1]["type"], "media")
        self.assertEqual(content[1]["mime_type"], "audio/ogg")
        self.assertEqual(content[1]["data"], base64.b64encode(fake_audio_data).decode("utf-8"))

    async def test_answer_voice_message_failure(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        chat_id = 1
        text = "Fail test"
        audio_url = "http://example.com/audio.ogg"

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Network error"))
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_session_cm):
            # Act
            response = await llm_bot.answer_voice_message(chat_id, text, audio_url)

        # Assert
        self.assertEqual(response.content, "NO_ANSWER")
        # Ensure no message was added to storage
        self.assertEqual(len(llm_bot.messages_storage.add_message.call_args_list), 0)


if __name__ == "__main__":
    unittest.main()
