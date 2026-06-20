import base64
import unittest
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from manolo_bot.ai.llmbot import FileTooLargeError, LLMBot
from manolo_bot.config import Config


class TestLlmBot(unittest.IsolatedAsyncioTestCase):
    def get_basic_llm_bot(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()
        mock_llm.bind_tools.return_value = mock_llm
        # Ensure token counting works in tests by returning an int, not a mock
        mock_llm.get_num_tokens = MagicMock(return_value=10)

        mock_config = MagicMock(spec=Config)
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
        mock_config.max_document_size = 10 * 1024 * 1024
        mock_config.max_voice_size = 10 * 1024 * 1024
        mock_messages_storage = MagicMock()
        mock_messages_storage.messages = []

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

    async def test_answer_document_message__success(self):
        # Arrange
        from manolo_bot.storage.documents.base import BaseDocumentStorage

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Analysis"))
        mock_llm.get_num_tokens = MagicMock(return_value=10)

        mock_storage = MagicMock(spec=BaseDocumentStorage)
        mock_storage.store = AsyncMock()

        bot = self.get_basic_llm_bot()
        bot.llm = mock_llm
        bot.documents_storage = mock_storage

        with patch("manolo_bot.ai.llmbot.LLMBot._download_file", return_value=b"something"):
            with patch("manolo_bot.ai.document_loaders.DocumentLoader.extract_text_from_pdf", return_value="text"):
                with patch("manolo_bot.ai.llmbot.DocumentLoader") as mock_loader_class:
                    mock_loader_class.return_value.extract_text.return_value = "text"
                    mock_loader_class.SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt", "md", "csv"]

                    # Act
                    result = await bot.answer_document_message(1, "prompt", "http://url", "file.pdf")

                    # Assert
                    self.assertEqual(result.content, "Analysis")
                    # Filename is now unique (contains uuid), so we use ANY or check it starts with uuid
                    mock_storage.store.assert_called_once_with(1, unittest.mock.ANY, "text")
                    actual_filename = mock_storage.store.call_args[0][1]
                    self.assertTrue(actual_filename.endswith("_file.pdf"))
                    self.assertEqual(len(actual_filename.split("_")[0]), 8)

    async def test_answer_document_message__too_large(self):
        # Arrange
        bot = self.get_basic_llm_bot()
        bot.generate_feedback_message = AsyncMock(return_value="Message too large")
        bot.bot_config.max_document_size = 100  # Small limit

        with patch(
            "manolo_bot.ai.llmbot.LLMBot._download_file", side_effect=FileTooLargeError("File is too large (200 bytes)")
        ):
            with patch("manolo_bot.ai.document_loaders.DocumentLoader.extract_text_from_pdf", return_value="text"):
                # Act
                result = await bot.answer_document_message(1, "prompt", "http://url", "file.pdf")

                # Assert
                self.assertEqual(result.content, "Message too large")

    async def test_answer_document_message__unsupported_format(self):
        # Arrange
        bot = self.get_basic_llm_bot()
        bot.generate_feedback_message = AsyncMock(return_value="Unsupported format message")

        # Act
        result = await bot.answer_document_message(1, "prompt", "http://url", "test.exe")

        # Assert
        self.assertEqual(result.content, "Unsupported format message")
        bot.generate_feedback_message.assert_called_once()
        # Verify it includes the extension in the prompt
        args = bot.generate_feedback_message.call_args[0][0]
        self.assertIn(".exe", args)

    async def test_clean_context__clears_messages_and_documents(self):
        # Arrange
        mock_messages_storage = AsyncMock()
        mock_document_storage = AsyncMock()
        bot = self.get_basic_llm_bot()
        bot.messages_storage = mock_messages_storage
        bot.documents_storage = mock_document_storage

        # Act
        await bot.clean_context()

        # Assert
        mock_messages_storage.clear_messages.assert_called_once()
        mock_document_storage.clear.assert_called_once()

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
        self.assertEqual(result, 258 + 3)  # image (258) + text (3)
        mock_llm.get_num_tokens.assert_called_once_with("\n Hello\n Test")

    def test_count_tokens__with_audio_media(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        # 16000 bytes of audio data.
        # At 16kbps (2000 bytes/s), this is 8 seconds.
        # 8 seconds * 32 tokens/s = 256 tokens.
        audio_data = b"a" * 16000
        encoded_audio = base64.b64encode(audio_data).decode("utf-8")

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Listen to this"},
                    {"type": "media", "mime_type": "audio/ogg", "data": encoded_audio},
                ]
            )
        ]
        mock_llm = unittest.mock.MagicMock()
        llm_bot.llm = mock_llm
        mock_llm.get_num_tokens.return_value = 1

        # Act
        result = llm_bot.count_tokens(messages)

        # Assert
        self.assertEqual(result, 256 + 1)
        mock_llm.get_num_tokens.assert_called_once_with("\n Listen to this")

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

        # Mock LLM response
        mock_ai_message = AIMessage(content="Heard it!")
        llm_bot.llm.ainvoke.return_value = mock_ai_message

        with patch("manolo_bot.ai.llmbot.LLMBot._download_file", return_value=fake_audio_data):
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

    async def test_answer_voice_message__too_large(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        llm_bot.generate_feedback_message = AsyncMock(return_value="Voice message too long")
        llm_bot.bot_config.max_voice_size = 100  # Small limit

        with patch(
            "manolo_bot.ai.llmbot.LLMBot._download_file",
            side_effect=FileTooLargeError("Voice message is too large (200 bytes)"),
        ):
            # Act
            response = await llm_bot.answer_voice_message(1, "prompt", "http://example.com/audio.ogg")

            # Assert
            self.assertEqual(response.content, "Voice message too long")

    def test_system_instructions_mapping(self):
        # Arrange
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_bot_config = MagicMock()
        mock_bot_config.enable_mcp = False
        mock_bot_config.use_tools = False
        mock_messages_storage = MagicMock()

        original_content = "Hello {name}, today is {day}"
        system_instructions = [SystemMessage(content=original_content)]

        mapping = {"{name}": lambda bot: "Manolo", "{day}": lambda bot: "Monday"}

        # Act
        bot = LLMBot(
            mock_llm, mock_bot_config, system_instructions, mock_messages_storage, system_instructions_mapping=mapping
        )

        # Assert
        # Check that property returns replaced content
        self.assertEqual(bot.system_instructions[0].content, "Hello Manolo, today is Monday")
        # Check that original instructions are NOT modified
        self.assertEqual(bot._system_instructions[0].content, original_content)
        # Check that it returns a new list (deepcopy)
        self.assertIsNot(bot.system_instructions, bot._system_instructions)
        self.assertIsNot(bot.system_instructions[0], bot._system_instructions[0])


if __name__ == "__main__":
    unittest.main()
