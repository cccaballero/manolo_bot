import base64
import unittest
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.llmbot import FileTooLargeError, LLMBot
from manolo_bot.config import Config
from manolo_bot.storage.messages.base import SUMMARY_PREFIX
from manolo_bot.storage.messages.memory_storage import MemoryMessagesStorage


class TestLlmBot(unittest.IsolatedAsyncioTestCase):
    def get_basic_llm_bot(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()
        mock_llm.bind_tools.return_value = mock_llm
        # Ensure token counting works in tests by returning an int, not a mock
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        mock_llm.get_num_tokens_from_messages = MagicMock(return_value=10)

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
        mock_llm.get_num_tokens_from_messages = MagicMock(return_value=10)

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
        # Arrange: pure-local counting (~chars/4 + per-message overhead).
        llm_bot = self.get_basic_llm_bot()
        messages = [HumanMessage(content="Hello"), HumanMessage(content="World")]

        # Act
        result = llm_bot.count_tokens(messages)

        # Assert: local estimate, provider hook never consulted.
        # ("Hello"/"World" = 5 chars + "human" role = 10 chars -> ceil(10/4)=3 +3 = 6 each.)
        self.assertEqual(result, 12)
        llm_bot.llm.get_num_tokens_from_messages.assert_not_called()

    def test_count_tokens__with_list_content(self):
        # Arrange: image_url blocks are flat-rated, base64 length must not matter.
        llm_bot = self.get_basic_llm_bot()
        small_payload = "a" * 100
        large_payload = "a" * 100000
        small_msg = HumanMessage(
            content=[
                {"type": "text", "text": "Test"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{small_payload}"}},
            ]
        )
        large_msg = HumanMessage(
            content=[
                {"type": "text", "text": "Test"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{large_payload}"}},
            ]
        )

        # Act
        small_count = llm_bot.count_tokens([small_msg])
        large_count = llm_bot.count_tokens([large_msg])

        # Assert: identical counts regardless of base64 size; provider never called.
        self.assertEqual(small_count, large_count)
        llm_bot.llm.get_num_tokens_from_messages.assert_not_called()
        # Image block left as-is (no sanitization mutation).
        self.assertIn(large_payload, large_msg.content[1]["image_url"]["url"])

    def test_count_tokens__with_audio_media(self):
        # Arrange: 16000 bytes of audio -> ~21KB base64. The approximate counter
        # measures unknown blocks with len(repr(block)), which would explode;
        # sanitization must replace the payload with a placeholder first.
        llm_bot = self.get_basic_llm_bot()
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

        # Act
        result = llm_bot.count_tokens(messages)

        # Assert: no phantom-token explosion, provider never called, original kept.
        self.assertLess(result, 1000)
        llm_bot.llm.get_num_tokens_from_messages.assert_not_called()
        self.assertEqual(messages[0].content[1]["data"], encoded_audio)

    def test_count_tokens__sanitizes_large_data_blocks_without_mutating(self):
        # Arrange: generalized data/video/file_data/inline_data shapes.
        llm_bot = self.get_basic_llm_bot()
        big = "x" * 5000
        messages = [
            HumanMessage(content=[{"type": "video", "data": big}]),
            HumanMessage(content=[{"type": "file", "file_data": big}]),
            HumanMessage(content=[{"type": "custom", "inline_data": big}]),
            HumanMessage(content=[{"type": "video", "source": {"data": big}}]),
        ]

        # Act
        result = llm_bot.count_tokens(messages)

        # Assert: all placeholders, tiny total; stored messages untouched.
        self.assertLess(result, 1000)
        llm_bot.llm.get_num_tokens_from_messages.assert_not_called()
        self.assertEqual(messages[0].content[0]["data"], big)
        self.assertEqual(messages[1].content[0]["file_data"], big)
        self.assertEqual(messages[2].content[0]["inline_data"], big)
        self.assertEqual(messages[3].content[0]["source"]["data"], big)

    def test_effective_token_limit__is_85_percent_of_configured_max(self):
        # Arrange
        bot, _ = self._make_summarization_bot([HumanMessage(content="hi")], context_max_tokens=800)

        # Act / Assert: user-facing config unchanged, truncation gates at 85%.
        self.assertEqual(bot.bot_config.context_max_tokens, 800)
        self.assertEqual(bot._effective_token_limit, 680)

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
        llm_bot.llm.get_num_tokens_from_messages = MagicMock(return_value=10)

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

    def _make_summarization_bot(self, messages, *, context_max_tokens=800, summarization=True, summary_max_tokens=512):
        """Build an LLMBot with a real MemoryMessagesStorage and a local
        token-count mock that forces truncation (100 tokens per message).

        ``LLMBot.count_tokens`` is now pure-local, so truncation tests mock it
        directly on the instance instead of the (dead) provider
        ``llm.get_num_tokens_from_messages`` hook."""
        storage = MemoryMessagesStorage(bot_uuid="test-bot", chat_id=424242)
        for msg in messages:
            storage.add_message(msg)

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="SUMMARIZED CONTENT"))
        mock_llm.bind_tools.return_value = mock_llm

        bot_config = BotConfig(
            bot_uuid="test-bot",
            bot_name="TestBot",
            bot_username="test_bot",
            bot_token="123456:ABC",
            user_id=0,
            context_max_tokens=context_max_tokens,
            context_summarization=summarization,
            summary_max_tokens=summary_max_tokens,
            summary_keep_messages=6,
        )
        bot = LLMBot(mock_llm, bot_config, [SystemMessage(content="You are a helpful assistant")], storage)
        bot.count_tokens = MagicMock(side_effect=lambda msgs: 100 * len(msgs))
        return bot, storage

    async def test_truncate_chat_context_summarizes_oldest_messages(self):
        # Arrange: 10 messages, 100 tokens each = 1000 > 800 limit. Retention
        # is reduced so that kept messages + a worst-case capped summary fit
        # in a single summarization pass: 600+512 > 800 -> keep_n shrinks to 2.
        messages = [HumanMessage(content=f"message-{i}") for i in range(10)]
        bot, storage = self._make_summarization_bot(messages)

        # Act
        await bot.truncate_chat_context()

        # Assert: a summarization prompt was sent containing the dropped messages.
        call_args = bot.llm.ainvoke.call_args[0][0]
        self.assertEqual(len(call_args), 1)
        prompt = call_args[0].content
        self.assertIn("message-0", prompt)
        self.assertIn("message-7", prompt)
        self.assertIn("Summarize the key information", prompt)
        # The most recent 2 messages are kept intact.
        self.assertNotIn("message-8", prompt)

        # Assert: summary persisted at the front, folded messages removed.
        self.assertEqual(storage.get_summary(), "SUMMARIZED CONTENT")
        remaining = storage.messages
        self.assertEqual(len(remaining), 3)
        self.assertIsInstance(remaining[0], SystemMessage)
        self.assertTrue(remaining[0].content.startswith(SUMMARY_PREFIX))
        self.assertEqual([m.content for m in remaining[1:]], ["message-8", "message-9"])

    async def test_truncate_chat_context_does_not_drop_when_summarization_succeeds(self):
        # Arrange: 50 messages x100 tokens = 5000 > 4000 limit. One summarization
        # pass (fold all but the last 6) must bring it under budget, so no
        # message may be dropped without summarization.
        messages = [HumanMessage(content=f"message-{i}") for i in range(50)]
        bot, storage = self._make_summarization_bot(messages, context_max_tokens=4000)

        # Act
        await bot.truncate_chat_context()

        # Assert: exactly one summarization call, summary + kept recent
        # messages remain, nothing silently dropped.
        self.assertEqual(bot.llm.ainvoke.await_count, 1)
        remaining = storage.messages
        self.assertEqual(len(remaining), 7)
        self.assertEqual(storage.get_summary(), "SUMMARIZED CONTENT")
        self.assertEqual([m.content for m in remaining[1:]], [f"message-{i}" for i in range(44, 50)])

    async def test_truncate_chat_context_summarizes_few_large_messages(self):
        # Arrange: only 8 messages but over the limit (8 x100 = 800 tokens,
        # limit 500). Retention shrinks to the floor of 2 so that everything
        # else is summarized in a single pass instead of being dropped.
        messages = [HumanMessage(content=f"message-{i}") for i in range(8)]
        bot, storage = self._make_summarization_bot(messages, context_max_tokens=500)

        # Act
        await bot.truncate_chat_context()

        # Assert: exactly one summarization call; most content was summarized
        # and nothing had to be dropped.
        self.assertEqual(bot.llm.ainvoke.await_count, 1)
        self.assertEqual(storage.get_summary(), "SUMMARIZED CONTENT")
        remaining = storage.messages
        self.assertEqual(len(remaining), 3)
        self.assertIsInstance(remaining[0], SystemMessage)
        self.assertTrue(remaining[0].content.startswith(SUMMARY_PREFIX))
        # Only the floor retention (2 newest messages) is kept intact.
        self.assertEqual([m.content for m in remaining[1:]], ["message-6", "message-7"])

    async def test_truncate_chat_context_falls_back_to_drop_oldest_on_failure(self):
        # Arrange: summarizer raises -> old drop-oldest behavior.
        messages = [HumanMessage(content=f"message-{i}") for i in range(10)]
        bot, storage = self._make_summarization_bot(messages)
        bot.llm.ainvoke = AsyncMock(side_effect=Exception("summarizer boom"))

        # Act: must not raise.
        await bot.truncate_chat_context()

        # Assert: dropped oldest until under the effective budget
        # (int(800 * 0.85) = 680, 1000 -> 680 = 6 messages).
        self.assertEqual(len(storage.messages), 6)
        self.assertIsNone(storage.get_summary())

    async def test_truncate_chat_context_disabled_drops_oldest(self):
        # Arrange: summarization disabled -> drop-oldest directly.
        messages = [HumanMessage(content=f"message-{i}") for i in range(10)]
        bot, storage = self._make_summarization_bot(messages, summarization=False)

        # Act
        await bot.truncate_chat_context()

        # Assert: dropped oldest until under the effective budget (6 messages).
        self.assertEqual(len(storage.messages), 6)
        bot.llm.ainvoke.assert_not_awaited()

    async def test_truncate_chat_context_incremental_summary(self):
        # Arrange: an existing summary plus 10 messages -> incremental prompt.
        messages = [HumanMessage(content=f"message-{i}") for i in range(10)]
        bot, storage = self._make_summarization_bot(messages)
        storage.set_summary("OLD SUMMARY")

        # Act
        await bot.truncate_chat_context()

        # Assert: prompt references the existing summary and asks to extend it.
        call_args = bot.llm.ainvoke.call_args[0][0]
        prompt = call_args[0].content
        self.assertIn("OLD SUMMARY", prompt)
        self.assertIn("Extend the summary", prompt)
        # The summary itself is never folded into the new summary.
        self.assertNotIn("CONVERSATION SUMMARY", prompt)
        # New summary replaces the old one.
        self.assertEqual(storage.get_summary(), "SUMMARIZED CONTENT")

    async def test_truncate_chat_context_caps_summary_length(self):
        # Arrange: summary token count is huge -> hard-truncated proportionally.
        messages = [HumanMessage(content=f"message-{i}") for i in range(10)]
        bot, storage = self._make_summarization_bot(messages)

        def token_count(msgs):
            if len(msgs) == 1 and isinstance(msgs[0], SystemMessage):
                return 10000  # summary far over the 512 budget
            return 100 * len(msgs)

        bot.count_tokens = MagicMock(side_effect=token_count)
        bot.llm.ainvoke = AsyncMock(return_value=AIMessage(content="X" * 1000))

        # Act
        await bot.truncate_chat_context()

        # Assert: summary truncated to ~len * 512/10000 = 51 chars.
        summary = storage.get_summary()
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(len(summary), 51)
        self.assertEqual(summary, "X" * 51)

    def _make_auth_bot(self, **bot_config_overrides):
        """Build an LLMBot with a real BotConfig for header-auth tests."""
        storage = MemoryMessagesStorage(bot_uuid="test-bot", chat_id=424242)
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        mock_llm.get_num_tokens_from_messages = MagicMock(return_value=10)
        bot_config = BotConfig(
            bot_uuid="test-bot",
            bot_name="TestBot",
            bot_username="test_bot",
            bot_token="123456:ABC",
            user_id=0,
            **bot_config_overrides,
        )
        return LLMBot(
            mock_llm,
            bot_config,
            [SystemMessage(content="You are a helpful assistant")],
            storage,
        )

    def _make_fake_session(self, payload: bytes = b"data"):
        """Mocked aiohttp session whose get() returns an async-CM response."""

        class _FakeContent:
            async def iter_chunked(self, _size):
                yield payload

        class _FakeResponse:
            headers = {}
            content = _FakeContent()

            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=_FakeResponse())
        return mock_session

    def test_download_headers__defaults_to_empty(self):
        bot = self._make_auth_bot()
        self.assertEqual(bot._download_headers(), {})

    def test_download_headers__explicit_headers(self):
        bot = self._make_auth_bot(attachment_headers={"Authorization": "Bearer secret-token"})
        self.assertEqual(bot._download_headers(), {"Authorization": "Bearer secret-token"})

    async def test_download_file__sends_headers_when_set(self):
        bot = self._make_auth_bot(
            attachment_headers={"Authorization": "Bearer secret-token"},
        )
        mock_session = self._make_fake_session()
        result = await bot._download_file("http://example.com/file.pdf", mock_session)
        self.assertEqual(result, b"data")
        _, kwargs = mock_session.get.call_args
        self.assertEqual(
            kwargs.get("headers"),
            {"Authorization": "Bearer secret-token"},
        )

    async def test_download_file__sends_empty_headers_by_default(self):
        bot = self._make_auth_bot()
        mock_session = self._make_fake_session()
        await bot._download_file("http://example.com/file.pdf", mock_session)
        _, kwargs = mock_session.get.call_args
        self.assertEqual(kwargs.get("headers"), {})

    async def test_download_file__subclass_override_of_download_headers(self):
        class CustomBot(LLMBot):
            def _download_headers(self):
                return {"X-Custom-Scheme": "abc123"}

        bot = self._make_auth_bot()
        custom_bot = CustomBot(bot.llm, bot.bot_config, bot._system_instructions, bot.messages_storage)
        mock_session = self._make_fake_session()
        await custom_bot._download_file("http://example.com/file.pdf", mock_session)
        _, kwargs = mock_session.get.call_args
        self.assertEqual(kwargs.get("headers"), {"X-Custom-Scheme": "abc123"})


if __name__ == "__main__":
    unittest.main()
