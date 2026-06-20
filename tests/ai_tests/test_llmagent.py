import unittest
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, SystemMessage

from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.ai.llmbot import FileTooLargeError
from manolo_bot.config import Config


class TestLlmAgent(unittest.IsolatedAsyncioTestCase):
    def get_basic_llm_agent(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()
        mock_llm.bind_tools.return_value = mock_llm
        # Ensure token counting works in tests
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        mock_llm.get_num_tokens_from_messages = MagicMock(return_value=10)

        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.google_api_key = None
        mock_config.openai_api_key = None
        mock_config.openai_api_base_url = None
        mock_config.context_max_tokens = 4000
        mock_config.web_content_request_timeout = 30
        # MCP configuration
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = {}
        mock_config.use_tools = True
        mock_config.can_use_tavily_search = False
        mock_config.max_document_size = 10 * 1024 * 1024
        mock_config.max_voice_size = 10 * 1024 * 1024
        mock_messages_storage = MagicMock()
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        agent = LLMAgent(mock_llm, mock_config, system_instructions, mock_messages_storage)
        return agent

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_llm_agent_initialization(self, mock_create_agent, mock_get_all_tools):
        # Arrange
        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.use_tools = True
        mock_config.context_max_tokens = 4000
        mock_config.web_content_request_timeout = 30
        # MCP configuration
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = {}
        mock_config.can_use_tavily_search = False
        mock_config.max_document_size = 10 * 1024 * 1024
        mock_config.max_voice_size = 10 * 1024 * 1024
        mock_messages_storage = MagicMock()

        mock_tools = ["tool1", "tool2"]
        mock_get_all_tools.return_value = mock_tools

        # Make create_agent return our mock_agent
        mock_agent = MagicMock()

        def create_agent_side_effect(model, tools):
            # Verify the model is our LLM (bound with tools)
            # The LLM gets bound with tools in LLMBot.__init__
            self.assertIsNotNone(model)
            self.assertEqual(tools, mock_tools)
            return mock_agent

        mock_create_agent.side_effect = create_agent_side_effect

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        mock_llm.get_num_tokens_from_messages = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        agent = LLMAgent(mock_llm, mock_config, system_instructions, mock_messages_storage)
        await agent.initialize_async_resources()

        # Assert
        mock_get_all_tools.assert_called_once()
        mock_create_agent.assert_called_once()
        self.assertEqual(agent.agent, mock_agent)
        self.assertEqual(agent.system_instructions, system_instructions)
        self.assertEqual(agent.bot_config, mock_config)

    async def test_answer_document_message__success(self):
        # Arrange
        from manolo_bot.storage.documents.base import BaseDocumentStorage

        mock_ai_message = AIMessage(content="Agent analysis")
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_ai_message]})

        agent = self.get_basic_llm_agent()
        agent.agent = mock_agent
        agent.documents_storage = MagicMock(spec=BaseDocumentStorage)
        agent.documents_storage.store = AsyncMock()

        with patch("manolo_bot.ai.llmbot.LLMBot._download_file", return_value="something"):
            with patch("manolo_bot.ai.llmbot.DocumentLoader") as mock_loader_class:
                mock_loader_class.return_value.extract_text.return_value = "text"
                mock_loader_class.SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt", "md", "csv"]

                # Act
                result = await agent.answer_document_message(1, "prompt", "http://url", "file.pdf")

                # Assert
                self.assertEqual(result.content, "Agent analysis")
                # Filename is now unique (contains uuid), so we use ANY or check it starts with uuid
                agent.documents_storage.store.assert_called_once_with(1, unittest.mock.ANY, "text")
                actual_filename = agent.documents_storage.store.call_args[0][1]
                self.assertTrue(actual_filename.endswith("_file.pdf"))
                self.assertEqual(len(actual_filename.split("_")[0]), 8)

    async def test_answer_document_message__unsupported_format(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        agent.generate_feedback_message = AsyncMock(return_value="Unsupported format message")

        # Act
        result = await agent.answer_document_message(1, "prompt", "http://url", "test.exe")

        # Assert
        self.assertEqual(result.content, "Unsupported format message")
        agent.generate_feedback_message.assert_called_once()
        # Verify it includes the extension in the prompt
        args = agent.generate_feedback_message.call_args[0][0]
        self.assertIn(".exe", args)

    async def test_answer_message(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        chat_id = 1
        user_message = "Hello, how are you?"

        # Create a mock response from the agent
        mock_ai_message = AIMessage(content="I'm doing well, thank you!")
        mock_agent = unittest.mock.MagicMock()
        mock_agent.ainvoke = unittest.mock.AsyncMock(return_value={"messages": [mock_ai_message]})
        agent.agent = mock_agent

        # Act
        response = await agent.answer_message(chat_id, user_message)

        # Assert
        self.assertEqual(response.content, "I'm doing well, thank you!")
        mock_agent.ainvoke.assert_called_once()

    async def test_answer_image_message(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        chat_id = 1
        text = "What is in this image?"
        image_url = "https://example.com/image.jpg"

        # Mock aiohttp session and response
        mock_session = unittest.mock.MagicMock()
        mock_response = unittest.mock.AsyncMock()
        mock_response.read.return_value = b"fake_image_data"
        mock_response.status = 200
        mock_response.raise_for_status = unittest.mock.MagicMock()

        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session

        # Mock agent response
        mock_ai_message = AIMessage(content="I see a cat in the image.")
        mock_agent = unittest.mock.MagicMock()
        mock_agent.ainvoke = unittest.mock.AsyncMock(return_value={"messages": [mock_ai_message]})
        agent.agent = mock_agent

        with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            response = await agent.answer_image_message(chat_id, text, image_url)

            # Assert
            self.assertEqual(response.content, "I see a cat in the image.")
            mock_agent.ainvoke.assert_called_once()

    async def test_answer_voice_message(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        chat_id = 1
        text = "What did I say?"
        audio_url = "https://example.com/audio.ogg"
        fake_audio_data = b"fake_audio_data"

        # Mock agent response
        mock_ai_message = AIMessage(content="You said 'Hello'.")
        mock_agent = unittest.mock.MagicMock()
        mock_agent.ainvoke = unittest.mock.AsyncMock(return_value={"messages": [mock_ai_message]})
        agent.agent = mock_agent

        with patch("manolo_bot.ai.llmbot.LLMBot._download_file", return_value=fake_audio_data):
            # Act
            response = await agent.answer_voice_message(chat_id, text, audio_url)

            # Assert
            self.assertEqual(response.content, "You said 'Hello'.")
            mock_agent.ainvoke.assert_called_once()

    async def test_answer_voice_message__too_large(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        agent.generate_feedback_message = AsyncMock(return_value="Voice message too long")
        agent.bot_config.max_voice_size = 100  # Small limit

        with patch(
            "manolo_bot.ai.llmbot.LLMBot._download_file",
            side_effect=FileTooLargeError("Voice message is too large (200 bytes)"),
        ):
            # Act
            response = await agent.answer_voice_message(1, "prompt", "http://example.com/audio.ogg")

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

        mapping = {"{name}": lambda bot: "ManoloAgent", "{day}": lambda bot: "Tuesday"}

        # Act
        agent = LLMAgent(
            mock_llm, mock_bot_config, system_instructions, mock_messages_storage, system_instructions_mapping=mapping
        )

        # Assert
        # Check that property returns replaced content
        self.assertEqual(agent.system_instructions[0].content, "Hello ManoloAgent, today is Tuesday")
        # Check that original instructions are NOT modified
        self.assertEqual(agent._system_instructions[0].content, original_content)


if __name__ == "__main__":
    unittest.main()
