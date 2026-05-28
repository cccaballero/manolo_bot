import base64
import unittest
import unittest.mock
from unittest.mock import AsyncMock, MagicMock

import aiohttp
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.config import Config


class TestLlmAgent(unittest.IsolatedAsyncioTestCase):
    def get_basic_llm_agent(self):
        mock_llm = MagicMock()
        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.google_api_key = None
        mock_config.openai_api_key = None
        mock_config.openai_api_base_url = None
        mock_config.context_max_tokens = 4000
        mock_config.web_content_request_timeout = 30
        mock_config.use_tools = False
        mock_config.can_use_tavily_search = False
        # MCP configuration
        mock_config.enable_mcp = False
        mock_config.mcp_servers_config = {}
        mock_messages_storage = MagicMock()
        # Make messages_storage actually store messages
        mock_messages_storage.messages = []
        mock_messages_storage.add_message = lambda msg: mock_messages_storage.messages.append(msg)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Create the agent with the mock LLM
        agent = LLMAgent(mock_llm, mock_config, system_instructions, mock_messages_storage)
        agent.chats = {1: {"messages": []}}

        # Mock the count_tokens method to return a fixed value
        def mock_count_tokens(messages):
            return 100  # Return a fixed token count that's less than context_max_tokens

        agent.count_tokens = mock_count_tokens
        return agent

    @unittest.mock.patch("manolo_bot.ai.llmagent.create_agent")
    @unittest.mock.patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    async def test_llm_agent_initialization(self, mock_get_all_tools, mock_create_agent):
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
        self.assertEqual(response, mock_ai_message)
        mock_agent.ainvoke.assert_called_once()

        # Check that the message history was updated
        self.assertEqual(len(agent.messages_storage.messages), 1)
        self.assertEqual(agent.messages_storage.messages[0].content, user_message)

    async def test_answer_image_message_success(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        chat_id = 1
        text = "What's in this image?"
        image_url = "http://example.com/image.jpg"

        # Mock the aiohttp response
        mock_response = unittest.mock.MagicMock()
        mock_response.status = 200
        mock_response.raise_for_status = unittest.mock.MagicMock()  # This is synchronous, not async
        mock_response.read = unittest.mock.AsyncMock(return_value=b"fake_image_data")

        # Mock the session
        mock_session = unittest.mock.MagicMock()  # Not AsyncMock, session itself is not async
        mock_context_manager = unittest.mock.MagicMock()  # Not AsyncMock for the CM object itself
        mock_context_manager.__aenter__ = unittest.mock.AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_session.get = unittest.mock.MagicMock(return_value=mock_context_manager)

        # Mock the agent's response
        mock_ai_message = AIMessage(content="This is an image response")
        mock_agent = MagicMock()
        mock_agent.ainvoke = unittest.mock.AsyncMock(return_value={"messages": [mock_ai_message]})
        agent.agent = mock_agent

        with unittest.mock.patch("aiohttp.ClientSession") as mock_session_class:
            # Mock the session context manager
            mock_session_context_manager = unittest.mock.MagicMock()
            mock_session_context_manager.__aenter__ = unittest.mock.AsyncMock(return_value=mock_session)
            mock_session_context_manager.__aexit__ = unittest.mock.AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session_context_manager

            # Act
            response = await agent.answer_image_message(chat_id, text, image_url)

            # Assert
            self.assertEqual(response, mock_ai_message)
            mock_session.get.assert_called_once_with(image_url, timeout=unittest.mock.ANY)
            mock_agent.ainvoke.assert_called_once()

            # Check that the message history was updated with the image message
            self.assertEqual(len(agent.messages_storage.messages), 1)
            self.assertIsInstance(agent.messages_storage.messages[0], HumanMessage)

            # Check the image content in the message
            content = agent.messages_storage.messages[0].content
            self.assertEqual(len(content), 2)
            self.assertEqual(content[0]["text"], text)
            self.assertIn("data:image/jpeg;base64,", content[1]["image_url"]["url"])

    async def test_answer_image_message_request_failure(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        chat_id = 1
        text = "What's in this image?"
        image_url = "http://example.com/image.jpg"

        # Mock the session to raise aiohttp.ClientError in __aenter__
        mock_session = unittest.mock.MagicMock()  # Not AsyncMock, session itself is not async
        mock_context_manager = unittest.mock.MagicMock()  # Not AsyncMock for the CM object itself
        # Raise the exception in __aenter__ to simulate network error during request
        mock_context_manager.__aenter__ = unittest.mock.AsyncMock(
            side_effect=aiohttp.ClientError("Failed to fetch image")
        )
        mock_context_manager.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_session.get = unittest.mock.MagicMock(return_value=mock_context_manager)

        # Act
        with (
            unittest.mock.patch("aiohttp.ClientSession") as mock_session_class,
            self.assertLogs(level="ERROR") as log_context,
        ):
            # Mock the session context manager
            mock_session_context_manager = unittest.mock.MagicMock()
            mock_session_context_manager.__aenter__ = unittest.mock.AsyncMock(
                side_effect=aiohttp.ClientError("Failed to fetch image")
            )
            mock_session_context_manager.__aexit__ = unittest.mock.AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session_context_manager

            response = await agent.answer_image_message(chat_id, text, image_url)

        # Assert
        self.assertIsInstance(response, BaseMessage)
        self.assertEqual(response.content, "NO_ANSWER")
        self.assertEqual(response.type, "text")
        # Note: mock_session.get is not called because the error happens in __aenter__

        # Verify error was logged
        self.assertTrue(any("Failed to get image" in log for log in log_context.output))

        # Check that no messages were added to storage on failure
        self.assertEqual(len(agent.messages_storage.messages), 0)

    async def test_answer_voice_message_success(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        chat_id = 1
        text = "Agent, listen"
        audio_url = "http://example.com/audio.ogg"
        fake_audio_data = b"agent_audio_data"

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

        # Mock Agent response
        mock_ai_message = AIMessage(content="Agent heard it!")
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_ai_message]})
        agent.agent = mock_agent

        with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_session_cm):
            # Act
            response = await agent.answer_voice_message(chat_id, text, audio_url)

        # Assert
        self.assertEqual(response, mock_ai_message)
        mock_agent.ainvoke.assert_called_once()

        content = agent.messages_storage.messages[0].content
        self.assertEqual(content[0]["text"], text)
        self.assertEqual(content[1]["data"], base64.b64encode(fake_audio_data).decode("utf-8"))
