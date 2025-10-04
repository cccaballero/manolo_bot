import unittest
import unittest.mock
from unittest.mock import MagicMock

import aiohttp
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.base import RunnableBinding

from ai.llmagent import LLMAgent
from config import Config


class TestLlmAgent(unittest.IsolatedAsyncioTestCase):
    def get_basic_llm_agent(self):
        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.google_api_key = None
        mock_config.openai_api_key = None
        mock_config.openai_api_base_url = None
        mock_config.context_max_tokens = 4000
        mock_config.web_content_request_timeout = 30
        mock_config.use_tools = False
        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Create a mock LLM for the agent
        mock_llm = MagicMock()

        # Mock the count_tokens method to return a fixed value
        def mock_count_tokens(messages):
            return 100  # Return a fixed token count that's less than context_max_tokens

        # Create the agent with the mock LLM
        agent = LLMAgent(mock_config, system_instructions)
        agent.llm = mock_llm
        agent.chats = {1: {"messages": []}}
        agent.count_tokens = mock_count_tokens
        return agent

    @unittest.mock.patch("ai.llmagent.create_react_agent")
    @unittest.mock.patch("ai.llmagent.get_tools")
    def test_llm_agent_initialization(self, mock_get_tools, mock_create_agent):
        # Arrange
        mock_config = MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.use_tools = True
        mock_config.context_max_tokens = 4000
        mock_config.web_content_request_timeout = 30

        mock_tools = ["tool1", "tool2"]
        mock_get_tools.return_value = mock_tools

        # Create a mock agent that will be returned by create_react_agent
        mock_agent = MagicMock()

        # Make create_react_agent return our mock_agent
        def create_react_agent_side_effect(model, tools):
            # Verify the model is our bound LLM with tools
            self.assertIsInstance(model, RunnableBinding)
            self.assertEqual(tools, mock_tools)
            return mock_agent

        mock_create_agent.side_effect = create_react_agent_side_effect

        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        agent = LLMAgent(mock_config, system_instructions)

        # Assert
        mock_get_tools.assert_called_once()
        mock_create_agent.assert_called_once()
        self.assertEqual(agent.agent, mock_agent)
        self.assertEqual(agent.system_instructions, system_instructions)
        self.assertEqual(agent.config, mock_config)

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

        # Check that the chat history was updated
        self.assertEqual(len(agent.chats[chat_id]["messages"]), 1)
        self.assertIsInstance(agent.chats[chat_id]["messages"][0], HumanMessage)
        self.assertEqual(agent.chats[chat_id]["messages"][0].content, user_message)

    async def test_answer_image_message_success(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        chat_id = 1
        text = "What's in this image?"
        image_url = "http://example.com/image.jpg"

        # Mock the aiohttp response
        mock_response = unittest.mock.AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = unittest.mock.AsyncMock()
        mock_response.read = unittest.mock.AsyncMock(return_value=b"fake_image_data")

        # Mock the session
        mock_session = unittest.mock.AsyncMock()
        mock_context_manager = unittest.mock.AsyncMock()
        mock_context_manager.__aenter__ = unittest.mock.AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_session.get.return_value = mock_context_manager

        # Mock the agent's response
        mock_ai_message = AIMessage(content="This is an image response")
        mock_agent = MagicMock()
        mock_agent.ainvoke = unittest.mock.AsyncMock(return_value={"messages": [mock_ai_message]})
        agent.agent = mock_agent

        with unittest.mock.patch.object(
            agent, "_get_session", new_callable=unittest.mock.AsyncMock, return_value=mock_session
        ):
            # Act
            response = await agent.answer_image_message(chat_id, text, image_url)

            # Assert
            self.assertEqual(response, mock_ai_message)
            mock_session.get.assert_called_once_with(image_url)
            mock_agent.ainvoke.assert_called_once()

            # Check that the chat history was updated with the image message
            self.assertEqual(len(agent.chats[chat_id]["messages"]), 1)
            self.assertIsInstance(agent.chats[chat_id]["messages"][0], HumanMessage)

            # Check the image content in the message
            content = agent.chats[chat_id]["messages"][0].content
            self.assertEqual(len(content), 2)
            self.assertEqual(content[0]["text"], text)
            self.assertIn("data:image/jpeg;base64,", content[1]["image_url"]["url"])

    async def test_answer_image_message_request_failure(self):
        # Arrange
        agent = self.get_basic_llm_agent()
        chat_id = 1
        text = "What's in this image?"
        image_url = "http://example.com/image.jpg"

        # Mock the session to raise aiohttp.ClientError
        mock_session = unittest.mock.AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Failed to fetch image")

        # Act
        with (
            unittest.mock.patch.object(
                agent, "_get_session", new_callable=unittest.mock.AsyncMock, return_value=mock_session
            ),
            self.assertLogs(level="ERROR") as log_context,
        ):
            response = await agent.answer_image_message(chat_id, text, image_url)

        # Assert
        self.assertIsInstance(response, BaseMessage)
        self.assertEqual(response.content, "NO_ANSWER")
        self.assertEqual(response.type, "text")
        mock_session.get.assert_called_once_with(image_url)

        # Verify error was logged
        self.assertTrue(any("Failed to get image" in log for log in log_context.output))

        # Check that no messages were added to chat on failure
        self.assertEqual(len(agent.chats[chat_id]["messages"]), 0)
