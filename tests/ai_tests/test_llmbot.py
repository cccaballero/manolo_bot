import unittest
import unittest.mock

import aiohttp
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai.llmbot import LLMBot
from config import Config


class TestLlmBot(unittest.IsolatedAsyncioTestCase):
    def get_basic_llm_bot(self):
        mock_config = unittest.mock.MagicMock(spec=Config)
        mock_config.ollama_model = "test_model"
        mock_config.google_api_key = None
        mock_config.openai_api_key = None
        mock_config.openai_api_base_url = None
        mock_config.context_max_tokens = 4000
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        llm_bot = LLMBot(mock_config, system_instructions)  # Initialize the bot
        llm_bot.chats = {1: {"messages": []}}
        return llm_bot

    @unittest.mock.patch.object(LLMBot, "_get_chat_ollama")
    @unittest.mock.patch.object(LLMBot, "_get_chat_google_generativeai")
    @unittest.mock.patch.object(LLMBot, "_get_chat_openai")
    def test_llm_bot__init_with_ollama_ai(self, mock_openai, mock_google, mock_ollama):
        # Arrange
        mock_config = unittest.mock.MagicMock(spec=Config)
        mock_config.ollama_model = "ollama_model"
        mock_config.google_api_key = None
        mock_config.google_api_model = None
        mock_config.openai_api_key = None
        mock_config.openai_api_base_url = None
        mock_config.use_tools = False  # Disable tools for this test

        # Create a mock LLM with bind_tools method
        mock_llm = unittest.mock.MagicMock()
        mock_llm.bind_tools.return_value = mock_llm  # Return self for chaining
        mock_ollama.return_value = mock_llm

        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        bot = LLMBot(mock_config, system_instructions)

        # Assert
        mock_ollama.assert_called_once()
        mock_google.assert_not_called()
        mock_openai.assert_not_called()
        self.assertEqual(bot.system_instructions, system_instructions)
        self.assertEqual(bot.config, mock_config)
        self.assertEqual(bot.llm, mock_llm)

    @unittest.mock.patch.object(LLMBot, "_get_chat_ollama")
    @unittest.mock.patch.object(LLMBot, "_get_chat_google_generativeai")
    @unittest.mock.patch.object(LLMBot, "_get_chat_openai")
    def test_llm_bot__init_with_google_ai(self, mock_openai, mock_google, mock_ollama):
        # Arrange
        mock_config = unittest.mock.MagicMock(spec=Config)
        mock_config.ollama_model = None
        mock_config.google_api_key = "fake_google_key"
        mock_config.google_api_model = "gemini-2.0-flash"
        mock_config.openai_api_key = None
        mock_config.openai_api_base_url = None
        mock_config.use_tools = False  # Disable tools for this test

        # Create a mock LLM with bind_tools method
        mock_llm = unittest.mock.MagicMock()
        mock_llm.bind_tools.return_value = mock_llm  # Return self for chaining
        mock_google.return_value = mock_llm

        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        bot = LLMBot(mock_config, system_instructions)

        # Assert
        mock_google.assert_called_once()
        mock_ollama.assert_not_called()
        mock_openai.assert_not_called()
        self.assertEqual(bot.system_instructions, system_instructions)
        self.assertEqual(bot.config, mock_config)
        self.assertEqual(bot.llm, mock_llm)

    @unittest.mock.patch.object(LLMBot, "_get_chat_ollama")
    @unittest.mock.patch.object(LLMBot, "_get_chat_google_generativeai")
    @unittest.mock.patch.object(LLMBot, "_get_chat_openai")
    def test_llm_bot__init_with_openai_ai(self, mock_openai, mock_google, mock_ollama):
        # Arrange
        mock_config = unittest.mock.MagicMock(spec=Config)
        mock_config.ollama_model = None
        mock_config.google_api_key = None
        mock_config.google_api_model = None
        mock_config.openai_api_key = "fake_openai_key"
        mock_config.openai_api_base_url = None
        mock_config.use_tools = False  # Disable tools for this test

        # Create a mock LLM with bind_tools method
        mock_llm = unittest.mock.MagicMock()
        mock_llm.bind_tools.return_value = mock_llm  # Return self for chaining
        mock_openai.return_value = mock_llm

        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        bot = LLMBot(mock_config, system_instructions)

        # Assert
        mock_openai.assert_called_once()
        mock_ollama.assert_not_called()
        mock_google.assert_not_called()
        self.assertEqual(bot.system_instructions, system_instructions)
        self.assertEqual(bot.config, mock_config)
        self.assertEqual(bot.llm, mock_llm)

    def test_llm_bot__init_raises_exception_with_no_llm_backend(self):
        # Arrange
        mock_config = unittest.mock.MagicMock(spec=Config)
        mock_config.ollama_model = None
        mock_config.google_api_key = None
        mock_config.openai_api_key = None
        mock_config.openai_api_base_url = None

        system_instructions = [SystemMessage(content="Hello")]

        # Act & Assert
        with self.assertRaises(Exception) as context:
            LLMBot(mock_config, system_instructions)

        self.assertEqual(str(context.exception), "No LLM backend data found")

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

    def test_call_sdapi__successful_api_call(self):
        # Arrange
        config_mock = unittest.mock.MagicMock()
        config_mock.sdapi_url = "http://test-sd-api.com"
        config_mock.sdapi_params = {
            "steps": 1,
            "cfg_scale": 1,
            "width": 512,
            "height": 512,
            "timestep_spacing": "trailing",
        }
        config_mock.sdapi_negative_prompt = None

        llm_bot = self.get_basic_llm_bot()
        llm_bot.config = config_mock

        prompt = "a beautiful landscape"
        expected_response = {"images": ["base64_image_data"]}

        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response

        # Mock the session's post method
        with unittest.mock.patch.object(llm_bot._session, "post", return_value=mock_response) as mock_post:
            # Act
            result = llm_bot.call_sdapi(prompt)

            # Assert
            mock_post.assert_called_once()
            call_args, call_kwargs = mock_post.call_args
            self.assertEqual(call_args[0], "http://test-sd-api.com/sdapi/v1/txt2img")
            self.assertEqual(call_kwargs["json"], {**config_mock.sdapi_params, "prompt": prompt})
            self.assertEqual(result, expected_response)

    def test_call_sdapi__non_200_response(self):
        # Arrange
        config_mock = unittest.mock.MagicMock()
        config_mock.sdapi_url = "http://test-sd-api.com"
        config_mock.sdapi_params = {
            "steps": 1,
            "cfg_scale": 1,
            "width": 512,
            "height": 512,
            "timestep_spacing": "trailing",
        }

        llm_bot = self.get_basic_llm_bot()
        llm_bot.config = config_mock

        prompt = "a beautiful landscape"

        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        # Mock the session's post method
        with unittest.mock.patch.object(llm_bot._session, "post", return_value=mock_response) as mock_post:
            # Act
            result = llm_bot.call_sdapi(prompt)

            # Assert
            mock_post.assert_called_once()
            call_args, call_kwargs = mock_post.call_args
            self.assertEqual(call_args[0], "http://test-sd-api.com/sdapi/v1/txt2img")

            # Check the json payload without negative_prompt since it's None
            expected_json = {**config_mock.sdapi_params, "prompt": prompt}
            actual_json = {k: v for k, v in call_kwargs["json"].items() if k != "negative_prompt"}
            self.assertEqual(actual_json, expected_json)
            self.assertIsNone(result)

    async def test_answer_image_message__successful_image_message_processing(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        text = "What's in this image?"
        image_url = "https://example.com/image.jpg"
        llm_bot.count_tokens = unittest.mock.Mock()
        llm_bot.count_tokens.return_value = 100

        # Mock aiohttp response
        mock_response = unittest.mock.AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = b"fake_image_data"

        # Mock LLM response
        expected_response = AIMessage(content="This is an image of a cat")
        llm_bot.llm = unittest.mock.Mock()
        llm_bot.llm.invoke.return_value = expected_response

        # Mock the session's get method
        mock_session = unittest.mock.AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with unittest.mock.patch.object(llm_bot, "_get_session", return_value=mock_session):
            # Act
            response = await llm_bot.answer_image_message(1, text, image_url)

            # Assert
            mock_session.get.assert_called_once_with(image_url)
            self.assertEqual(response, expected_response)
            self.assertEqual(len(llm_bot.chats[1]["messages"]), 1)
            llm_bot.llm.invoke.assert_called_once_with(llm_bot.chats[1]["messages"])

    async def test_answer_image_message__handles_request_exception(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        text = "What's in this image?"
        image_url = "https://invalid-url.com/image.jpg"

        # Mock the session's get method to raise aiohttp.ClientError
        mock_session = unittest.mock.AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Failed to get image")

        with (
            unittest.mock.patch.object(llm_bot, "_get_session", return_value=mock_session),
            unittest.mock.patch("logging.error") as mock_logger,
            unittest.mock.patch("logging.exception") as mock_exception_logger,
        ):
            # Act
            response = await llm_bot.answer_image_message(1, text, image_url)

            # Assert
            mock_logger.assert_called_once_with(f"Failed to get image: {image_url}")
            mock_exception_logger.assert_called_once()
            self.assertEqual(response.content, "NO_ANSWER")
            self.assertEqual(response.type, "text")

    async def test_generate_image__success(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        mock_response = {"images": ["base64_encoded_image_data"]}
        with unittest.mock.patch.object(llm_bot, "call_sdapi", return_value=mock_response) as mock_call_sdapi:
            # Act
            result = await llm_bot.generate_image("a beautiful landscape")

            # Assert
            self.assertEqual(result, "base64_encoded_image_data")
            mock_call_sdapi.assert_called_once_with("a beautiful landscape")

    async def test_generate_image__returns_none(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()
        with unittest.mock.patch.object(llm_bot, "call_sdapi", return_value=None) as mock_call_sdapi:
            # Act
            result = await llm_bot.generate_image("a beautiful landscape")

            # Assert
            self.assertIsNone(result)
            mock_call_sdapi.assert_called_once_with("a beautiful landscape")

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

        llm_bot = self.get_basic_llm_bot()
        llm_bot.config = mock_config

        expected_response = AIMessage(content="¡Contexto de chat borrado con éxito!")
        llm_bot.llm = unittest.mock.Mock()
        llm_bot.llm.invoke.return_value = expected_response

        # Act
        result = await llm_bot.generate_feedback_message("some prompt")

        # Assert
        self.assertEqual(result, "¡Contexto de chat borrado con éxito!")
        llm_bot.llm.invoke.assert_called_once()

    async def test_generate_feedback_message__truncates_long_messages(self):
        # Arrange
        mock_config = unittest.mock.MagicMock(spec=Config)
        mock_config.preferred_language = "English"

        llm_bot = self.get_basic_llm_bot()
        llm_bot.config = mock_config

        # Create a response that's longer than 200 characters
        long_message = "This is a very long message that exceeds the 200 character limit. " * 5
        expected_response = AIMessage(content=long_message)
        llm_bot.llm = unittest.mock.Mock()
        llm_bot.llm.invoke.return_value = expected_response

        # Act
        result = await llm_bot.generate_feedback_message("success")

        # Assert
        self.assertEqual(len(result), 200)  # 197 chars + 3 for "..."
        self.assertTrue(result.endswith("..."))
        self.assertEqual(result, long_message[:197] + "...")

    async def test_answer_webcontent__no_url_found_returns_none(self):
        # Arrange
        llm_bot = self.get_basic_llm_bot()

        message_text = "Summarize this webpage"
        response_content = "There is no URL in this content"

        # Mock _extract_url to return None (no URL found)
        unittest.mock.patch.object(llm_bot, "_extract_url", return_value=None)
        with (
            unittest.mock.patch.object(llm_bot, "_extract_url", return_value=None) as _extract_url_mock,
            unittest.mock.patch.object(llm_bot, "_remove_urls") as _remove_urls_mock,
        ):
            # Act
            result = await llm_bot.answer_webcontent(message_text, response_content, 1)

            # Assert
            self.assertIsNone(result)
            _extract_url_mock.assert_called_once_with(response_content)
            # Verify that _remove_urls was not called since no URL was found
            _remove_urls_mock.assert_not_called()

    # TODO: find a way to test process_message_buffer logic, probably the function needs a refactor to make it more
    #  testable


if __name__ == "__main__":
    unittest.main()
