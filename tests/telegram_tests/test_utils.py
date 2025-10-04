import datetime
import unittest
import unittest.mock

from telegram.async_utils import (
    _get_time_from_wpm,
    clean_standard_message,
    convert_markdown_to_telegram_format,
    fallback_telegram_call,
    get_message_from,
    get_message_text,
    get_telegram_file_url,
    is_bot_reply,
    is_image,
    reply_to_telegram_message,
    simulate_typing,
    user_is_admin,
)


class TestTelegramUtils(unittest.IsolatedAsyncioTestCase):
    def test_get_telegram_file_url__returns_correct_url_format(self):
        # Arrange
        bot_token = "test_token_123"
        file_path = "photos/test_image.jpg"
        expected_url = "https://api.telegram.org/file/bottest_token_123/photos/test_image.jpg"

        # Act
        result = get_telegram_file_url(bot_token, file_path)

        # Assert
        self.assertEqual(result, expected_url)

    def test_get_telegram_file_url__handles_special_characters_in_path(self):
        # Arrange
        bot_token = "test_token_123"
        file_path = "photos/test image with spaces.jpg"
        expected_url = "https://api.telegram.org/file/bottest_token_123/photos/test image with spaces.jpg"

        # Act
        result = get_telegram_file_url(bot_token, file_path)

        # Assert
        self.assertEqual(result, expected_url)

    def test_get_telegram_file_url__handles_empty_file_path(self):
        # Arrange
        bot_token = "test_token_123"
        file_path = ""
        expected_url = "https://api.telegram.org/file/bottest_token_123/"

        # Act
        result = get_telegram_file_url(bot_token, file_path)

        # Assert
        self.assertEqual(result, expected_url)

    async def test_fallback_telegram_call__successful_reply_to_telegram_message(self):
        # Arrange
        mock_bot = unittest.mock.AsyncMock()
        mock_message = unittest.mock.AsyncMock()
        mock_message.reply = unittest.mock.AsyncMock()
        response_content = "Test response"

        # Act
        result = await fallback_telegram_call(mock_bot, mock_message, response_content)

        # Assert
        self.assertTrue(result)
        mock_message.reply.assert_called_once_with(response_content)

    async def test_fallback_telegram_call__return_false_when_exception_raised(self):
        # Arrange
        mock_bot = unittest.mock.AsyncMock()
        mock_message = unittest.mock.AsyncMock()
        mock_message.reply = unittest.mock.AsyncMock()
        response_content = "Test response"
        mock_message.reply.side_effect = Exception("API Error")

        # Act
        result = await fallback_telegram_call(mock_bot, mock_message, response_content)

        # Assert
        self.assertFalse(result)
        mock_message.reply.assert_called_once_with(response_content)

    async def test_user_is_admin__when_user_in_admin_list(self):
        # Arrange
        mock_bot = unittest.mock.AsyncMock()
        user_id = 12345
        chat_id = 67890

        # Create a mock admin object with the same user ID
        mock_admin = unittest.mock.Mock()
        mock_admin.user.id = user_id

        # Configure the mock bot to return a list with our admin
        mock_bot.get_chat_administrators.return_value = [mock_admin]

        # Act
        result = await user_is_admin(mock_bot, user_id, chat_id)

        # Assert
        self.assertTrue(result)
        mock_bot.get_chat_administrators.assert_called_once_with(chat_id)

    async def test_user_is_admin__when_admin_list_empty(self):
        # Arrange
        mock_bot = unittest.mock.AsyncMock()
        user_id = 12345
        chat_id = 67890

        # Configure the mock bot to return an empty list
        mock_bot.get_chat_administrators.return_value = []

        # Act
        result = await user_is_admin(mock_bot, user_id, chat_id)

        # Assert
        self.assertFalse(result)
        mock_bot.get_chat_administrators.assert_called_once_with(chat_id)

    def test_is_bot_reply__message_is_reply_to_bot_with_matching_username(self):
        # Arrange
        bot_username = "test_bot"

        # Mock the reply_to_message with a bot user
        mock_bot_user = unittest.mock.Mock()
        mock_bot_user.username = bot_username
        mock_bot_user.is_bot = True

        mock_reply_message = unittest.mock.Mock()
        mock_reply_message.from_user = mock_bot_user

        # Mock the main message
        message = unittest.mock.Mock()
        message.reply_to_message = mock_reply_message

        # Act
        result = is_bot_reply(bot_username, message)

        # Assert
        self.assertTrue(result)

    def test_is_bot_reply__message_has_no_reply_to_bot(self):
        # Arrange
        bot_username = "test_bot"

        # Mock the main message with no reply
        message = unittest.mock.Mock()
        message.reply_to_message = None

        # Act
        result = is_bot_reply(bot_username, message)

        # Assert
        self.assertFalse(result)

    def test_is_image__returns_true_when_content_type_is_photo(self):
        # Create a mock message with photo attribute
        mock_message = unittest.mock.Mock()
        mock_message.photo = [unittest.mock.Mock()]  # Non-empty photo list

        # Act
        result = is_image(mock_message)

        # Assert
        self.assertTrue(result)

    def test_is_image__handles_none_message(self):
        # Act & Assert
        with self.assertRaises(AttributeError):
            is_image(None)

    def test_get_message_text__returns_text_when_not_image(self):
        # Arrange
        message = unittest.mock.Mock()
        message.text = "Hello, world!"
        message.photo = None  # Not an image

        # Act
        result = get_message_text(message)

        # Assert
        self.assertEqual(result, "Hello, world!")

    def test_get_message_text__handles_none_text_attribute(self):
        # Arrange
        message = unittest.mock.Mock()
        message.text = None
        message.caption = None
        message.content_type = "text"

        # Act
        result = get_message_text(message)

        # Assert
        self.assertIsNone(result)

    def test_get_message_from__returns_username_when_valid(self):
        # Arrange
        mock_user = unittest.mock.Mock()
        mock_user.username = "test_user"

        mock_message = unittest.mock.Mock()
        mock_message.from_user = mock_user

        # Act
        result = get_message_from(mock_message)

        # Assert
        self.assertEqual(result, "test_user")

    def test_get_message_from__handles_none_username(self):
        # Arrange
        mock_message = unittest.mock.Mock()
        mock_message.from_user = None

        # Act
        result = get_message_from(mock_message)

        # Assert
        self.assertIsNone(result)

    async def test_reply_to_telegram_message__successful_reply_with_markdown(self):
        # Arrange
        mock_bot = unittest.mock.AsyncMock()
        mock_message = unittest.mock.AsyncMock()
        mock_message.chat.id = 123456
        mock_message.reply = unittest.mock.AsyncMock()
        response_content = "Hello @username, this is a *markdown* message"

        # Mock the logging
        with unittest.mock.patch("logging.debug") as mock_logging:
            # Act
            await reply_to_telegram_message(mock_bot, mock_message, response_content)

            # Assert
            mock_message.reply.assert_called_once_with(
                "Hello @username, this is a _markdown_ message\n", parse_mode="MarkdownV2"
            )
            mock_logging.assert_called_once_with(f"Sent response for chat {mock_message.chat.id}")

    async def test_reply_to_telegram_message__empty_response_content(self):
        # Arrange
        mock_bot = unittest.mock.AsyncMock()
        mock_message = unittest.mock.AsyncMock()
        mock_message.chat.id = 123456
        mock_message.reply = unittest.mock.AsyncMock()
        response_content = ""

        # Mock logging
        with unittest.mock.patch("logging.debug") as mock_logging:
            # Act
            await reply_to_telegram_message(mock_bot, mock_message, response_content)

            # Assert
            mock_message.reply.assert_called_once_with("", parse_mode="MarkdownV2")
            mock_logging.assert_called_once_with(f"Sent response for chat {mock_message.chat.id}")

    def test_message_with_bot_username_prefix_is_cleaned(self):
        # Arrange
        bot_username = "test_bot"
        message_text = "@test_bot: Hello world"
        expected_result = "Hello world"

        # Act
        result = clean_standard_message(bot_username, message_text)

        # Assert
        self.assertEqual(result, expected_result)

    def test_none_message_raises_exception(self):
        # Arrange
        bot_username = "test_bot"
        message_text = None

        # Act & Assert
        with self.assertRaises(AttributeError):
            clean_standard_message(bot_username, message_text)

    def test_calculate_time_with_average_wpm(self):
        # Arrange
        text = "This is a simple text with eight words"
        wpm = 60.0  # Average reading speed
        expected_time = 8.0  # 8 words at 60 wpm = 8 seconds

        # Act
        actual_time = _get_time_from_wpm(text, wpm)

        # Assert
        self.assertEqual(actual_time, expected_time)

    def test_calculate_time_with_single_word(self):
        # Arrange
        text = "Hello"
        wpm = 120.0
        expected_time = 0.5  # 1 word at 120 wpm = 0.5 seconds

        # Act
        actual_time = _get_time_from_wpm(text, wpm)

        # Assert
        self.assertEqual(actual_time, expected_time)

    def test_calculate_time_with_very_slow_wpm(self):
        # Arrange
        text = "Two words"
        wpm = 0.1  # Very slow typing speed
        expected_time = 1200.0  # 2 words at 0.1 wpm = 1200 seconds (20 minutes)

        # Act
        actual_time = _get_time_from_wpm(text, wpm)

        # Assert
        self.assertEqual(actual_time, expected_time)

    async def test_simulate_typing_with_default_parameters(self):
        # Arrange
        mock_bot = unittest.mock.AsyncMock()
        chat_id = 123456
        text = "Hello, this is a test message"
        start_time = datetime.datetime.now() - datetime.timedelta(seconds=2)

        # Act
        with unittest.mock.patch("asyncio.sleep") as mock_sleep:
            await simulate_typing(mock_bot, chat_id, text, start_time)

        # Assert
        # For a short text with default parameters (wpm=50), typing should be simulated
        # The function should call send_chat_action at least once
        mock_bot.send_chat_action.assert_called_with(chat_id, "typing")
        # Sleep should be called at least once
        mock_sleep.assert_called_with(1)

    async def test_simulate_typing_with_empty_text(self):
        # Arrange
        mock_bot = unittest.mock.AsyncMock()
        chat_id = 123456
        text = ""
        start_time = datetime.datetime.now()

        # Act
        with unittest.mock.patch("asyncio.sleep") as mock_sleep:
            await simulate_typing(mock_bot, chat_id, text, start_time)

        # Assert
        # For empty text, typing time should be minimal
        # The function should not call send_chat_action or sleep
        mock_bot.send_chat_action.assert_not_called()
        mock_sleep.assert_not_called()

    async def test_simulate_typing_capped_at_max_time(self):
        # Arrange
        mock_bot = unittest.mock.AsyncMock()
        chat_id = 123456
        # Create a very long text (100 words)
        text = "word " * 100
        start_time = datetime.datetime.now()
        max_typing_time = 8  # 8 seconds max
        wpm = 10  # Very slow typing speed

        # Act
        with unittest.mock.patch("asyncio.sleep") as mock_sleep:
            await simulate_typing(mock_bot, chat_id, text, start_time, max_typing_time, wpm)

        # Assert
        # For a long text with low WPM, typing time should be capped at max_typing_time
        # The function should call send_chat_action twice (8 seconds / 5 seconds per call = 1.6 calls, rounded to 2)
        self.assertEqual(mock_bot.send_chat_action.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 8)
        mock_bot.send_chat_action.assert_called_with(chat_id, "typing")

    def test_convert_simple_markdown(self):
        # Arrange
        markdown_text = "# Heading\n**Bold text** and *italic text*"

        # Act
        result = convert_markdown_to_telegram_format(markdown_text)

        # Assert
        self.assertIn("*Bold text*", result)
        self.assertIn("_italic text_", result)
        self.assertNotEqual(markdown_text, result)

    def test_markdown_with_special_characters(self):
        # Arrange
        markdown_text = """
## Special Characters
* List item with [link](https://example.com)
* Item with `code` and ~~strikethrough~~
"""

        # Act
        result = convert_markdown_to_telegram_format(markdown_text)

        # Assert
        self.assertNotEqual(markdown_text, result)
        self.assertIn("*✏ Special Characters*", result)
        self.assertIn("[link](https://example.com)", result)
        self.assertIn("`code`", result)
        self.assertIn("~strikethrough~", result)


if __name__ == "__main__":
    unittest.main()
