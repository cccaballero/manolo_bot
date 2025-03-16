import unittest
import unittest.mock

import telebot

from telegram.utils import (
    clean_standard_message,
    fallback_telegram_call,
    get_message_from,
    get_message_text,
    is_bot_reply,
    is_image,
    reply_to_telegram_message,
    user_is_admin,
)


class TestTelegramUtils(unittest.TestCase):
    def test_fallback_telegram_call__successful_reply_to_telegram_message(self):
        # Arrange
        mock_bot = unittest.mock.Mock()
        mock_message = unittest.mock.Mock()
        response_content = "Test response"

        # Act
        result = fallback_telegram_call(mock_bot, mock_message, response_content)

        # Assert
        self.assertTrue(result)
        mock_bot.reply_to.assert_called_once_with(mock_message, response_content)

    def test_fallback_telegram_call__return_false_when_exception_raised(self):
        # Arrange
        mock_bot = unittest.mock.Mock()
        mock_message = unittest.mock.Mock()
        response_content = "Test response"
        mock_bot.reply_to.side_effect = Exception("API Error")

        # Act
        result = fallback_telegram_call(mock_bot, mock_message, response_content)

        # Assert
        self.assertFalse(result)
        mock_bot.reply_to.assert_called_once_with(mock_message, response_content)

    def test_user_is_admin__when_user_in_admin_list(self):
        # Arrange
        mock_bot = unittest.mock.Mock()
        user_id = 12345
        chat_id = 67890

        # Create a mock admin object with the same user ID
        mock_admin = unittest.mock.Mock()
        mock_admin.user.id = user_id

        # Configure the mock telegram_bot to return a list with our admin
        mock_bot.get_chat_administrators.return_value = [mock_admin]

        # Act
        result = user_is_admin(mock_bot, user_id, chat_id)

        # Assert
        self.assertTrue(result)
        mock_bot.get_chat_administrators.assert_called_once_with(chat_id)

    def test_user_is_admin__when_admin_list_empty(self):
        # Arrange
        mock_bot = unittest.mock.Mock()
        user_id = 12345
        chat_id = 67890

        # Configure the mock telegram_bot to return an empty list
        mock_bot.get_chat_administrators.return_value = []

        # Act
        result = user_is_admin(mock_bot, user_id, chat_id)

        # Assert
        self.assertFalse(result)
        mock_bot.get_chat_administrators.assert_called_once_with(chat_id)

    def test_is_bot_reply__message_is_reply_to_bot_with_matching_username(self):
        # Arrange
        bot_username = "test_bot"
        message = telebot.types.Message(
            message_id=1,
            from_user=telebot.types.User(id=123, is_bot=False, first_name="User"),
            date=0,
            chat=telebot.types.Chat(id=1, type="private"),
            content_type="text",
            options={},
            json_string="",
        )
        message.reply_to_message = telebot.types.Message(
            message_id=2,
            from_user=telebot.types.User(id=456, is_bot=True, first_name="Bot", username=bot_username),
            date=0,
            chat=telebot.types.Chat(id=1, type="private"),
            content_type="text",
            options={},
            json_string="",
        )

        # Act
        result = is_bot_reply(bot_username, message)

        # Assert
        self.assertTrue(result)

    def test_is_bot_reply__message_has_no_reply_to_bot(self):
        # Arrange
        bot_username = "test_bot"
        message = telebot.types.Message(
            message_id=1,
            from_user=telebot.types.User(id=123, is_bot=False, first_name="User"),
            date=0,
            chat=telebot.types.Chat(id=1, type="private"),
            content_type="text",
            options={},
            json_string="",
        )
        message.reply_to_message = None

        # Act
        result = is_bot_reply(bot_username, message)

        # Assert
        self.assertFalse(result)

    def test_is_image__returns_true_when_content_type_is_photo(self):
        # Create a mock message with content_type 'photo'
        mock_message = type("obj", (object,), {"content_type": "photo"})

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
        message.content_type = "text"

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
        mock_user = telebot.types.User(id=123, is_bot=False, first_name="Test", username="test_user")
        mock_message = telebot.types.Message(
            message_id=1,
            from_user=mock_user,
            date=1234567890,
            chat=telebot.types.Chat(id=1, type="private"),
            content_type="text",
            options={},
            json_string="",
        )

        # Act
        result = get_message_from(mock_message)

        # Assert
        self.assertEqual(result, "test_user")

    def test_get_message_from__handles_none_username(self):
        # Arrange
        mock_user = telebot.types.User(id=123, is_bot=False, first_name="Test", username=None)
        mock_message = telebot.types.Message(
            message_id=1,
            from_user=mock_user,
            date=1234567890,
            chat=telebot.types.Chat(id=1, type="private"),
            content_type="text",
            options={},
            json_string="",
        )

        # Act
        result = get_message_from(mock_message)

        # Assert
        self.assertIsNone(result)

    def test_reply_to_telegram_message__successful_reply_with_markdown(self):
        # Arrange
        mock_bot = unittest.mock.Mock()
        mock_message = unittest.mock.Mock()
        mock_message.chat.id = 123456
        response_content = "Hello @username, this is a *markdown* message"

        # Mock the logging
        with unittest.mock.patch("logging.debug") as mock_logging:
            # Act
            reply_to_telegram_message(mock_bot, mock_message, response_content)

            # Assert
            mock_bot.reply_to.assert_called_once_with(
                mock_message, "Hello @username, this is a *markdown* message", parse_mode="markdown"
            )
            mock_logging.assert_called_once_with(f"Sent response for chat {mock_message.chat.id}")

    def test_reply_to_telegram_message__empty_response_content(self):
        # Arrange
        mock_bot = unittest.mock.Mock()
        mock_message = unittest.mock.Mock()
        mock_message.chat.id = 123456
        response_content = ""

        # Mock logging
        with unittest.mock.patch("logging.debug") as mock_logging:
            # Act
            reply_to_telegram_message(mock_bot, mock_message, response_content)

            # Assert
            mock_bot.reply_to.assert_called_once_with(mock_message, "", parse_mode="markdown")
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


if __name__ == "__main__":
    unittest.main()
