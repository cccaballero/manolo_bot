import datetime
import logging
import time

import telegramify_markdown
from telebot import TeleBot, util
from telebot.apihelper import ApiTelegramException
from telebot.types import Message
from telegramify_markdown import customize


def get_telegram_file_url(bot_token: str, file_path: str) -> str:
    """
    Get the URL for a Telegram file.
    :param bot_token: Telegram bot token
    :param file_path: File path from Telegram
    :return: Full URL to access the file
    """
    return f"https://api.telegram.org/file/bot{bot_token}/{file_path}"


def fallback_telegram_call(bot: TeleBot, message: Message, response_content: str) -> bool:
    """
    Call the Telegram API without using Markdown formatting.
    :param bot: Telegram telegram_bot instance
    :param message: Telegram message to reply to
    :param response_content: Response content
    :return: True if the call was successful, False otherwise
    """
    try:
        text_chunks = util.smart_split(response_content, chars_per_string=3000)
        previous_message = None
        for text in text_chunks:
            if previous_message:
                previous_message = bot.reply_to(previous_message, text)
            else:
                previous_message = bot.reply_to(message, text)
    except Exception as e:
        logging.exception(e)
        return False
    return True


def user_is_admin(bot: TeleBot, user_id: int, chat_id: int) -> bool:
    """
    Check if the user is an admin of the chat.
    :param bot: Telegram telegram_bot instance
    :param user_id: User ID
    :param chat_id: Chat ID
    :return: True if the user is an admin, False otherwise
    """
    admins = bot.get_chat_administrators(chat_id)
    return any(admin.user.id == user_id for admin in admins)


def is_bot_reply(bot_username: str, message: Message) -> bool:
    """
    Check if the message is a reply to a telegram_bot message.
    :param message: Telegram message
    :param bot_username: Telegram telegram_bot username
    :return: True if the message is a reply, False otherwise
    """
    return True if message.reply_to_message and message.reply_to_message.from_user.username == bot_username else False


def is_reply(message: Message) -> bool:
    """
    Check if the message is a reply.
    :param message: Telegram message
    :return: True if the message is a reply, False otherwise
    """
    return True if message.reply_to_message else False


def is_image(message: Message) -> bool:
    """
    Check if the message is an image.
    :param message: Telegram message
    :return: True if the message is an image, False otherwise
    """
    return message.content_type == "photo"


def get_message_text(message: Message) -> str:
    """
    Get the text of the message.
    :param message: Telegram message
    :return: Text of the message (caption if message is an image message)
    """
    return message.caption if is_image(message) else message.text


def get_message_from(message: Message) -> str:
    """
    Get the sender of the message.
    :param message: Telegram message
    :return: Sender of the message
    """
    return message.from_user.username


def convert_markdown_to_telegram_format(markdown_text: str) -> str:
    """
    Convert markdown to Telegram format.
    :param markdown_text: Text to convert
    :return: Converted text
    """
    customize.strict_markdown = False
    return telegramify_markdown.markdownify(markdown_text)


def reply_to_telegram_message(bot: TeleBot, message: Message, response_content: str) -> None:
    """
    Reply to a message.
    :param bot: Telegram telegram_bot instance
    :param message: Telegram message to reply to
    :param response_content: Response content
    :return: True if the call was successful, False otherwise
    """
    chat_id = message.chat.id
    use_fallback = False

    if len(response_content) > 4096:
        logging.warning(f"Response for chat {chat_id} is too long. Attempting fallback to plain text.")
        use_fallback = True
    else:
        try:
            markdown_text = convert_markdown_to_telegram_format(response_content)
            bot.reply_to(message, markdown_text, parse_mode="MarkdownV2")
            logging.debug(f"Sent response for chat {chat_id}")
        except ApiTelegramException as e:
            logging.exception(e)
            logging.warning(
                f"Failed to send response for chat {chat_id} using Markdown. Attempting fallback to plain text."
            )
            use_fallback = True
        except Exception as e:
            logging.exception(e)
            use_fallback = True

    if use_fallback and not fallback_telegram_call(bot, message, response_content):
        logging.error(f"Failed to send response for chat {chat_id}")


def clean_standard_message(bot_username: str, message_text: str) -> str:
    """
    Clean a standard message.
    :param bot_username: Telegram telegram_bot username
    :param message_text: Text to clean
    :return: Cleaned text
    """
    replace = f"@{bot_username}: "
    if message_text.startswith(replace):
        message_text = message_text[len(replace) :]
    return message_text


def _get_time_from_wpm(text: str, wpm: float) -> float:
    """
    Get the time it takes to write a text with a given WPM.
    :param text: Text to write
    :param wpm: Words per minute
    :return: Time in seconds
    """
    return (len(text.split()) / wpm) * 60


def send_typing_action(bot, chat_id):
    bot.send_chat_action(chat_id, "typing")


def simulate_typing(
    bot: TeleBot, chat_id: int, text: str, start_time: datetime.datetime, max_typing_time: int = 10, wpm: int = 50
) -> None:
    """
    Simulate typing for a given text.
    :param bot: Telegram bot instance
    :param chat_id: Chat ID
    :param text: Text to simulate typing for
    :param start_time: Start time of the typing simulation
    :param max_typing_time: Maximum typing time in seconds, defaults to 10 seconds
    :param wpm: Words per minute, defaults to 50
    :return: None
    """
    writing_time = _get_time_from_wpm(text, wpm)
    logging.debug(f"Writing time: {writing_time} seconds")
    time_left = writing_time - (datetime.datetime.now() - start_time).total_seconds()
    time_left = min(time_left, max_typing_time)
    logging.debug(f"Time left: {time_left} seconds")

    for second in range(int(time_left)):
        if second % 6 == 0:
            bot.send_chat_action(chat_id, "typing")
        time.sleep(1)
