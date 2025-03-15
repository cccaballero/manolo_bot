import logging
import re

import telebot


def fallback_telegram_call(bot, message, response_content):
    """
    Call the Telegram API without using Markdown formatting.
    :param bot: Telegram bot instance
    :param message: Telegram message to reply to
    :param response_content: Response content
    :return: True if the call was successful, False otherwise
    """
    try:
        bot.reply_to(message, response_content)
    except Exception as e:
        logging.exception(e)
        return False
    return True


def user_is_admin(bot, user_id, chat_id):
    """
    Check if the user is an admin of the chat.
    :param bot: Telegram bot instance
    :param user_id: User ID
    :param chat_id: Chat ID
    :return: True if the user is an admin, False otherwise
    """
    admins = bot.get_chat_administrators(chat_id)
    return any(admin.user.id == user_id for admin in admins)


def is_bot_reply(bot_username, message):
    """
    Check if the message is a reply to a bot message.
    :param message: Telegram message
    :param bot_username: Telegram bot username
    :return: True if the message is a reply, False otherwise
    """
    return True if message.reply_to_message and message.reply_to_message.from_user.username == bot_username else False


def is_reply(message):
    """
    Check if the message is a reply.
    :param message: Telegram message
    :return: True if the message is a reply, False otherwise
    """
    return True if message.reply_to_message else False


def is_image(message):
    """
    Check if the message is an image.
    :param message: Telegram message
    :return: True if the message is an image, False otherwise
    """
    return message.content_type == 'photo'


def get_message_text(message):
    """
    Get the text of the message.
    :param message: Telegram message
    :return: Text of the message (caption if message is an image message)
    """
    return message.caption if is_image(message) else message.text


def get_message_from(message):
    """
    Get the sender of the message.
    :param message: Telegram message
    :return: Sender of the message
    """
    return message.from_user.username


def reply_to_telegram_message(bot, message, response_content):
    """
    Reply to a message.
    :param bot: Telegram bot instance
    :param message: Telegram message to reply to
    :param response_content: Response content
    :return: True if the call was successful, False otherwise
    """
    chat_id = message.chat.id
    try:
        usernames = re.findall(r"(?<!\S)@\w+", response_content)
        for username in usernames:
            response_content = response_content.replace(username, telebot.formatting.escape_markdown(username))
        bot.reply_to(message, response_content, parse_mode='markdown')
        logging.debug(f"Sent response for chat {chat_id}")
    except Exception as e:
        logging.exception(e)
        if not fallback_telegram_call(bot, message, response_content):
            logging.error(f"Failed to send response for chat {chat_id}")


def clean_standard_message(bot_username, message_text):
    """
    Clean a standard message.
    :param bot_username: Telegram bot username
    :param message_text: Text to clean
    :return: Cleaned text
    """
    replace = f'@{bot_username}: '
    if message_text.startswith(replace):
        message_text = message_text[len(replace):]
    return message_text