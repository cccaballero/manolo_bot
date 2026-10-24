import asyncio
import base64
import datetime
import logging
import re

import telegramify_markdown
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile, Message


def get_telegram_file_url(bot_token: str, file_path: str) -> str:
    """
    Get the URL for a Telegram file.
    :param bot_token: Telegram bot token
    :param file_path: File path from Telegram
    :return: Full URL to access the file
    """
    return f"https://api.telegram.org/file/bot{bot_token}/{file_path}"


async def fallback_telegram_call(bot: Bot, message: Message, response_content: str) -> bool:
    """
    Call the Telegram API without using Markdown formatting.
    :param bot: Telegram bot instance
    :param message: Telegram message to reply to
    :param response_content: Response content
    :return: True if the call was successful, False otherwise
    """
    try:
        # Split into chunks if message is too long
        max_length = 3000
        text_chunks = [response_content[i : i + max_length] for i in range(0, len(response_content), max_length)]
        previous_message = None
        for text in text_chunks:
            if previous_message:
                previous_message = await previous_message.reply(text)
            else:
                previous_message = await message.reply(text)
    except Exception as e:
        logging.exception(e)
        return False
    return True


async def user_is_admin(bot: Bot, user_id: int, chat_id: int) -> bool:
    """
    Check if the user is an admin of the chat.
    :param bot: Telegram bot instance
    :param user_id: User ID
    :param chat_id: Chat ID
    :return: True if the user is an admin, False otherwise
    """
    try:
        admins = await bot.get_chat_administrators(chat_id)
        return any(admin.user.id == user_id for admin in admins)
    except Exception as e:
        logging.exception(e)
        return False


def is_bot_reply(bot_username: str, message: Message) -> bool:
    """
    Check if the message is a reply to a bot message.
    :param message: Telegram message
    :param bot_username: Telegram bot username
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
    return message.photo is not None


def is_voice(message: Message) -> bool:
    """
    Check if the message is a voice message.
    :param message: Telegram message
    :return: True if the message is a voice message, False otherwise
    """
    return message.voice is not None


def is_document(message: Message) -> bool:
    """
    Check if the message is a document.
    :param message: Telegram message
    :return: True if the message is a document, False otherwise
    """
    return message.document is not None


def get_message_text(message: Message) -> str:
    """
    Get the text of the message.
    :param message: Telegram message
    :return: Text of the message (caption if message is an image or document message)
    """
    if is_image(message) or is_document(message):
        return message.caption
    return message.text


def get_message_from(message: Message) -> str:
    """
    Get the sender of the message.
    :param message: Telegram message
    :return: Sender of the message
    """
    return message.from_user.username if message.from_user else None


def convert_markdown_to_telegram_format(markdown_text: str) -> str:
    """
    Convert markdown to Telegram format.
    :param markdown_text: Text to convert
    :return: Converted text
    """
    telegramify_markdown.customize.strict_markdown = False
    return telegramify_markdown.markdownify(markdown_text)


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks, including one truncated by the token limit."""
    cleaned = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # A response cut off mid-reasoning has no closing tag; drop the tail too
    # rather than leaking the model's internal trace into the chat.
    return re.sub(r"<(think|thinking)>.*\Z", "", cleaned, flags=re.DOTALL | re.IGNORECASE)


def _chunk_text(text: str, size: int = 3000) -> list[str]:
    """Split text into Telegram-safe plain-text chunks."""
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    return chunks or [""]


async def _send_plain_text(bot: Bot, chat_id: int, text: str, reply_to: Message | None = None) -> None:
    """Send plain text in chunks, replying to ``reply_to`` when provided."""
    previous_message = None
    for chunk in _chunk_text(text):
        if previous_message is not None:
            previous_message = await previous_message.reply(chunk)
        elif reply_to is not None:
            previous_message = await reply_to.reply(chunk)
        else:
            await bot.send_message(chat_id=chat_id, text=chunk)


async def send_text_to_chat(bot: Bot, chat_id: int, text: str, reply_to: Message | None = None) -> bool:
    """Send a text response using MarkdownV2 with a plain-text fallback.

    Works for both a Telegram reply (``reply_to`` set) and an unsolicited
    message to a chat (scheduler deliveries), which has no source message.

    :return: True when the text was sent (or there was nothing to send);
        False when both the Markdown and plain-text attempts failed.
    """
    response_content = strip_thinking(text)
    if not response_content.strip():
        logging.debug(f"Nothing to send for chat {chat_id}")
        return True
    if len(response_content) > 4096:
        logging.warning(f"Response for chat {chat_id} is too long. Attempting fallback to plain text.")
    else:
        try:
            markdown_text = convert_markdown_to_telegram_format(response_content)
            if reply_to is not None:
                await reply_to.reply(markdown_text, parse_mode="MarkdownV2")
            else:
                await bot.send_message(chat_id=chat_id, text=markdown_text, parse_mode="MarkdownV2")
            logging.debug(f"Sent response for chat {chat_id}")
            return True
        except TelegramAPIError as e:
            logging.exception(e)
            logging.warning(
                f"Failed to send response for chat {chat_id} using Markdown. Attempting fallback to plain text."
            )
        except Exception as e:
            logging.exception(e)

    try:
        await _send_plain_text(bot, chat_id, response_content, reply_to=reply_to)
    except Exception as e:
        logging.exception(e)
        logging.error(f"Failed to send response for chat {chat_id}")
        return False
    return True


async def send_photo_to_chat(bot: Bot, chat_id: int, base64_image: str, reply_to: Message | None = None) -> bool:
    """Send a base64 image to a chat, optionally as a reply.

    :return: True when the photo reached Telegram, False otherwise.
    """
    try:
        logging.debug(f"Sending image for chat {chat_id}")
        photo_bytes = base64.b64decode(base64_image)
        photo = BufferedInputFile(photo_bytes, filename="image.png")
        kwargs = {"chat_id": chat_id, "photo": photo}
        if reply_to is not None:
            kwargs["reply_to_message_id"] = reply_to.message_id
        await bot.send_photo(**kwargs)
        logging.debug(f"Sent image response for chat {chat_id}")
        return True
    except Exception as e:
        logging.exception(e)
        logging.error(f"Failed to send image response for chat {chat_id}")
        try:
            if reply_to is not None:
                await reply_to.reply("⚠️ Error sending image.")
            else:
                await send_text_to_chat(bot, chat_id, "⚠️ Error sending image.")
        except Exception:
            logging.exception("Failed to report the image delivery error for chat %s", chat_id)
        return False


async def send_final_response(
    bot: Bot, chat_id: int, final_response: dict | None, reply_to: Message | None = None
) -> bool:
    """Render a ``postprocess_response`` result to a chat.

    :return: True when the response was handled (including a no-op or an empty
        text); False when a text/photo delivery was attempted and failed.
    """
    if not final_response:
        return True
    response_type = final_response.get("type")
    if response_type == "image":
        image_data = final_response.get("data")
        if isinstance(image_data, str):
            return await send_photo_to_chat(bot, chat_id, image_data, reply_to=reply_to)
        logging.error("Image data is not a string")
        return False
    if response_type == "text":
        text = final_response.get("data") or final_response.get("text") or ""
        return await send_text_to_chat(bot, chat_id, text, reply_to=reply_to)
    return True


async def reply_to_telegram_message(bot: Bot, message: Message, response_content: str) -> None:
    """Reply to a message (backward-compatible wrapper)."""
    await send_text_to_chat(bot, message.chat.id, response_content, reply_to=message)


async def reply_photo_to_telegram_message(bot: Bot, message: Message, base64_image: str) -> None:
    """Reply to a message with a photo (backward-compatible wrapper)."""
    await send_photo_to_chat(bot, message.chat.id, base64_image, reply_to=message)


def clean_standard_message(bot_username: str, message_text: str) -> str:
    """
    Clean a standard message.
    :param bot_username: Telegram bot username
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


async def send_typing_action(bot: Bot, chat_id: int):
    """Send typing action to chat."""
    await bot.send_chat_action(chat_id, "typing")


async def simulate_typing(
    bot: Bot, chat_id: int, text: str, start_time: datetime.datetime, max_typing_time: int = 10, wpm: int = 50
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
            await bot.send_chat_action(chat_id, "typing")
        await asyncio.sleep(1)
