"""Tests for chat-addressed Telegram delivery used by the scheduler."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from manolo_bot.telegram.utils import (
    send_final_response,
    send_photo_to_chat,
    send_text_to_chat,
    strip_thinking,
)


class TestStripThinking(unittest.TestCase):
    def test_removes_think_blocks(self):
        self.assertEqual(strip_thinking("<thinking>secret</thinking>hello"), "hello")

    def test_removes_an_unterminated_reasoning_block(self):
        self.assertEqual(strip_thinking("<thinking>still reasoning"), "")

    def test_keeps_text_before_a_reasoning_block(self):
        self.assertEqual(strip_thinking("answer<thinking>secret</thinking>"), "answer")

    def test_keeps_plain_text(self):
        self.assertEqual(strip_thinking("hello"), "hello")


class TestSendTextToChat(unittest.IsolatedAsyncioTestCase):
    async def test_sends_markdown_to_chat_without_a_source_message(self):
        bot = AsyncMock()

        result = await send_text_to_chat(bot, 42, "**bold**")

        self.assertTrue(result)
        bot.send_message.assert_awaited_once()
        kwargs = bot.send_message.await_args.kwargs
        self.assertEqual(kwargs["chat_id"], 42)
        self.assertEqual(kwargs["parse_mode"], "MarkdownV2")
        self.assertIn("bold", kwargs["text"])

    async def test_replies_when_a_source_message_is_given(self):
        bot = AsyncMock()
        message = AsyncMock()

        await send_text_to_chat(bot, 42, "hello", reply_to=message)

        message.reply.assert_awaited_once()
        bot.send_message.assert_not_awaited()

    async def test_strips_thinking_before_sending(self):
        bot = AsyncMock()

        await send_text_to_chat(bot, 42, "<thinking>x</thinking>visible")

        self.assertIn("visible", bot.send_message.await_args.kwargs["text"])
        self.assertNotIn("thinking", bot.send_message.await_args.kwargs["text"])

    async def test_falls_back_to_plain_text_on_send_error(self):
        bot = AsyncMock()
        bot.send_message.side_effect = [Exception("markdown failed"), None]

        result = await send_text_to_chat(bot, 42, "hello")

        self.assertTrue(result)
        self.assertEqual(bot.send_message.await_count, 2)
        self.assertEqual(bot.send_message.await_args.kwargs["text"], "hello")

    async def test_returns_false_when_both_attempts_fail(self) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = Exception("telegram down")

        self.assertFalse(await send_text_to_chat(bot, 42, "hello"))

    async def test_skips_blank_text(self) -> None:
        bot = AsyncMock()

        result = await send_text_to_chat(bot, 42, "<thinking>only thoughts</thinking>")

        self.assertTrue(result)
        bot.send_message.assert_not_awaited()

    async def test_chunks_long_messages(self):
        bot = AsyncMock()

        await send_text_to_chat(bot, 42, "x" * 5000)

        self.assertEqual(bot.send_message.await_count, 2)
        first = bot.send_message.await_args_list[0].kwargs
        second = bot.send_message.await_args_list[1].kwargs
        self.assertEqual(len(first["text"]), 3000)
        self.assertEqual(len(second["text"]), 2000)
        self.assertNotIn("parse_mode", first)


class TestSendPhotoToChat(unittest.IsolatedAsyncioTestCase):
    async def test_sends_photo_to_chat(self):
        bot = AsyncMock()

        result = await send_photo_to_chat(bot, 42, "dGVzdA==")

        self.assertTrue(result)
        bot.send_photo.assert_awaited_once()
        kwargs = bot.send_photo.await_args.kwargs
        self.assertEqual(kwargs["chat_id"], 42)
        self.assertNotIn("reply_to_message_id", kwargs)

    async def test_replies_with_photo(self):
        bot = AsyncMock()
        message = AsyncMock()
        message.message_id = 7

        await send_photo_to_chat(bot, 42, "dGVzdA==", reply_to=message)

        self.assertEqual(bot.send_photo.await_args.kwargs["reply_to_message_id"], 7)

    async def test_failure_reports_error_in_chat(self):
        bot = AsyncMock()
        bot.send_photo.side_effect = Exception("boom")

        result = await send_photo_to_chat(bot, 42, "dGVzdA==")

        self.assertFalse(result)
        self.assertIn("Error sending image", str(bot.send_message.await_args.kwargs["text"]))


class TestSendFinalResponse(unittest.IsolatedAsyncioTestCase):
    async def test_text_response(self):
        bot = AsyncMock()

        result = await send_final_response(bot, 42, {"type": "text", "data": "hello"})

        self.assertTrue(result)
        self.assertIn("hello", bot.send_message.await_args.kwargs["text"])

    async def test_image_response(self):
        bot = AsyncMock()

        result = await send_final_response(bot, 42, {"type": "image", "data": "dGVzdA=="})

        self.assertTrue(result)
        bot.send_photo.assert_awaited_once()

    async def test_none_and_unknown_are_ignored(self):
        bot = AsyncMock()

        self.assertTrue(await send_final_response(bot, 42, None))
        self.assertTrue(await send_final_response(bot, 42, {"type": "other"}))

        bot.send_message.assert_not_awaited()
        bot.send_photo.assert_not_awaited()

    async def test_reports_delivery_failure(self):
        bot = AsyncMock()
        bot.send_message.side_effect = Exception("telegram down")

        self.assertFalse(await send_final_response(bot, 42, {"type": "text", "data": "hello"}))


if __name__ == "__main__":
    unittest.main()
