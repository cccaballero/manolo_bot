"""Tests for silent scheduled-agent responses in ``postprocess_response``."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, SystemMessage

from manolo_bot.ai.llmbot import LLMBot

PLACEHOLDERS = {"😐", "😶", "😳", "😕", "😑"}


class TestPostprocessSilentResponse(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        config = MagicMock()
        config.use_tools = False
        self.storage = MagicMock()
        self.bot = LLMBot(MagicMock(), config, [SystemMessage(content="sys")], self.storage)

    async def test_no_answer_posts_a_placeholder_by_default(self) -> None:
        result = await self.bot.postprocess_response(AIMessage(content="NO_ANSWER"), "hi", 1)

        self.assertEqual(result["type"], "text")
        self.assertIn(result["data"], PLACEHOLDERS)
        self.storage.add_message.assert_called_once()

    async def test_silent_on_no_answer_returns_none_without_persisting(self) -> None:
        result = await self.bot.postprocess_response(AIMessage(content="NO_ANSWER"), "hi", 1, silent_on_no_answer=True)

        self.assertIsNone(result)
        self.storage.add_message.assert_not_called()

    async def test_silent_flag_does_not_suppress_real_answers(self) -> None:
        result = await self.bot.postprocess_response(
            AIMessage(content="There is a match tomorrow."), "hi", 1, silent_on_no_answer=True
        )

        self.assertEqual(result, {"type": "text", "data": "There is a match tomorrow."})
        self.storage.add_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
