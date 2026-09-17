"""Atomic tool-block truncation under overflow (issue #85 lane).

Proves that ``LLMBot`` truncation (both the drop-oldest path and the
summarization fold path) never splits an atomic AI->Tool* block: tool
blocks drop all-or-nothing. Also pins the ``FinalOnly`` default
(policy + adapter plumbing) as unchanged.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.history_policy import FinalOnlyPolicy, FullTracePolicy
from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.ai.llmbot import LLMBot
from manolo_bot.config import Config
from manolo_bot.storage.messages.memory_storage import MemoryMessagesStorage


def _ai_tool_call(call_id="call_1", name="search"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"q": "hi"}, "id": call_id, "type": "tool_call"}],
    )


def _tool_result(call_id="call_1", name="search", content="r1"):
    return ToolMessage(content=content, tool_call_id=call_id, name=name)


def _assert_no_orphans(testcase, messages):
    """Every ToolMessage has a paired AI tool call; every AI tool call has results."""
    for i, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            testcase.assertTrue(
                any(
                    isinstance(m, AIMessage) and any(c.get("id") == message.tool_call_id for c in (m.tool_calls or []))
                    for m in messages[:i]
                ),
                f"orphan ToolMessage at index {i} (tool_call_id={message.tool_call_id})",
            )
    for i, message in enumerate(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            call_ids = {c.get("id") for c in message.tool_calls}
            found = {m.tool_call_id for m in messages[i + 1 :] if isinstance(m, ToolMessage)}
            testcase.assertTrue(
                call_ids & found,
                f"AI tool call at index {i} lost all its ToolMessages",
            )


def _make_bot(messages, *, context_max_tokens, summarization, llm=None):
    storage = MemoryMessagesStorage(bot_uuid="test-bot", chat_id=424242)
    for msg in messages:
        storage.add_message(msg)
    if llm is None:
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="SUMMARIZED CONTENT"))
        llm.bind_tools.return_value = llm
    llm.get_num_tokens = MagicMock(return_value=10)
    llm.get_num_tokens_from_messages = MagicMock(side_effect=lambda msgs: 100 * len(msgs))
    bot_config = BotConfig(
        bot_uuid="test-bot",
        bot_name="TestBot",
        bot_username="test_bot",
        bot_token="123456:ABC",
        user_id=0,
        context_max_tokens=context_max_tokens,
        context_summarization=summarization,
        summary_max_tokens=512,
        summary_keep_messages=6,
    )
    bot = LLMBot(llm, bot_config, [SystemMessage(content="You are a helpful assistant")], storage)
    return bot, storage


def _run(coro):
    return asyncio.run(coro)


class TestDropOldestAtomic(unittest.TestCase):
    def test_tool_block_drops_all_or_nothing(self):
        # FullTrace-style persisted trace: a tool block sits at the front and
        # the drop candidate lands inside it. 5 x 100 tokens > 300 limit.
        messages = [
            HumanMessage(content="hi"),
            _ai_tool_call("call_1"),
            _tool_result("call_1", content="r1"),
            _tool_result("call_1", content="r2"),
            HumanMessage(content="thanks"),
        ]
        bot, storage = _make_bot(messages, context_max_tokens=300, summarization=False)

        _run(bot.truncate_chat_context())

        remaining = storage.messages
        # The block is gone entirely — never partially.
        self.assertFalse(any(isinstance(m, ToolMessage) for m in remaining))
        self.assertFalse(any(isinstance(m, AIMessage) and m.tool_calls for m in remaining))
        _assert_no_orphans(self, remaining)

    def test_drop_preserves_summary_and_stays_atomic(self):
        storage = MemoryMessagesStorage(bot_uuid="test-bot", chat_id=424242)
        storage.set_summary("OLD SUMMARY")
        for msg in [
            HumanMessage(content="hi"),
            _ai_tool_call("call_1"),
            _tool_result("call_1", content="r1"),
            HumanMessage(content="thanks"),
        ]:
            storage.add_message(msg)
        llm = MagicMock()
        llm.get_num_tokens = MagicMock(return_value=10)
        llm.get_num_tokens_from_messages = MagicMock(side_effect=lambda msgs: 100 * len(msgs))
        bot_config = BotConfig(
            bot_uuid="test-bot",
            bot_name="TestBot",
            bot_username="test_bot",
            bot_token="123456:ABC",
            user_id=0,
            context_max_tokens=300,
            context_summarization=False,
        )
        bot = LLMBot(llm, bot_config, [SystemMessage(content="sys")], storage)

        _run(bot.truncate_chat_context())

        # Summary (index 0) is never dropped; the tool block is all-or-nothing.
        self.assertEqual(storage.get_summary(), "OLD SUMMARY")
        _assert_no_orphans(self, storage.messages)


class TestFoldPathAtomic(unittest.TestCase):
    def test_fold_expands_to_block_edges_single_call(self):
        # 12 messages; limit 850 forces keep_n=3 (300+512<=850<400+512), so the
        # naive fold_end=9 would split the AI(8)->Tool(9) block. The fix folds
        # the whole block: exactly one summarization call, no orphans.
        messages = [HumanMessage(content=f"message-{i}") for i in range(8)]
        messages += [_ai_tool_call("call_9"), _tool_result("call_9", content="r9")]
        messages += [HumanMessage(content="message-10"), HumanMessage(content="message-11")]
        bot, storage = _make_bot(messages, context_max_tokens=850, summarization=True)

        _run(bot.truncate_chat_context())

        self.assertEqual(bot.llm.ainvoke.await_count, 1)
        remaining = storage.messages
        # Whole block folded into the summary; only summary + kept tail remain.
        self.assertEqual(storage.get_summary(), "SUMMARIZED CONTENT")
        self.assertFalse(any(isinstance(m, ToolMessage) for m in remaining))
        self.assertFalse(any(isinstance(m, AIMessage) and m.tool_calls for m in remaining))
        _assert_no_orphans(self, remaining)
        self.assertEqual([m.content for m in remaining[1:]], ["message-10", "message-11"])


class TestFinalOnlyDefaultUnchanged(unittest.TestCase):
    def _agent(self, **kwargs):
        storage = MemoryMessagesStorage(bot_uuid="test-bot", chat_id=424242)
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        bot_config = BotConfig(
            bot_uuid="test-bot",
            bot_name="TestBot",
            bot_username="test_bot",
            bot_token="123456:ABC",
            user_id=0,
        )
        return LLMAgent(llm, bot_config, [SystemMessage(content="sys")], storage, **kwargs)

    def test_agent_defaults_to_final_only(self):
        agent = self._agent()
        self.assertIsInstance(agent.history_policy, FinalOnlyPolicy)

    def test_full_trace_select_is_verbatim(self):
        tail = [_ai_tool_call("call_1"), _tool_result("call_1"), AIMessage(content="done")]
        self.assertEqual(FullTracePolicy().select(tail), tail)

    def test_bot_config_defaults_to_final_only(self):
        config = BotConfig(
            bot_uuid="b",
            bot_name="n",
            bot_username="u",
            bot_token="t",
            user_id=0,
        )
        self.assertEqual(config.history_policy, "final-only")

    def test_env_config_defaults_to_final_only(self):
        os.environ.setdefault("TELEGRAM_BOT_NAME", "Manolo")
        os.environ.setdefault("TELEGRAM_BOT_USERNAME", "ManoloBot")
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1234567890")
        os.environ.pop("AGENT_HISTORY_POLICY", None)
        self.assertEqual(Config(lazy=True).agent_history_policy, "final-only")


if __name__ == "__main__":
    unittest.main()
