"""Deterministic tests for ``CompactTracePolicy`` (issue #87).

No LLM: the policy is pure and synchronous. Covers the verbatim threshold,
the ``[ledger, final]`` shape, line ordering/caps, error/empty/missing
signals, multimodal placeholders, the total-ledger cap, byte-identical final
flattening, and pair atomicity via ``expand_tool_block``.
"""

import os
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.history_policy import (
    DEFAULT_MAX_TAIL_CHARS,
    CompactTracePolicy,
    FinalOnlyPolicy,
    FullTracePolicy,
)
from manolo_bot.config import Config
from manolo_bot.storage.messages.base import expand_tool_block


def _ai_tool_calls(*calls):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"} for call_id, name, args in calls],
    )


def _tool_result(call_id, content, name="search", status="success"):
    return ToolMessage(content=content, tool_call_id=call_id, name=name, status=status)


def _filler(size=1000):
    """Non-tool AI text: inflates the tail without appearing in the ledger."""
    return AIMessage(content="y" * size)


def _tail_chars(tail):
    return sum(len(str(message.content or "")) for message in tail)


class TestCompactTracePolicy(unittest.TestCase):
    def test_empty_tail(self):
        self.assertEqual(CompactTracePolicy().select([]), [])

    def test_under_threshold_passthrough_matches_full_trace(self):
        tail = [
            _ai_tool_calls(("call_1", "search", {"q": "weather"})),
            _tool_result("call_1", "sunny"),
            AIMessage(content="It is sunny."),
        ]
        selected = CompactTracePolicy(max_tail_chars=10_000).select(tail)
        self.assertEqual(selected, FullTracePolicy().select(tail))
        self.assertIs(selected[0], tail[0])

    def test_threshold_is_inclusive(self):
        tail = [
            _ai_tool_calls(("call_1", "search", {"q": "weather"})),
            _tool_result("call_1", "sunny"),
            AIMessage(content="It is sunny."),
        ]
        size = _tail_chars(tail)
        self.assertEqual(CompactTracePolicy(max_tail_chars=size).select(tail), tail)

        compacted = CompactTracePolicy(max_tail_chars=size - 1).select(tail)
        self.assertEqual(len(compacted), 2)

    def test_over_threshold_is_ledger_then_final_without_tool_artifacts(self):
        tail = [
            _ai_tool_calls(("call_1", "search", {"q": "weather"})),
            _tool_result("call_1", "sunny"),
            AIMessage(content="It is sunny."),
        ]
        selected = CompactTracePolicy(max_tail_chars=_tail_chars(tail) - 1).select(tail)

        self.assertEqual(len(selected), 2)
        ledger, final = selected
        self.assertIsInstance(ledger, AIMessage)
        self.assertIsInstance(final, AIMessage)
        self.assertFalse(ledger.tool_calls)
        self.assertFalse(final.tool_calls)
        self.assertFalse(any(isinstance(message, ToolMessage) for message in selected))
        self.assertEqual(final.content, "It is sunny.")

    def test_ledger_lines_follow_tool_call_order(self):
        tail = [
            _ai_tool_calls(
                ("call_1", "search", {"q": "weather"}),
                ("call_2", "read_document", {"filename": "a.txt"}),
            ),
            # Results arrive in reverse order; the ledger follows the calls.
            _tool_result("call_2", "hello world", name="read_document"),
            _tool_result("call_1", "sunny"),
            _filler(),
            AIMessage(content="done"),
        ]
        ledger = CompactTracePolicy(max_tail_chars=600).select(tail)[0].content
        lines = ledger.splitlines()

        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(lines[0].startswith('search({"q": "weather"}) -> '), lines[0])
        self.assertIn("sunny", lines[0])
        self.assertTrue(lines[1].startswith('read_document({"filename": "a.txt"}) -> '), lines[1])
        self.assertIn("hello world", lines[1])

    def test_output_cap_uses_ellipsis_and_keeps_original_size(self):
        tail = [
            _ai_tool_calls(("call_1", "echo", {})),
            _tool_result("call_1", "a" * 1000),
            AIMessage(content="done"),
        ]
        ledger = CompactTracePolicy(max_tail_chars=500, per_tool_chars=100).select(tail)[0].content

        self.assertIn("a" * 97 + "...", ledger)
        self.assertIn(", 1000 chars", ledger)
        self.assertNotIn("a" * 200, ledger)

    def test_args_cap_keeps_ledger_bounded(self):
        tail = [
            _ai_tool_calls(("call_1", "echo", {"q": "z" * 500})),
            _tool_result("call_1", "x" * 3000),
            AIMessage(content="done"),
        ]
        ledger = CompactTracePolicy(max_tail_chars=1000, per_tool_chars=50).select(tail)[0].content

        self.assertTrue(ledger.startswith('echo({"q": "zzz'), ledger)
        self.assertIn("...", ledger)
        self.assertNotIn("z" * 200, ledger)
        self.assertLessEqual(len(ledger), 1000)

    def test_error_empty_and_missing_results_are_signals(self):
        tail = [
            _ai_tool_calls(
                ("call_1", "fail", {}),
                ("call_2", "empty", {}),
                ("call_3", "missing", {}),
            ),
            _tool_result("call_1", "boom", name="fail", status="error"),
            _tool_result("call_2", "", name="empty"),
            _filler(),
            AIMessage(content="done"),
        ]
        lines = CompactTracePolicy(max_tail_chars=500).select(tail)[0].content.splitlines()

        self.assertEqual(len(lines), 3, lines)
        self.assertIn("[error] boom, 4 chars", lines[0])
        self.assertIn("(empty), 0 chars", lines[1])
        self.assertIn("(no result)", lines[2])
        self.assertNotIn(", ", lines[2])

    def test_multimodal_output_becomes_placeholder(self):
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 500}},
            {"type": "text", "text": "caption"},
        ]
        tail = [
            _ai_tool_calls(("call_1", "screenshot", {})),
            _tool_result("call_1", content, name="screenshot"),
            AIMessage(content="done"),
        ]
        ledger = CompactTracePolicy(max_tail_chars=200).select(tail)[0].content

        self.assertIn("[image output omitted]", ledger)
        self.assertIn("caption", ledger)
        self.assertNotIn("base64", ledger)

    def test_multiple_results_for_one_call_are_collapsed_to_one_line(self):
        tail = [
            _ai_tool_calls(("call_1", "search", {})),
            _tool_result("call_1", "r1"),
            _tool_result("call_1", "r2"),
            _filler(),
            AIMessage(content="done"),
        ]
        lines = CompactTracePolicy(max_tail_chars=500).select(tail)[0].content.splitlines()

        self.assertEqual(len(lines), 1, lines)
        self.assertIn("r1 r2, 5 chars", lines[0])

    def test_final_is_byte_identical_to_final_only(self):
        tail = [
            _ai_tool_calls(("call_1", "echo", {})),
            _tool_result("call_1", "out"),
            AIMessage(content=[{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]),
        ]
        final = CompactTracePolicy(max_tail_chars=3).select(tail)[-1]
        expected = FinalOnlyPolicy().select(tail)[0]

        self.assertEqual(final.content, expected.content)
        self.assertEqual(final.content, "hello\n\nworld")
        self.assertEqual(final.type, expected.type)

    def test_no_tool_calls_over_threshold_collapses_to_final_only(self):
        tail = [AIMessage(content="x" * 500)]
        selected = CompactTracePolicy(max_tail_chars=100).select(tail)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].content, "x" * 500)
        self.assertIsNot(selected[0], tail[0])

    def test_ledger_total_is_capped_with_omission_marker(self):
        calls = tuple((f"call_{i}", "echo", {"i": i}) for i in range(50))
        tail = [
            _ai_tool_calls(*calls),
            *[_tool_result(f"call_{i}", "r" * 100) for i in range(50)],
            AIMessage(content="done"),
        ]
        ledger = CompactTracePolicy(max_tail_chars=500, per_tool_chars=100).select(tail)[0].content

        self.assertLessEqual(len(ledger), 500)
        self.assertIn("more tool calls omitted)", ledger)
        self.assertLess(len(ledger.splitlines()), 50)

    def test_ledger_messages_are_pair_atomic_singletons(self):
        tail = [
            _ai_tool_calls(("call_1", "search", {"q": "weather"})),
            _tool_result("call_1", "sunny"),
            _filler(),
            AIMessage(content="done"),
        ]
        selected = CompactTracePolicy(max_tail_chars=500).select(tail)

        for index in range(len(selected)):
            with self.subTest(index=index):
                self.assertEqual(expand_tool_block(selected, index), (index, index))

    def test_constructor_validates_bounds(self):
        with self.assertRaises(ValueError):
            CompactTracePolicy(max_tail_chars=-1)
        with self.assertRaises(ValueError):
            CompactTracePolicy(per_tool_chars=3)


class TestCompactTraceAdapterConfig(unittest.TestCase):
    @staticmethod
    def _ensure_required_env():
        os.environ.setdefault("TELEGRAM_BOT_NAME", "Manolo")
        os.environ.setdefault("TELEGRAM_BOT_USERNAME", "ManoloBot")
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1234567890")

    def _env(self, **values):
        saved = {key: os.environ.get(key) for key in values}
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return saved

    @staticmethod
    def _restore(saved):
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_env_parses_compact_trace_and_max_chars(self):
        self._ensure_required_env()
        saved = self._env(AGENT_HISTORY_POLICY="compact-trace", AGENT_HISTORY_MAX_CHARS="1234")
        try:
            config = Config(lazy=True)
            self.assertEqual(config.agent_history_policy, "compact-trace")
            self.assertEqual(config.agent_history_max_chars, 1234)
        finally:
            self._restore(saved)

    def test_bot_config_forwarded_to_policy(self):
        bot_config = BotConfig(
            bot_uuid="b",
            bot_name="n",
            bot_username="u",
            bot_token="t",
            user_id=0,
            history_policy="compact-trace",
            history_max_chars=800,
        )
        policy = CompactTracePolicy(max_tail_chars=bot_config.history_max_chars)

        self.assertEqual(policy.max_tail_chars, 800)

    def test_bot_config_defaults_match_policy_defaults(self):
        bot_config = BotConfig(
            bot_uuid="b",
            bot_name="n",
            bot_username="u",
            bot_token="t",
            user_id=0,
        )
        self.assertEqual(bot_config.history_max_chars, DEFAULT_MAX_TAIL_CHARS)

    def test_env_default_matches_policy_default(self):
        self._ensure_required_env()
        saved = self._env(AGENT_HISTORY_MAX_CHARS=None)
        try:
            self.assertEqual(Config(lazy=True).agent_history_max_chars, DEFAULT_MAX_TAIL_CHARS)
        finally:
            self._restore(saved)


if __name__ == "__main__":
    unittest.main()
