import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from manolo_bot.storage.messages.base import (
    convert_json_to_message,
    expand_tool_block,
    find_safe_drop_index,
)
from manolo_bot.storage.messages.memory_storage import MemoryMessagesStorage


def _ai_tool_call(call_id="call_123", name="search"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"q": "hi"}, "id": call_id, "type": "tool_call"}],
    )


class TestHistoryPolicySerialization(unittest.TestCase):
    def test_tool_message_round_trip_preserves_type_and_tool_call_id(self):
        original = ToolMessage(content="result", tool_call_id="call_123", name="search")
        restored = convert_json_to_message(original.model_dump_json())
        self.assertIsInstance(restored, ToolMessage)
        self.assertEqual(restored.type, "tool")
        self.assertEqual(restored.tool_call_id, "call_123")
        self.assertEqual(restored.content, "result")

    def test_ai_tool_call_message_round_trip(self):
        original = _ai_tool_call()
        restored = convert_json_to_message(original.model_dump_json())
        self.assertIsInstance(restored, AIMessage)
        self.assertEqual([c["id"] for c in restored.tool_calls], ["call_123"])


class TestToolBlockAtomicity(unittest.TestCase):
    def _history(self):
        return [
            HumanMessage(content="hi"),
            _ai_tool_call("call_1"),
            ToolMessage(content="r1", tool_call_id="call_1", name="search"),
            ToolMessage(content="r2", tool_call_id="call_1", name="search"),
            HumanMessage(content="thanks"),
        ]

    def test_expand_inside_block_covers_whole_block(self):
        messages = self._history()
        for idx in (1, 2, 3):
            with self.subTest(idx=idx):
                self.assertEqual(expand_tool_block(messages, idx), (1, 3))

    def test_expand_outside_block_is_identity(self):
        messages = self._history()
        for idx in (0, 4):
            with self.subTest(idx=idx):
                self.assertEqual(expand_tool_block(messages, idx), (idx, idx))

    def test_find_safe_drop_index_redirects_into_block_start(self):
        messages = self._history()
        self.assertEqual(find_safe_drop_index(messages, 2), 1)
        self.assertEqual(find_safe_drop_index(messages, 3), 1)
        self.assertEqual(find_safe_drop_index(messages, 1), 1)
        self.assertEqual(find_safe_drop_index(messages, 4), 4)
        self.assertEqual(find_safe_drop_index(messages, 0), 0)

    def test_atomic_drop_pattern_never_splits_pair(self):
        # Simulate the documented AI-lane drop pattern: expand then delete
        # highest-index-first; the pair must disappear together.
        storage = MemoryMessagesStorage(bot_uuid="test-bot", chat_id=1)
        for message in self._history():
            storage.add_message(message)
        messages = storage.messages
        safe = find_safe_drop_index(messages, 2)
        start, end = expand_tool_block(messages, safe)
        self.assertEqual((start, end), (1, 3))
        for i in range(end, start - 1, -1):
            storage.delete_message(i)
        remaining = storage.messages
        self.assertEqual(len(remaining), 2)
        self.assertNotIn("call_1", str([getattr(m, "tool_calls", None) for m in remaining]))
        self.assertFalse(any(isinstance(m, ToolMessage) for m in remaining))

    def test_find_safe_drop_index_out_of_range_passthrough(self):
        messages = self._history()
        self.assertEqual(find_safe_drop_index(messages, 99), 99)
        self.assertEqual(find_safe_drop_index([], 0), 0)
        self.assertEqual(expand_tool_block([], 0), (0, 0))


if __name__ == "__main__":
    unittest.main()
