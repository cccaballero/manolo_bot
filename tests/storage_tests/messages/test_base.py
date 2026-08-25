import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from manolo_bot.storage.messages.base import SUMMARY_PREFIX
from manolo_bot.storage.messages.memory_storage import MemoryMessagesStorage


class TestMessagesSummary(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.storage = MemoryMessagesStorage(bot_uuid="test-bot", chat_id=424243)

    async def asyncTearDown(self):
        await self.storage.clear_messages()

    def _summary_messages(self):
        return [
            sm.message
            for sm in self.storage._messages
            if not sm.deleted
            and isinstance(sm.message, SystemMessage)
            and isinstance(sm.message.content, str)
            and sm.message.content.startswith(SUMMARY_PREFIX)
        ]

    def test_get_summary_returns_none_when_absent(self):
        self.assertIsNone(self.storage.get_summary())
        self.storage.add_message(HumanMessage(content="hello"))
        self.assertIsNone(self.storage.get_summary())

    def test_set_summary_get_summary_round_trip(self):
        self.storage.set_summary("The user prefers short answers.")
        self.assertEqual(self.storage.get_summary(), "The user prefers short answers.")
        # Summary is stored as a flagged SystemMessage at the front.
        self.assertEqual(len(self.storage.messages), 1)
        self.assertIsInstance(self.storage.messages[0], SystemMessage)
        self.assertTrue(self.storage.messages[0].content.startswith(SUMMARY_PREFIX))

    def test_set_summary_replaces_previous(self):
        self.storage.set_summary("First summary")
        self.storage.set_summary("Second summary")
        self.assertEqual(self.storage.get_summary(), "Second summary")
        # Only one summary message remains.
        self.assertEqual(len(self._summary_messages()), 1)

    def test_summary_first_ordering(self):
        self.storage.add_message(HumanMessage(content="m1"))
        self.storage.add_message(HumanMessage(content="m2"))
        self.storage.set_summary("A summary")
        messages = self.storage.messages
        self.assertEqual(len(messages), 3)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertTrue(messages[0].content.startswith(SUMMARY_PREFIX))
        self.assertEqual([m.content for m in messages[1:]], ["m1", "m2"])

    async def test_summary_survives_commit_and_refresh(self):
        self.storage.add_message(HumanMessage(content="m1"))
        self.storage.add_message(HumanMessage(content="m2"))
        self.storage.set_summary("Persisted summary")
        await self.storage.commit()
        await self.storage.refresh_messages()

        self.assertEqual(self.storage.get_summary(), "Persisted summary")
        messages = self.storage.messages
        self.assertEqual(len(messages), 3)
        # Summary stays at the front after a commit + refresh.
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertTrue(messages[0].content.startswith(SUMMARY_PREFIX))
        self.assertEqual([m.content for m in messages[1:]], ["m1", "m2"])

    async def test_summary_replacement_survives_commit(self):
        self.storage.set_summary("Old summary")
        self.storage.add_message(HumanMessage(content="m1"))
        await self.storage.commit()

        self.storage.set_summary("New summary")
        await self.storage.commit()
        await self.storage.refresh_messages()

        self.assertEqual(self.storage.get_summary(), "New summary")
        self.assertEqual(len(self._summary_messages()), 1)
        self.assertEqual([m.content for m in self.storage.messages[1:]], ["m1"])

    def test_delete_message_indexing_after_set_summary(self):
        for i in range(5):
            self.storage.add_message(HumanMessage(content=f"m{i}"))
        self.storage.set_summary("A summary")

        # delete_message(0) removes the summary itself.
        self.storage.delete_message(0)
        self.assertIsNone(self.storage.get_summary())
        self.assertEqual([m.content for m in self.storage.messages], ["m0", "m1", "m2", "m3", "m4"])

        # Re-add a summary; delete_message(1) removes the first real message.
        self.storage.set_summary("A summary")
        self.storage.delete_message(1)
        self.assertEqual(self.storage.get_summary(), "A summary")
        self.assertEqual([m.content for m in self.storage.messages[1:]], ["m1", "m2", "m3", "m4"])


if __name__ == "__main__":
    unittest.main()
