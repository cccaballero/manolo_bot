import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from manolo_bot.storage.messages.base import SUMMARY_PREFIX, get_messages_key
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


class TestMessagesStorageKeys(unittest.IsolatedAsyncioTestCase):
    """Storage keys are opaque strings; a numeric id and its string form match.

    ``_chats`` is a process-global that survives between tests, so every test
    here uses its own bot UUID and chat key.
    """

    def setUp(self):
        self._storages: list[MemoryMessagesStorage] = []

    async def asyncTearDown(self):
        for storage in self._storages:
            await storage.clear_messages()

    def _storage(self, bot_uuid: str, chat_id: str | int) -> MemoryMessagesStorage:
        storage = MemoryMessagesStorage(bot_uuid=bot_uuid, chat_id=chat_id)
        self._storages.append(storage)
        return storage

    def test_blank_or_missing_key_raises(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MemoryMessagesStorage(bot_uuid="test-bot-blank-key", chat_id=value)

    def test_key_is_normalized_on_construction(self):
        storage = MemoryMessagesStorage(bot_uuid="test-bot-normalized-key", chat_id=" 424243 ")
        self.assertEqual(storage.chat_id, "424243")
        self.assertIsInstance(storage.chat_id, str)

    async def test_numeric_and_string_keys_share_the_same_history(self):
        as_int = self._storage("test-bot-numeric-key", 991001)
        as_str = self._storage("test-bot-numeric-key", "991001")
        self.assertEqual(as_str.chat_id, as_int.chat_id)

        as_int.add_message(HumanMessage(content="written through the int key"))
        await as_int.commit()

        await as_str.refresh_messages()
        self.assertEqual([m.content for m in as_str.messages], ["written through the int key"])
        self.assertEqual(as_str.messages, as_int.messages)

    async def test_opaque_key_round_trip_commit_refresh_and_clear(self):
        key = "dashboard:user:agent:uuid"
        writer = self._storage("test-bot-opaque-key", key)
        writer.add_message(HumanMessage(content="opaque key message"))
        await writer.commit()

        reader = self._storage("test-bot-opaque-key", key)
        await reader.refresh_messages()
        self.assertEqual([m.content for m in reader.messages], ["opaque key message"])

        await reader.clear_messages()
        await writer.refresh_messages()
        self.assertEqual(writer.messages, [])

    async def test_different_bots_with_the_same_key_do_not_share_history(self):
        key = "shared-chat-key"
        first = self._storage("test-bot-isolation-a", key)
        second = self._storage("test-bot-isolation-b", key)

        first.add_message(HumanMessage(content="from bot a"))
        second.add_message(HumanMessage(content="from bot b"))
        await first.commit()
        await second.commit()

        await first.refresh_messages()
        await second.refresh_messages()
        self.assertEqual([m.content for m in first.messages], ["from bot a"])
        self.assertEqual([m.content for m in second.messages], ["from bot b"])

        # Clearing one bot's history leaves the other bot's history intact.
        await first.clear_messages()
        await second.refresh_messages()
        self.assertEqual([m.content for m in second.messages], ["from bot b"])


class TestGetMessagesKey(unittest.TestCase):
    def test_numeric_and_string_chat_ids_address_the_same_key(self):
        self.assertEqual(get_messages_key("bot", 42), get_messages_key("bot", "42"))
        self.assertEqual(get_messages_key("bot", 42), "bot:42")

    def test_opaque_chat_key_is_preserved(self):
        self.assertEqual(get_messages_key("bot", "C012345"), "bot:C012345")

    def test_colon_key_is_escaped(self):
        self.assertEqual(get_messages_key("bot", "documents:42"), "bot:documents%3A42")

    def test_blank_keys_raise(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    get_messages_key("bot", value)


if __name__ == "__main__":
    unittest.main()
