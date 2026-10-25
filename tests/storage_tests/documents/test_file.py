import os
import shutil
import tempfile
import unittest
from pathlib import Path

from manolo_bot.storage.documents.file_storage import FileDocumentsStorage
from manolo_bot.storage.keys import storage_path_segment


class TestFileDocumentsStorage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot_uuid = "test-bot"
        self.test_dir = Path(tempfile.mkdtemp(prefix="manolo_bot_test_docs_"))
        self.storage = FileDocumentsStorage(bot_uuid=self.bot_uuid, base_path=str(self.test_dir))
        self.chat_id = 12345

    def tearDown(self):
        # Clean up test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @property
    def documents_base(self) -> Path:
        return self.test_dir / self.bot_uuid

    async def test_store_and_retrieve(self):
        # Arrange
        filename = "test_doc"
        text = "This is the document content."

        # Act
        await self.storage.store(self.chat_id, filename, text)
        result = await self.storage.retrieve(self.chat_id, filename)

        # Assert
        self.assertEqual(result, text)

    async def test_retrieve_non_existent(self):
        # Act
        result = await self.storage.retrieve(self.chat_id, "ghost_doc")

        # Assert
        self.assertIsNone(result)

    async def test_list_documents(self):
        # Arrange
        await self.storage.store(self.chat_id, "doc1", "content1")
        await self.storage.store(self.chat_id, "doc2", "content2")

        # Act
        docs = await self.storage.list_documents(self.chat_id)

        # Assert
        self.assertIn("doc1", docs)
        self.assertIn("doc2", docs)
        self.assertEqual(len(docs), 2)

    async def test_clear(self):
        # Arrange
        await self.storage.store(self.chat_id, "doc1", "content1")

        # Act
        await self.storage.clear(self.chat_id)
        docs = await self.storage.list_documents(self.chat_id)

        # Assert
        self.assertEqual(len(docs), 0)

    async def test_security_traversal(self):
        # Arrange
        filename = "../other_chat/stolen"
        text = "This should not be stored"

        # Act & Assert
        with self.assertRaises(ValueError):
            await self.storage.store(self.chat_id, filename, text)

        # Act & Assert
        result = await self.storage.retrieve(self.chat_id, filename)
        self.assertIsNone(result)

    async def test_numeric_and_string_keys_address_the_same_records(self):
        # Store with an int, read/clear with the string form.
        await self.storage.store(42, "doc", "content")
        self.assertEqual(await self.storage.retrieve("42", "doc"), "content")
        self.assertEqual(await self.storage.list_documents("42"), ["doc"])
        await self.storage.clear("42")
        self.assertIsNone(await self.storage.retrieve(42, "doc"))

        # Store with a string, read/clear with the int form.
        await self.storage.store("77", "other", "content2")
        self.assertEqual(await self.storage.retrieve(77, "other"), "content2")
        self.assertEqual(await self.storage.list_documents(77), ["other"])
        await self.storage.clear(77)
        self.assertEqual(await self.storage.list_documents("77"), [])

    async def test_numeric_key_keeps_its_existing_directory(self):
        await self.storage.store(42, "doc", "content")

        self.assertTrue((self.documents_base / "42" / "doc.txt").is_file())

    async def test_opaque_key_lands_in_a_hashed_directory(self):
        key = "C012345"

        await self.storage.store(key, "doc", "content")

        segments = [entry.name for entry in self.documents_base.iterdir()]
        self.assertEqual(segments, [storage_path_segment(key)])
        self.assertRegex(segments[0], r"^h\.[0-9a-f]{40}$")

        chat_path = self.documents_base / segments[0]
        self.assertEqual(Path(os.path.realpath(chat_path)).parent, Path(os.path.realpath(self.documents_base)))
        self.assertEqual(await self.storage.retrieve(key, "doc"), "content")
        self.assertEqual(await self.storage.list_documents(key), ["doc"])

        await self.storage.clear(key)
        self.assertEqual(await self.storage.list_documents(key), [])
        self.assertFalse((chat_path / "doc.txt").exists())

    async def test_traversal_looking_key_is_contained(self):
        key = "../../escape"

        await self.storage.store(key, "doc", "contained")

        # Only the bot_uuid directory exists under the configured base, and only
        # the hashed segment exists inside it: nothing escaped upwards.
        self.assertEqual([entry.name for entry in self.test_dir.iterdir()], [self.bot_uuid])
        chat_path = self.documents_base / storage_path_segment(key)
        self.assertTrue(chat_path.is_dir())
        resolved_base = Path(os.path.realpath(self.documents_base))
        self.assertEqual(Path(os.path.realpath(chat_path)).parent, resolved_base)
        self.assertFalse((self.test_dir / "escape").exists())
        self.assertEqual(await self.storage.retrieve(key, "doc"), "contained")

    async def test_sibling_prefix_filename_cannot_escape_the_chat_directory(self):
        # Chat directories 1 and 10 share the "/base/<bot>/1" prefix. A filename
        # that hops to the sibling must not read or overwrite its documents.
        await self.storage.store(1, "own", "chat-1 secret")
        await self.storage.store(10, "private", "chat-10 TOP SECRET")

        self.assertIsNone(await self.storage.retrieve(1, "../10/private"))
        with self.assertRaises(ValueError):
            await self.storage.store(1, "../10/private", "overwritten")

        self.assertEqual(await self.storage.retrieve(10, "private"), "chat-10 TOP SECRET")

    async def test_blank_key_raises(self):
        for key in (None, "", "   ", "\n\t "):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    await self.storage.store(key, "doc", "content")
                with self.assertRaises(ValueError):
                    await self.storage.retrieve(key, "doc")
                with self.assertRaises(ValueError):
                    await self.storage.list_documents(key)
                with self.assertRaises(ValueError):
                    await self.storage.clear(key)


if __name__ == "__main__":
    unittest.main()
