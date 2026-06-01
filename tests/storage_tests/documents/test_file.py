import shutil
import tempfile
import unittest
from pathlib import Path

from manolo_bot.storage.documents.file import FileDocumentStorage


class TestFileDocumentStorage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot_uuid = "test-bot"
        self.test_dir = Path(tempfile.mkdtemp(prefix="manolo_bot_test_docs_"))
        self.storage = FileDocumentStorage(bot_uuid=self.bot_uuid, base_path=str(self.test_dir))
        self.chat_id = 12345

    def tearDown(self):
        # Clean up test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

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


if __name__ == "__main__":
    unittest.main()
