import os
import shutil
import tempfile
import unittest

from manolo_bot.storage.deep_agent_backends.filesystem_backend import FilesystemDeepAgentBackend
from manolo_bot.storage.deep_agent_backends.memory_backend import MemoryDeepAgentBackend


class TestMemoryDeepAgentBackend(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot_uuid = "test-bot"

    async def test_initialization(self):
        backend = MemoryDeepAgentBackend(self.bot_uuid, 12345)
        self.assertEqual(backend.bot_uuid, self.bot_uuid)
        self.assertEqual(backend.chat_id, 12345)

    async def test_same_chat_id_shares_backend(self):
        b1 = MemoryDeepAgentBackend(self.bot_uuid, 12345)
        b2 = MemoryDeepAgentBackend(self.bot_uuid, 12345)
        self.assertIs(b1.backend, b2.backend)

    async def test_different_chat_id_different_backend(self):
        b1 = MemoryDeepAgentBackend(self.bot_uuid, 12345)
        b2 = MemoryDeepAgentBackend(self.bot_uuid, 67890)
        self.assertIsNot(b1.backend, b2.backend)

    async def test_clear_removes_backend(self):
        b1 = MemoryDeepAgentBackend(self.bot_uuid, 12345)
        backend_instance = b1.backend
        await b1.clear()
        b2 = MemoryDeepAgentBackend(self.bot_uuid, 12345)
        self.assertIsNot(backend_instance, b2.backend)

    async def test_backend_is_state_backend(self):
        from deepagents.backends import StateBackend

        backend = MemoryDeepAgentBackend(self.bot_uuid, 12345)
        self.assertIsInstance(backend.backend, StateBackend)


class TestFilesystemDeepAgentBackend(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot_uuid = "test-bot"
        self.workspace = tempfile.mkdtemp(prefix="manolo_bot_test_agent_backend_")

    def tearDown(self):
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)

    async def test_initialization(self):
        backend = FilesystemDeepAgentBackend(self.bot_uuid, 12345, self.workspace)
        self.assertEqual(backend.bot_uuid, self.bot_uuid)
        self.assertEqual(backend.chat_id, 12345)

    async def test_path_contains_bot_uuid_and_chat_id(self):
        backend = FilesystemDeepAgentBackend(self.bot_uuid, 12345, self.workspace)
        cwd = str(backend.backend.cwd)
        self.assertIn(self.bot_uuid, cwd)
        self.assertIn("12345", cwd)

    async def test_different_chat_id_different_path(self):
        b1 = FilesystemDeepAgentBackend(self.bot_uuid, 12345, self.workspace)
        b2 = FilesystemDeepAgentBackend(self.bot_uuid, 67890, self.workspace)
        self.assertNotEqual(str(b1.backend.cwd), str(b2.backend.cwd))

    async def test_clear_removes_chat_directory(self):
        backend = FilesystemDeepAgentBackend(self.bot_uuid, 12345, self.workspace)
        chat_path = os.path.join(self.workspace, self.bot_uuid, "12345")
        backend.backend.write("/test.txt", "hello")
        self.assertTrue(os.path.exists(chat_path))
        await backend.clear()
        self.assertFalse(os.path.exists(chat_path))

    async def test_backend_is_filesystem_backend(self):
        from deepagents.backends.filesystem import FilesystemBackend

        backend = FilesystemDeepAgentBackend(self.bot_uuid, 12345, self.workspace)
        self.assertIsInstance(backend.backend, FilesystemBackend)

    async def test_virtual_mode_default(self):
        backend = FilesystemDeepAgentBackend(self.bot_uuid, 12345, self.workspace)
        self.assertTrue(backend._virtual_mode)

    async def test_virtual_mode_disabled(self):
        backend = FilesystemDeepAgentBackend(self.bot_uuid, 12345, self.workspace, virtual_mode=False)
        self.assertFalse(backend._virtual_mode)


if __name__ == "__main__":
    unittest.main()
