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

    async def test_different_bot_uuid_same_chat_id_isolated(self):
        """Regression: the same chat_id under different bot_uuids must not share state."""
        b1 = MemoryDeepAgentBackend("bot-a", 12345)
        b2 = MemoryDeepAgentBackend("bot-b", 12345)
        self.assertIsNot(b1.backend, b2.backend)

    async def test_clear_is_scoped_to_bot_uuid(self):
        """Regression: clearing one bot's state must not touch another's."""
        b_a = MemoryDeepAgentBackend("bot-a", 12345)
        b_b = MemoryDeepAgentBackend("bot-b", 12345)
        b_b_instance = b_b.backend
        await b_a.clear()
        # bot-b is untouched
        self.assertIs(MemoryDeepAgentBackend("bot-b", 12345).backend, b_b_instance)

    async def test_clear_is_idempotent_and_safe_when_missing(self):
        """Clearing an unknown key must not raise."""
        backend = MemoryDeepAgentBackend("never-seen", 99999)
        await backend.clear()  # no prior _build_backend call
        await backend.clear()  # twice, still safe

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

    async def test_clear_refuses_path_outside_workspace(self):
        """Regression: bot_uuid carrying path-traversal characters is rejected
        at construction (Layer 1 of the security model), so ``clear()`` never
        has the chance to operate on a path outside the workspace.

        The original regression test assumed Layer 2 (containment check at
        ``clear()`` time) was the primary defense. After adding Layer 1, the
        unsafe ``bot_uuid`` never makes it past ``__init__``, but Layer 2
        still exists as defense in depth (see ``test_clear_layer2_refuses_runtime_workspace_mutation``).
        """
        # Plant a victim directory OUTSIDE the workspace.
        outside_base = tempfile.mkdtemp(prefix="manolo_bot_test_outside_")
        victim = os.path.join(outside_base, "victim_dir")
        os.makedirs(victim)
        victim_file = os.path.join(victim, "important.txt")
        with open(victim_file, "w") as f:
            f.write("do-not-delete")

        try:
            workspace = tempfile.mkdtemp(prefix="manolo_bot_test_ws_")
            traversal_uuid = os.pardir + os.sep + os.path.basename(outside_base) + os.sep + "victim_dir"
            with self.assertRaises(ValueError) as cm:
                FilesystemDeepAgentBackend(traversal_uuid, 0, workspace)
            self.assertIn("bot_uuid", str(cm.exception))
            # Victim file is preserved because construction failed.
            self.assertTrue(os.path.exists(victim_file))
        finally:
            shutil.rmtree(workspace)
            shutil.rmtree(outside_base)

    def test_bot_uuid_with_path_traversal_raises_at_construction(self):
        """Layer 1: bot_uuid containing '..' is rejected at construction."""
        with self.assertRaises(ValueError) as cm:
            FilesystemDeepAgentBackend("../etc", 12345, self.workspace)
        self.assertIn("bot_uuid", str(cm.exception))

    def test_bot_uuid_with_slash_raises_at_construction(self):
        """Layer 1: bot_uuid containing '/' is rejected."""
        with self.assertRaises(ValueError):
            FilesystemDeepAgentBackend("foo/bar", 12345, self.workspace)

    def test_bot_uuid_with_dot_only_raises_at_construction(self):
        """Layer 1: bot_uuid equal to '.' is rejected."""
        with self.assertRaises(ValueError):
            FilesystemDeepAgentBackend(".", 12345, self.workspace)

    def test_empty_bot_uuid_raises_at_construction(self):
        """Layer 1: empty bot_uuid is rejected."""
        with self.assertRaises(ValueError):
            FilesystemDeepAgentBackend("", 12345, self.workspace)

    def test_bot_uuid_with_non_string_raises_at_construction(self):
        """Layer 1: non-string bot_uuid is rejected."""
        with self.assertRaises(ValueError):
            FilesystemDeepAgentBackend(None, 12345, self.workspace)  # type: ignore[arg-type]

    def test_relative_workspace_path_raises_at_construction(self):
        """Layer 1: relative workspace_path is rejected."""
        with self.assertRaises(ValueError) as cm:
            FilesystemDeepAgentBackend("ok-bot", 12345, "relative/path")
        self.assertIn("workspace_path", str(cm.exception))

    def test_empty_workspace_path_raises_at_construction(self):
        """Layer 1: empty workspace_path is rejected."""
        with self.assertRaises(ValueError):
            FilesystemDeepAgentBackend("ok-bot", 12345, "")

    def test_valid_config_passes_validation(self):
        """Layer 1: realistic bot_uuid (with dashes and underscores) and absolute
        workspace_path construct successfully."""
        backend = FilesystemDeepAgentBackend("valid-bot_123", 456, self.workspace)
        self.assertEqual(backend.bot_uuid, "valid-bot_123")
        self.assertEqual(backend.chat_id, 456)

    async def test_layer2_refuses_symlink_escape_at_construction(self):
        """Layer 2 (defense in depth): if the workspace contains a symlink
        pointing outside, a valid ``bot_uuid`` whose name matches that
        symlink makes the resolved ``chat_path`` escape the workspace root.
        ``os.path.realpath`` follows the symlink during construction (via
        ``_build_backend``); the containment check rejects the backend
        before any agent can run.
        """
        workspace = tempfile.mkdtemp(prefix="manolo_bot_test_ws_")
        outside = tempfile.mkdtemp(prefix="manolo_bot_test_outside_")
        # Plant a symlink inside the workspace that points outside.
        os.symlink(outside, os.path.join(workspace, "escape_link"))
        try:
            # Layer 1 passes (bot_uuid matches the regex), Layer 2 must catch
            # the symlink escape at construction.
            with self.assertRaises(ValueError) as cm:
                FilesystemDeepAgentBackend("escape_link", 12345, workspace)
            self.assertIn("outside workspace_path", str(cm.exception))
        finally:
            shutil.rmtree(workspace)
            shutil.rmtree(outside)

    def test_layer2_passes_for_normal_chat_path(self):
        """Smoke test: Layer 2 must NOT raise for an ordinary chat path
        (no symlink escape, no traversal)."""
        workspace = tempfile.mkdtemp(prefix="manolo_bot_test_ws_")
        try:
            backend = FilesystemDeepAgentBackend("ok-bot", 12345, workspace)
            chat_path = backend._safe_chat_path()
            self.assertIn("ok-bot", chat_path)
            self.assertIn("12345", chat_path)
        finally:
            shutil.rmtree(workspace)


if __name__ == "__main__":
    unittest.main()
