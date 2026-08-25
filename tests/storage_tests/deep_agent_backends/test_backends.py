import os
import shutil
import tempfile
import unittest

from manolo_bot.storage.deep_agent_backends.filesystem_backend import FilesystemDeepAgentBackend
from manolo_bot.storage.deep_agent_backends.memory_backend import MemoryDeepAgentBackend
from manolo_bot.storage.deep_agent_backends.memory_filesystem_backend import (
    MemoryFilesystemDeepAgentBackend,
)
from manolo_bot.storage.deep_agent_backends.skills_filesystem_backend import (
    SkillsFilesystemDeepAgentBackend,
)


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


class TestMemoryFilesystemDeepAgentBackend(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot_uuid = "test-bot"
        self.memory_root = tempfile.mkdtemp(prefix="manolo_bot_test_memory_backend_")

    def tearDown(self):
        if os.path.exists(self.memory_root):
            shutil.rmtree(self.memory_root)

    def _make(self, chat_id=12345, **kwargs):
        return MemoryFilesystemDeepAgentBackend(
            bot_uuid=kwargs.pop("bot_uuid", self.bot_uuid),
            chat_id=chat_id,
            memory_root=kwargs.pop("memory_root", self.memory_root),
            **kwargs,
        )

    async def test_initialization(self):
        from deepagents.backends.filesystem import FilesystemBackend

        backend = self._make()
        self.assertEqual(backend.bot_uuid, self.bot_uuid)
        self.assertEqual(backend.chat_id, 12345)
        self.assertIsInstance(backend.backend, FilesystemBackend)
        # Middleware-facing backend: virtual_mode=False so MemoryMiddleware's
        # raw absolute source paths pass through unchanged.
        self.assertFalse(backend.backend.virtual_mode)
        # Backend is rooted at the chat memory dir.
        chat_dir = os.path.join(self.memory_root, self.bot_uuid, "12345")
        self.assertEqual(str(backend.backend.cwd.resolve()), os.path.realpath(chat_dir))

    async def test_different_chat_id_different_memory_dir(self):
        b1 = self._make(chat_id=12345)
        b2 = self._make(chat_id=67890)
        self.assertNotEqual(str(b1.backend.cwd), str(b2.backend.cwd))
        self.assertNotEqual(b1.source, b2.source)

    async def test_seeding_creates_dir_and_non_empty_agents_file(self):
        """Construction seeds the chat memory dir with a NON-EMPTY AGENTS.md —
        deepagents' MemoryMiddleware skips empty sources (HTML comments are
        stripped) and would hide the source path from the agent."""
        from manolo_bot.storage.deep_agent_backends.memory_filesystem_backend import MEMORY_TEMPLATE

        backend = self._make()
        agents_file = backend.source
        self.assertTrue(os.path.exists(agents_file))
        content = open(agents_file).read()
        self.assertEqual(content, MEMORY_TEMPLATE)
        self.assertTrue(content.strip(), "seeded memory must be non-empty")

    async def test_seeding_preserves_existing_file(self):
        """An existing AGENTS.md (e.g. written back by the agent) is untouched."""
        backend = self._make()
        agents_file = backend.source
        with open(agents_file, "w") as f:
            f.write("# custom learnings")
        # Reconstructing the wrapper must not overwrite the file.
        backend2 = self._make()
        self.assertEqual(open(backend2.source).read(), "# custom learnings")

    async def test_source_is_chat_scoped_agents_file(self):
        backend = self._make()
        self.assertEqual(
            backend.source,
            os.path.join(os.path.realpath(self.memory_root), self.bot_uuid, "12345", "AGENTS.md"),
        )

    async def test_store_defaults_to_process_local_in_memory_store(self):
        from langgraph.store.memory import InMemoryStore

        backend = self._make()
        self.assertIsInstance(backend.store, InMemoryStore)

    async def test_store_injected(self):
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore()
        backend = self._make(store=store)
        self.assertIs(backend.store, store)

    async def test_clear_removes_chat_memory_dir(self):
        backend = self._make()
        chat_dir = os.path.join(self.memory_root, self.bot_uuid, "12345")
        self.assertTrue(os.path.exists(chat_dir))
        await backend.clear()
        self.assertFalse(os.path.exists(chat_dir))
        # Idempotent when already gone.
        await backend.clear()

    async def test_clear_is_scoped_to_chat(self):
        """Clearing one chat's memory must not touch another's."""
        b_a = self._make(chat_id=12345)
        b_b = self._make(chat_id=67890)
        await b_a.clear()
        self.assertTrue(os.path.exists(b_b.source))

    async def test_routes_single_chat_dir_route(self):
        """routes() returns a single route for the chat memory dir, backed by a
        SEPARATE virtual_mode=True FilesystemBackend (distinct from the
        middleware-facing virtual_mode=False backend). CompositeBackend strips
        the route prefix and hands the target a leading-slash key, which
        virtual_mode=True resolves under the chat root."""
        from deepagents.backends.filesystem import FilesystemBackend

        backend = self._make()
        routes = backend.routes(["/ignored/source"])
        chat_dir = os.path.join(os.path.realpath(self.memory_root), self.bot_uuid, "12345")
        self.assertEqual(set(routes.keys()), {chat_dir + "/"})
        route_backend = routes[chat_dir + "/"]
        self.assertIsInstance(route_backend, FilesystemBackend)
        self.assertTrue(route_backend.virtual_mode)
        self.assertEqual(str(route_backend.cwd.resolve()), chat_dir)
        # Distinct instance from the middleware-facing backend.
        self.assertIsNot(route_backend, backend.backend)
        self.assertFalse(backend.backend.virtual_mode)

    async def test_absolute_paths_resolve_on_host(self):
        """The middleware-facing virtual_mode=False backend reads the seeded
        AGENTS.md through its raw ABSOLUTE source path — the exact path
        MemoryMiddleware uses (virtual_mode=True would append the absolute
        path under the root and silently miss the file)."""
        backend = self._make()
        result = backend.backend.download_files([backend.source])
        self.assertIsNone(result[0].error)
        self.assertIn(b"Learnings from this conversation", result[0].content)

    def test_bot_uuid_with_path_traversal_raises_at_construction(self):
        """Layer 1: bot_uuid containing '..' is rejected at construction."""
        with self.assertRaises(ValueError) as cm:
            self._make(bot_uuid="../etc")
        self.assertIn("bot_uuid", str(cm.exception))

    def test_bot_uuid_with_slash_raises_at_construction(self):
        """Layer 1: bot_uuid containing '/' is rejected."""
        with self.assertRaises(ValueError):
            self._make(bot_uuid="foo/bar")

    def test_relative_memory_root_raises_at_construction(self):
        """Layer 1: relative memory_root is rejected."""
        with self.assertRaises(ValueError) as cm:
            self._make(memory_root="relative/path")
        self.assertIn("memory_root", str(cm.exception))

    def test_empty_memory_root_raises_at_construction(self):
        """Layer 1: empty memory_root is rejected."""
        with self.assertRaises(ValueError):
            self._make(memory_root="")

    def test_layer2_refuses_symlink_escape_at_construction(self):
        """Layer 2 (defense in depth): if the memory root contains a symlink
        pointing outside, a valid bot_uuid whose name matches that symlink
        makes the resolved chat dir escape the root. The containment check
        rejects the backend before any agent can run."""
        outside = tempfile.mkdtemp(prefix="manolo_bot_test_outside_")
        os.symlink(outside, os.path.join(self.memory_root, "escape_link"))
        try:
            with self.assertRaises(ValueError) as cm:
                self._make(bot_uuid="escape_link")
            self.assertIn("outside memory_root", str(cm.exception))
        finally:
            shutil.rmtree(outside)


class TestSkillsFilesystemDeepAgentBackend(unittest.IsolatedAsyncioTestCase):
    async def test_routes_prefix_per_source(self):
        """routes() maps each skill source directory to a route prefix ending
        in '/', rooted at that source with virtual_mode=True."""
        from deepagents.backends.filesystem import FilesystemBackend

        backend = SkillsFilesystemDeepAgentBackend()
        routes = backend.routes(["/path/to/skills", ("/path/to/other", "Label")])

        self.assertEqual(set(routes.keys()), {"/path/to/skills/", "/path/to/other/"})
        for route_prefix, route_backend in routes.items():
            self.assertIsInstance(route_backend, FilesystemBackend)
            self.assertTrue(route_backend.virtual_mode)
            self.assertEqual(str(route_backend.cwd.resolve()), route_prefix.rstrip("/"))

    async def test_routes_dedupes_repeated_prefixes(self):
        backend = SkillsFilesystemDeepAgentBackend()
        routes = backend.routes(["/a", "/a", ("/a", "Label")])
        self.assertEqual(len(routes), 1)
        self.assertIn("/a/", routes)

    async def test_routes_empty_for_no_sources(self):
        backend = SkillsFilesystemDeepAgentBackend()
        self.assertEqual(backend.routes([]), {})


if __name__ == "__main__":
    unittest.main()
