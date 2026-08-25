import os
import pathlib
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from langchain_core.messages import SystemMessage
from langgraph.store.memory import InMemoryStore

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.llmdeepagent import LLMDeepAgent, _resolve_skills_sources
from manolo_bot.storage.deep_agent_backends.base import BaseDeepAgentBackend
from manolo_bot.storage.deep_agent_backends.memory_filesystem_backend import (
    MemoryFilesystemDeepAgentBackend,
)
from manolo_bot.storage.deep_agent_backends.skills_filesystem_backend import (
    SkillsFilesystemDeepAgentBackend,
)


def _make_config(**overrides) -> BotConfig:
    defaults = dict(
        bot_uuid="test-bot",
        bot_name="TestBot",
        bot_username="test_bot",
        bot_token="123456:ABC",
        user_id=0,
        use_tools=False,
        enable_mcp=False,
        mcp_servers_config={},
        context_max_tokens=4000,
        web_content_request_timeout=30,
        can_use_tavily_search=False,
        max_document_size=10 * 1024 * 1024,
        max_voice_size=10 * 1024 * 1024,
    )
    defaults.update(overrides)
    return BotConfig(**defaults)


class TestLLMDeepAgent(unittest.IsolatedAsyncioTestCase):
    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_initialization_uses_create_deep_agent(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        # Arrange
        bot_config = _make_config(use_tools=True)

        mock_messages_storage = MagicMock()

        mock_tools = ["tool1", "tool2"]
        mock_get_all_tools.return_value = mock_tools

        mock_agent = MagicMock()
        mock_create_deep_agent.return_value = mock_agent

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        # Act
        agent = LLMDeepAgent(mock_llm, bot_config, system_instructions, mock_messages_storage)
        await agent.initialize_async_resources()

        # Assert
        mock_get_all_tools.assert_called_once()
        mock_create_deep_agent.assert_called_once()
        call_kwargs = mock_create_deep_agent.call_args.kwargs
        self.assertEqual(call_kwargs["model"], mock_llm)
        self.assertEqual(call_kwargs["subagents"], [])
        middleware = call_kwargs["middleware"]
        middleware_names = [type(m).__name__ for m in middleware]
        self.assertIn("FilesystemMiddleware", middleware_names)
        self.assertIn("TodoListMiddleware", middleware_names)
        # No memory configured → no MemoryMiddleware, behavior identical to today.
        self.assertNotIn("MemoryMiddleware", middleware_names)
        self.assertNotIn("store", call_kwargs)
        self.assertEqual(agent.agent, mock_agent)

    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_deep_agent_can_receive_custom_backend(self, mock_create_agent):
        """Verify backend injection works."""
        mock_backend = MagicMock()
        mock_backend.backend = MagicMock()  # the actual BackendProtocol
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        system_instructions = [SystemMessage(content="Test")]

        agent = LLMDeepAgent(mock_llm, bot_config, system_instructions, mock_messages_storage, backend=mock_backend)

        self.assertIs(agent._backend, mock_backend.backend)
        self.assertIs(agent._backend_wrapper, mock_backend)

    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_deep_agent_inherits_answer_message_from_llm_agent(self, mock_create_agent):
        """LLMDeepAgent should reuse LLMAgent's answer_message via self.agent.ainvoke."""
        mock_ai_message = MagicMock()
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_ai_message]})

        bot_config = _make_config(use_tools=True)

        mock_messages_storage = MagicMock()
        mock_messages_storage.messages = []

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]

        agent = LLMDeepAgent(mock_llm, bot_config, system_instructions, mock_messages_storage)
        agent.agent = mock_agent
        agent.truncate_chat_context = AsyncMock()  # isolate from token counting

        response = await agent.answer_message(1, "Hello")

        self.assertEqual(response, mock_ai_message)
        mock_agent.ainvoke.assert_awaited_once()

    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_deep_agent_defaults_to_state_backend(self, mock_create_agent):
        """When no backend is provided, defaults to StateBackend. With no skills_backend
        either, no skills middleware is wired and ``_skills_backend`` stays None."""
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        system_instructions = [SystemMessage(content="Test")]

        agent = LLMDeepAgent(mock_llm, bot_config, system_instructions, mock_messages_storage)

        self.assertIsNone(agent._backend_wrapper)
        self.assertIsInstance(agent._backend, StateBackend)
        # No skills_backend was injected → no implicit default; SkillsMiddleware
        # will be omitted by initialize_async_resources.
        self.assertIsNone(agent._skills_backend)

    async def test_deep_agent_clean_context_clears_backend_wrapper(self):
        """clean_context should also clear the backend wrapper."""
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_messages_storage.chat_id = 123
        mock_messages_storage.clear_messages = AsyncMock()
        mock_llm = MagicMock()
        system_instructions = [SystemMessage(content="Test")]

        mock_backend_wrapper = MagicMock()
        mock_backend_wrapper.backend = MagicMock()
        mock_backend_wrapper.clear = AsyncMock()

        agent = LLMDeepAgent(
            mock_llm, bot_config, system_instructions, mock_messages_storage, backend=mock_backend_wrapper
        )
        agent.documents_storage = None

        await agent.clean_context()

        mock_backend_wrapper.clear.assert_awaited_once()

    def test_resolve_skills_sources_parses_label_syntax(self):
        """_resolve_skills_sources parses bare paths and ::LABEL= syntax."""
        self.assertEqual(_resolve_skills_sources(None), [])
        self.assertEqual(_resolve_skills_sources([]), [])
        self.assertEqual(_resolve_skills_sources(["/a"]), ["/a"])
        self.assertEqual(_resolve_skills_sources(["/a", "/b::LABEL=Alpha"]), ["/a", ("/b", "Alpha")])
        self.assertEqual(_resolve_skills_sources(["/c::LABEL=My Skills"]), [("/c", "My Skills")])
        self.assertEqual(_resolve_skills_sources([""]), [])
        self.assertEqual(_resolve_skills_sources(["/a", "", "/b"]), ["/a", "/b"])

    def test_resolve_skills_sources_passes_tuples_through(self):
        """_resolve_skills_sources passes (path, label) tuples through unchanged."""
        self.assertEqual(_resolve_skills_sources([("/a", "A"), ("/b", "B")]), [("/a", "A"), ("/b", "B")])
        # Mixed strings, tuples, and ::LABEL= syntax coexist.
        self.assertEqual(
            _resolve_skills_sources(["/a", ("/b", "B"), "/c::LABEL=C"]),
            ["/a", ("/b", "B"), ("/c", "C")],
        )

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_passes_skills_when_paths_and_backend_provided(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """When both skills_paths and skills_backend are provided, a SkillsMiddleware is
        wired into the middleware list with the explicit backend."""
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        mock_get_all_tools.return_value = []
        mock_create_deep_agent.return_value = MagicMock()

        skills_wrapper = SkillsFilesystemDeepAgentBackend()

        agent = LLMDeepAgent(
            mock_llm,
            bot_config,
            system_instructions,
            mock_messages_storage,
            skills_paths=["/path/to/skills"],
            skills_backend=skills_wrapper,
        )
        await agent.initialize_async_resources()

        middleware = mock_create_deep_agent.call_args.kwargs["middleware"]
        skills_mw = next((m for m in middleware if isinstance(m, SkillsMiddleware)), None)
        self.assertIsNotNone(skills_mw, "SkillsMiddleware not in middleware list")
        self.assertEqual(skills_mw.sources, ["/path/to/skills"])
        self.assertIs(skills_mw._backend, skills_wrapper.backend)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_omits_skills_middleware_when_no_backend(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """skills_paths without skills_backend produces NO SkillsMiddleware. LLMDeepAgent
        does not instantiate a skills backend itself — the caller must inject one.
        A warning is logged so misconfigurations are loud, not silent."""
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        mock_get_all_tools.return_value = []
        mock_create_deep_agent.return_value = MagicMock()

        with self.assertLogs("manolo_bot.ai.llmdeepagent", level="WARNING") as captured:
            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                skills_paths=["/path/to/skills"],
                # No skills_backend injected.
            )
        await agent.initialize_async_resources()

        self.assertEqual(agent._skills_backend, None)
        middleware = mock_create_deep_agent.call_args.kwargs["middleware"]
        self.assertEqual(
            [m for m in middleware if isinstance(m, SkillsMiddleware)],
            [],
            "SkillsMiddleware must be omitted when no skills_backend was injected",
        )
        # Warning was emitted at construction time.
        self.assertTrue(any("skills_backend" in m for m in captured.output))

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_skills_label_parsing_via_ctor(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """skills_paths passed via ctor supports ::LABEL= parsing — provided a
        skills_backend is also injected. Configuration flows in through the
        caller (main.py reads Config and passes it explicitly)."""
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        mock_get_all_tools.return_value = []
        mock_create_deep_agent.return_value = MagicMock()

        skills_wrapper = SkillsFilesystemDeepAgentBackend()

        agent = LLMDeepAgent(
            mock_llm,
            bot_config,
            system_instructions,
            mock_messages_storage,
            skills_paths=["/a::LABEL=Alpha", "/b"],
            skills_backend=skills_wrapper,
        )
        await agent.initialize_async_resources()

        middleware = mock_create_deep_agent.call_args.kwargs["middleware"]
        skills_mw = next((m for m in middleware if isinstance(m, SkillsMiddleware)), None)
        self.assertIsNotNone(skills_mw)
        self.assertIs(skills_mw._backend, skills_wrapper.backend)
        # deepagents 0.7.5 stores paths on `sources` and labels on `source_labels`.
        self.assertEqual(skills_mw.sources, ["/a", "/b"])
        self.assertEqual(skills_mw.source_labels, ["Alpha", "B"])

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_omits_skills_when_neither_set(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """When skills_paths is not set, no SkillsMiddleware is passed."""
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        mock_get_all_tools.return_value = []
        mock_create_deep_agent.return_value = MagicMock()

        agent = LLMDeepAgent(mock_llm, bot_config, system_instructions, mock_messages_storage)
        await agent.initialize_async_resources()

        middleware = mock_create_deep_agent.call_args.kwargs["middleware"]
        skills_mw = [m for m in middleware if isinstance(m, SkillsMiddleware)]
        self.assertEqual(skills_mw, [], "SkillsMiddleware should not be present")

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_skills_backend_independent_from_main_backend(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """skills_backend is independent from the agent's main filesystem backend."""
        main_backend = StateBackend()
        skills_backend = StateBackend()

        class _MainBackendWrapper(BaseDeepAgentBackend):
            def _build_backend(self):
                return main_backend

            async def clear(self):
                pass

        class _SkillsBackendWrapper(BaseDeepAgentBackend):
            def _build_backend(self):
                return skills_backend

            async def clear(self):
                pass

            def routes(self, sources):
                return {f"{s.rstrip('/')}/": skills_backend for s in sources}

        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="Test")]
        mock_get_all_tools.return_value = []
        mock_create_deep_agent.return_value = MagicMock()

        main_wrapper = _MainBackendWrapper(bot_uuid="b", chat_id=1)
        skills_wrapper = _SkillsBackendWrapper(bot_uuid="b", chat_id=1)

        agent = LLMDeepAgent(
            mock_llm,
            bot_config,
            system_instructions,
            mock_messages_storage,
            backend=main_wrapper,
            skills_paths=["/path/to/skills"],
            skills_backend=skills_wrapper,
        )
        await agent.initialize_async_resources()

        middleware = mock_create_deep_agent.call_args.kwargs["middleware"]
        fs_mw = next(m for m in middleware if isinstance(m, FilesystemMiddleware))
        skills_mw = next(m for m in middleware if isinstance(m, SkillsMiddleware))

        # The FilesystemMiddleware is now backed by a CompositeBackend that wraps
        # the per-chat backend and adds a route per skill source — so the agent's
        # read_file tool can access skill files at their absolute paths.
        self.assertIsInstance(fs_mw.backend, CompositeBackend)
        self.assertIs(fs_mw.backend.default, main_backend)
        self.assertIs(skills_mw._backend, skills_backend)
        self.assertIsNot(skills_mw._backend, fs_mw.backend)

    async def test_deep_agent_clean_context_does_not_clear_skills_backend(self):
        """Skills are operator-provided, not per-chat state — clean_context must not clear them."""
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_messages_storage.chat_id = 123
        mock_messages_storage.clear_messages = AsyncMock()
        mock_llm = MagicMock()
        system_instructions = [SystemMessage(content="Test")]

        skills_wrapper = MagicMock()
        skills_wrapper.backend = StateBackend()
        skills_wrapper.routes.return_value = {}
        skills_wrapper.clear = AsyncMock()

        agent = LLMDeepAgent(
            mock_llm,
            bot_config,
            system_instructions,
            mock_messages_storage,
            skills_paths=["/etc/manolo_bot/skills"],
            skills_backend=skills_wrapper,
        )
        agent.documents_storage = None

        await agent.clean_context()

        skills_wrapper.clear.assert_not_awaited()

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_main_backend_has_skill_route_per_source(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """Each configured skill source becomes a CompositeBackend route whose
        FilesystemBackend is rooted at that source with virtual_mode=True. This
        is what lets the agent's read_file tool reach skill files at their
        absolute paths — the per-chat backend alone would reject them.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            bot_config = _make_config()
            mock_messages_storage = MagicMock()
            mock_llm = MagicMock()
            mock_llm.get_num_tokens = MagicMock(return_value=10)
            system_instructions = [SystemMessage(content="Test")]
            mock_get_all_tools.return_value = []
            mock_create_deep_agent.return_value = MagicMock()

            skills_wrapper = SkillsFilesystemDeepAgentBackend()

            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                skills_paths=[tmp_a, (tmp_b, "Project")],
                skills_backend=skills_wrapper,
            )
            await agent.initialize_async_resources()

            middleware = mock_create_deep_agent.call_args.kwargs["middleware"]
            fs_mw = next(m for m in middleware if isinstance(m, FilesystemMiddleware))

            # A CompositeBackend wraps the default (per-chat) backend.
            self.assertIsInstance(fs_mw.backend, CompositeBackend)
            self.assertIsInstance(fs_mw.backend.default, StateBackend)
            # One route per skill source. Trailing slashes are normalized.
            self.assertIn(tmp_a.rstrip("/") + "/", fs_mw.backend.routes)
            self.assertIn(tmp_b.rstrip("/") + "/", fs_mw.backend.routes)
            # Each route is a FilesystemBackend rooted at the source itself
            # with virtual_mode=True, so CompositeBackend's prefix-stripping
            # resolves the stripped path back inside the source directory.
            for route_prefix, route_backend in fs_mw.backend.routes.items():
                self.assertIsInstance(route_backend, FilesystemBackend)
                self.assertTrue(route_backend.virtual_mode)
                self.assertEqual(
                    str(route_backend.cwd.resolve()),
                    route_prefix.rstrip("/"),
                )

            # End-to-end: the route's FilesystemBackend can ls the source path
            # and read a SKILL.md living inside it (simulating what the agent's
            # read_file tool does after this fix).
            skill_root = tmp_a
            skill_dir = pathlib.Path(skill_root) / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: demo\n---\nbody", encoding="utf-8")
            ls_result = fs_mw.backend.ls(skill_root + "/")
            self.assertIsNone(ls_result.error)
            self.assertTrue(any(e["path"].endswith("my-skill/") for e in (ls_result.entries or [])))
            read_result = fs_mw.backend.read(skill_root + "/my-skill/SKILL.md")
            self.assertIsNone(getattr(read_result, "error", None))
            self.assertIn("demo", read_result.file_data["content"])

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_main_backend_has_memory_route_for_chat_dir(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """The per-chat memory wrapper contributes a single CompositeBackend
        route for the chat memory dir, rooted there with virtual_mode=True.
        This is what lets the agent's runtime read_file/write_file/edit_file
        tools reach the real seeded AGENTS.md on disk."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bot_config = _make_config()
            mock_messages_storage = MagicMock()
            mock_llm = MagicMock()
            mock_llm.get_num_tokens = MagicMock(return_value=10)
            system_instructions = [SystemMessage(content="Test")]
            mock_get_all_tools.return_value = []
            mock_create_deep_agent.return_value = MagicMock()

            memory_backend = MemoryFilesystemDeepAgentBackend(bot_uuid="b", chat_id=1, memory_root=tmp)

            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                memory_backend=memory_backend,
            )
            await agent.initialize_async_resources()

            middleware = mock_create_deep_agent.call_args.kwargs["middleware"]
            fs_mw = next(m for m in middleware if isinstance(m, FilesystemMiddleware))

            # A CompositeBackend wraps the default (per-chat) backend.
            self.assertIsInstance(fs_mw.backend, CompositeBackend)
            self.assertIsInstance(fs_mw.backend.default, StateBackend)
            # Single route for the chat memory dir.
            chat_dir = os.path.dirname(memory_backend.source)
            self.assertEqual(set(fs_mw.backend.routes.keys()), {chat_dir + "/"})
            route_backend = fs_mw.backend.routes[chat_dir + "/"]
            self.assertIsInstance(route_backend, FilesystemBackend)
            self.assertTrue(route_backend.virtual_mode)
            self.assertEqual(str(route_backend.cwd.resolve()), chat_dir)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_memory_routes_reach_real_files_on_disk(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """End-to-end: the routed main backend reads the seeded memory file and
        write_file/edit_file updates persist to disk — the durable write-back
        path the agent uses to save learnings."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bot_config = _make_config()
            mock_messages_storage = MagicMock()
            mock_llm = MagicMock()
            mock_llm.get_num_tokens = MagicMock(return_value=10)
            system_instructions = [SystemMessage(content="Test")]
            mock_get_all_tools.return_value = []
            mock_create_deep_agent.return_value = MagicMock()

            memory_backend = MemoryFilesystemDeepAgentBackend(bot_uuid="b", chat_id=1, memory_root=tmp)
            mem_file = memory_backend.source

            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                memory_backend=memory_backend,
            )
            await agent.initialize_async_resources()

            middleware = mock_create_deep_agent.call_args.kwargs["middleware"]
            fs_mw = next(m for m in middleware if isinstance(m, FilesystemMiddleware))

            # read_file via the routed main backend reaches the real file.
            read_result = fs_mw.backend.read(mem_file)
            self.assertIsNone(getattr(read_result, "error", None))
            self.assertIn("Learnings from this conversation", read_result.file_data["content"])

            # write_file via the routed main backend updates the real file.
            write_result = fs_mw.backend.write(mem_file, "# updated fact")
            self.assertIsNone(getattr(write_result, "error", None))
            self.assertEqual(pathlib.Path(mem_file).read_text(encoding="utf-8"), "# updated fact")

            # edit_file via the routed main backend updates the real file.
            edit_result = fs_mw.backend.edit(mem_file, "updated", "edited")
            self.assertIsNone(getattr(edit_result, "error", None))
            self.assertEqual(pathlib.Path(mem_file).read_text(encoding="utf-8"), "# edited fact")

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_memory_explicit_injection(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """Per-chat wrapper injection (main-style wiring): with no memory_paths,
        the agent derives the source from the wrapper's chat-scoped AGENTS.md,
        constructs MemoryMiddleware from the wrapper's backend, and forwards
        the wrapper's store to create_deep_agent."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bot_config = _make_config()
            mock_messages_storage = MagicMock()
            mock_llm = MagicMock()
            mock_llm.get_num_tokens = MagicMock(return_value=10)
            system_instructions = [SystemMessage(content="You are a helpful assistant")]
            mock_get_all_tools.return_value = []
            mock_create_deep_agent.return_value = MagicMock()

            memory_backend = MemoryFilesystemDeepAgentBackend(bot_uuid="b", chat_id=1, memory_root=tmp)

            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                memory_backend=memory_backend,
            )
            await agent.initialize_async_resources()

            call_kwargs = mock_create_deep_agent.call_args.kwargs
            memory_mw = next((m for m in call_kwargs["middleware"] if isinstance(m, MemoryMiddleware)), None)
            self.assertIsNotNone(memory_mw, "MemoryMiddleware not in middleware list")
            # Source derived from the wrapper's chat-scoped AGENTS.md.
            self.assertEqual(memory_mw.sources, [memory_backend.source])
            # The agent constructs MemoryMiddleware from the injected wrapper's
            # backend (mirroring SkillsMiddleware).
            self.assertIs(memory_mw._backend, memory_backend.backend)
            # The wrapper's store is forwarded to create_deep_agent(store=...).
            self.assertIs(call_kwargs["store"], memory_backend.store)
            self.assertIsInstance(call_kwargs["store"], InMemoryStore)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_memory_explicit_paths_override_wrapper_source(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """Explicit memory_paths (advanced use) override the wrapper-derived
        source."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bot_config = _make_config()
            mock_messages_storage = MagicMock()
            mock_llm = MagicMock()
            mock_llm.get_num_tokens = MagicMock(return_value=10)
            system_instructions = [SystemMessage(content="You are a helpful assistant")]
            mock_get_all_tools.return_value = []
            mock_create_deep_agent.return_value = MagicMock()

            memory_backend = MemoryFilesystemDeepAgentBackend(bot_uuid="b", chat_id=1, memory_root=tmp)

            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                memory_paths=["/abs/AGENTS.md"],
                memory_backend=memory_backend,
            )
            await agent.initialize_async_resources()

            memory_mw = next(
                (m for m in mock_create_deep_agent.call_args.kwargs["middleware"] if isinstance(m, MemoryMiddleware)),
                None,
            )
            self.assertIsNotNone(memory_mw)
            self.assertEqual(memory_mw.sources, ["/abs/AGENTS.md"])
            self.assertIs(memory_mw._backend, memory_backend.backend)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_memory_warns_and_omits_when_no_backend(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """memory_paths without memory_backend produces NO MemoryMiddleware and
        no store kwarg. LLMDeepAgent does not instantiate a memory backend
        itself — the caller must inject one. A warning is logged so
        misconfigurations are loud, not silent."""
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        mock_get_all_tools.return_value = []
        mock_create_deep_agent.return_value = MagicMock()

        with self.assertLogs("manolo_bot.ai.llmdeepagent", level="WARNING") as captured:
            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                memory_paths=["/abs/AGENTS.md"],
                # No memory_backend injected.
            )
        await agent.initialize_async_resources()

        self.assertEqual(agent._memory_backend, None)
        call_kwargs = mock_create_deep_agent.call_args.kwargs
        self.assertEqual(
            [m for m in call_kwargs["middleware"] if isinstance(m, MemoryMiddleware)],
            [],
            "MemoryMiddleware must be omitted when no memory_backend was injected",
        )
        self.assertNotIn("store", call_kwargs)
        # Warning was emitted at construction time.
        self.assertTrue(any("memory_backend" in m for m in captured.output))

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_memory_backend_independent_from_main_and_skills(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """memory_backend is independent from the agent's main filesystem backend
        and from skills_backend — all three backends can be distinct."""
        import tempfile

        main_backend = StateBackend()
        skills_backend = StateBackend()

        class _MainBackendWrapper(BaseDeepAgentBackend):
            def _build_backend(self):
                return main_backend

            async def clear(self):
                pass

        class _SkillsBackendWrapper(BaseDeepAgentBackend):
            def _build_backend(self):
                return skills_backend

            async def clear(self):
                pass

            def routes(self, sources):
                return {f"{s.rstrip('/')}/": skills_backend for s in sources}

        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="Test")]
        mock_get_all_tools.return_value = []
        mock_create_deep_agent.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            memory_backend = MemoryFilesystemDeepAgentBackend(bot_uuid="b", chat_id=1, memory_root=tmp)

            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                backend=_MainBackendWrapper(bot_uuid="b", chat_id=1),
                skills_paths=["/path/to/skills"],
                skills_backend=_SkillsBackendWrapper(bot_uuid="b", chat_id=1),
                memory_backend=memory_backend,
            )
            await agent.initialize_async_resources()

            middleware = mock_create_deep_agent.call_args.kwargs["middleware"]
            fs_mw = next(m for m in middleware if isinstance(m, FilesystemMiddleware))
            skills_mw = next(m for m in middleware if isinstance(m, SkillsMiddleware))
            memory_mw = next(m for m in middleware if isinstance(m, MemoryMiddleware))

            self.assertIs(fs_mw.backend.default, main_backend)
            self.assertIs(skills_mw._backend, skills_backend)
            self.assertIs(memory_mw._backend, memory_backend.backend)
            self.assertIsNot(memory_mw._backend, fs_mw.backend)
            self.assertIsNot(memory_mw._backend, skills_mw._backend)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_omits_memory_when_unset(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """When no memory_backend is injected, no MemoryMiddleware is added and
        no store kwarg is passed — behavior identical to today."""
        bot_config = _make_config()
        mock_messages_storage = MagicMock()
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=10)
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
        mock_get_all_tools.return_value = []
        mock_create_deep_agent.return_value = MagicMock()

        agent = LLMDeepAgent(mock_llm, bot_config, system_instructions, mock_messages_storage)
        await agent.initialize_async_resources()

        call_kwargs = mock_create_deep_agent.call_args.kwargs
        memory_mw = [m for m in call_kwargs["middleware"] if isinstance(m, MemoryMiddleware)]
        self.assertEqual(memory_mw, [], "MemoryMiddleware should not be present")
        self.assertNotIn("store", call_kwargs)
        # No memory (or skills) configured → the main backend is NOT wrapped in
        # a CompositeBackend; no routes are added.
        fs_mw = next(m for m in call_kwargs["middleware"] if isinstance(m, FilesystemMiddleware))
        self.assertNotIsInstance(fs_mw.backend, CompositeBackend)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_deep_agent_memory_add_cache_control_passthrough(
        self, mock_create_deep_agent, mock_create_agent, mock_get_all_tools
    ):
        """memory_add_cache_control is an LLMDeepAgent ctor param passed through
        to the MemoryMiddleware the agent constructs."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bot_config = _make_config()
            mock_messages_storage = MagicMock()
            mock_llm = MagicMock()
            mock_llm.get_num_tokens = MagicMock(return_value=10)
            system_instructions = [SystemMessage(content="You are a helpful assistant")]
            mock_get_all_tools.return_value = []
            mock_create_deep_agent.return_value = MagicMock()

            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                memory_backend=MemoryFilesystemDeepAgentBackend(bot_uuid="b", chat_id=1, memory_root=tmp),
                memory_add_cache_control=True,
            )
            await agent.initialize_async_resources()

            memory_mw = next(
                (m for m in mock_create_deep_agent.call_args.kwargs["middleware"] if isinstance(m, MemoryMiddleware)),
                None,
            )
            self.assertIsNotNone(memory_mw)
            self.assertTrue(memory_mw._add_cache_control)

            # Default is False.
            agent2 = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                memory_backend=MemoryFilesystemDeepAgentBackend(bot_uuid="b", chat_id=1, memory_root=tmp),
            )
            await agent2.initialize_async_resources()
            memory_mw2 = next(
                (m for m in mock_create_deep_agent.call_args.kwargs["middleware"] if isinstance(m, MemoryMiddleware)),
                None,
            )
            self.assertFalse(memory_mw2._add_cache_control)

    async def test_deep_agent_clean_context_clears_memory_backend(self):
        """clean_context clears the per-chat memory backend (mirroring the main
        backend wrapper) so /flushcontext wipes chat memory along with the
        workspace."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bot_config = _make_config()
            mock_messages_storage = MagicMock()
            mock_messages_storage.chat_id = 123
            mock_messages_storage.clear_messages = AsyncMock()
            mock_llm = MagicMock()
            system_instructions = [SystemMessage(content="Test")]

            memory_backend = MemoryFilesystemDeepAgentBackend(bot_uuid="b", chat_id=1, memory_root=tmp)
            agent = LLMDeepAgent(
                mock_llm,
                bot_config,
                system_instructions,
                mock_messages_storage,
                memory_backend=memory_backend,
            )
            agent.documents_storage = None
            self.assertTrue(os.path.exists(memory_backend.source))

            await agent.clean_context()

            self.assertFalse(os.path.exists(memory_backend.source))

    def test_deep_agent_memory_before_agent_loads_real_file(self):
        """REGRESSION: MemoryMiddleware fed the wrapper's backend + absolute
        source path must load the seeded AGENTS.md — a virtual_mode=True
        backend would append the absolute path under its root and silently
        report "(No memory loaded)"."""
        import tempfile

        from deepagents.middleware.memory import MemoryMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            wrapper = MemoryFilesystemDeepAgentBackend(bot_uuid="b", chat_id=1, memory_root=tmp)
            mw = MemoryMiddleware(backend=wrapper.backend, sources=[wrapper.source])
            update = mw.before_agent({}, MagicMock(), MagicMock())
            self.assertIsNotNone(update)
            self.assertIn(wrapper.source, update["memory_contents"])
            self.assertIn("Learnings from this conversation", update["memory_contents"][wrapper.source])


if __name__ == "__main__":
    unittest.main()
