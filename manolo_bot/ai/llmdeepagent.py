import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.ai.tools import get_all_tools
from manolo_bot.storage.deep_agent_backends.base import (
    BaseDeepAgentBackend,
    BaseMemoryBackend,
    BaseSkillsBackend,
)
from manolo_bot.storage.documents.base import BaseDocumentStorage
from manolo_bot.storage.messages.base import BaseMessagesStorage

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)


def _resolve_skills_sources(
    raw_paths: Sequence[str | tuple[str, str]] | None,
) -> list[str | tuple[str, str]]:
    """
    Resolve a list of raw skill source paths into the shape deepagents expects.

    Each entry is either a bare path (``str``), a ``(path, label)`` tuple, or
    a ``<path>::LABEL=<text>`` string which is split into a tuple. Tuples are
    passed through unchanged. The ``::LABEL=`` syntax is a convenience for
    env-var-driven configuration; deepagents itself only accepts bare
    paths or ``(path, label)`` tuples.
    """
    if not raw_paths:
        return []
    resolved: list[str | tuple[str, str]] = []
    for entry in raw_paths:
        if not entry:
            continue
        if isinstance(entry, tuple):
            resolved.append(entry)
            continue
        if "::LABEL=" in entry:
            path, _, label = entry.partition("::LABEL=")
            if label.strip():
                resolved.append((path, label.strip()))
                continue
        resolved.append(entry)
    return resolved


class LLMDeepAgent(LLMAgent):
    """
    Advanced Telegram LLM Chat Bot using LangChain Deep Agents harness.

    Extends LLMAgent with the full deep agents stack:
    - To-do list planning (TodoListMiddleware)
    - Virtual filesystem (FilesystemMiddleware + StateBackend)
    - Sub-agents (via create_deep_agent subagents parameter)
    - Skills (progressive disclosure via SkillsMiddleware)
    - Long-term memory (AGENTS.md files loaded into the system prompt via MemoryMiddleware)

    The deep agent harness has its own internal system prompt (planning,
    filesystem, sub-agent instructions). The bot's character/persona
    instructions from :attr:`system_instructions` are passed as the
    ``system_prompt`` string so they are merged with the harness prompt:
    ``bot_instructions + "\\n\\n" + harness_base_prompt``.

    The virtual filesystem backend is injected from outside
    (see :meth:`main.instance_llm_bot`), keeping the class decoupled
    from chat-specific path logic. Skills follow the same explicit
    injection pattern: ``LLMDeepAgent`` does **not** instantiate a
    skills backend itself. Pass ``skills_paths`` together with an
    explicit ``skills_backend`` wrapper (typically a
    :class:`SkillsFilesystemDeepAgentBackend` constructed once at
    module load and shared across every chat in the process). If
    ``skills_backend`` is ``None`` the ``SkillsMiddleware`` is omitted
    entirely, even when ``skills_paths`` is set.

    Skills are operator-provided, not per-chat state, and are
    intentionally **not** cleared by :meth:`clean_context`.

    When ``skills_paths`` or ``memory_paths`` is configured, the agent's
    main filesystem backend is wrapped with a :class:`CompositeBackend`
    whose routes come from the injected wrappers' ``routes()`` factories
    — each wrapper owns the storage-specific path→backend mapping
    (filesystem wrappers return sandboxed :class:`FilesystemBackend`
    routes rooted at each source; Redis/DB wrappers may map prefixes to
    ``StoreBackend``-style backends or return ``{}``). This lets the
    agent's runtime ``read_file`` / ``write_file`` / ``edit_file`` /
    ``ls`` tools reach those files at their absolute paths — the
    per-chat backend's ``virtual_mode=True`` root would otherwise reject
    any path outside the chat workspace. For skills this is read-only
    access to capability bundles; for memory it also enables durable
    write-back: learnings the agent persists via ``edit_file`` land on
    the real memory file on disk.

    Long-term memory follows the same explicit-injection pattern as
    skills: ``LLMDeepAgent`` does **not** instantiate a memory backend
    itself. Pass an explicit ``memory_backend`` wrapper (typically a
    per-chat :class:`MemoryFilesystemDeepAgentBackend` constructed
    alongside the main backend in :meth:`main.instance_llm_bot`). If
    ``memory_backend`` is ``None`` the ``MemoryMiddleware`` is omitted
    entirely. The injected wrapper supplies the backend used to read the
    memory file, the chat-scoped source path (derived automatically when
    ``memory_paths`` is not given), and the langgraph store forwarded to
    ``create_deep_agent(store=...)``; the ``MemoryMiddleware`` itself is
    constructed by the agent, exactly like ``SkillsMiddleware``.
    ``memory_add_cache_control`` is passed through to
    ``MemoryMiddleware`` (adds an Anthropic prompt-cache breakpoint on
    the memory block; no-op on other models).

    Memory is per-chat state: each chat gets its own independent, seeded
    ``AGENTS.md`` file (no cross-chat leakage), and :meth:`clean_context`
    clears it along with the chat workspace — ``/flushcontext`` wipes
    both. The agent writes new knowledge back to memory via ``edit_file``.

    Memory files are fully loaded into the system prompt on every turn,
    so keep them concise — unlike skills, they cost tokens on every
    message.
    """

    bind_tools_on_init = False

    def __init__(
        self,
        llm: BaseChatModel,
        bot_config: BotConfig,
        system_instructions: list[BaseMessage],
        messages_storage: BaseMessagesStorage,
        tools: list[BaseTool] | None = None,
        documents_storage: BaseDocumentStorage | None = None,
        system_instructions_mapping=None,
        backend: BaseDeepAgentBackend | None = None,
        skills_paths: Sequence[str | tuple[str, str]] | None = None,
        skills_backend: BaseSkillsBackend | None = None,
        memory_paths: Sequence[str] | None = None,
        memory_backend: BaseMemoryBackend | None = None,
        memory_add_cache_control: bool = False,
    ) -> None:
        super().__init__(
            llm,
            bot_config,
            system_instructions,
            messages_storage,
            tools=tools,
            documents_storage=documents_storage,
            system_instructions_mapping=system_instructions_mapping,
        )
        self._backend_wrapper = backend
        self._backend: BackendProtocol = backend.backend if backend else StateBackend()
        self.agent = None
        # Skills backend is **explicitly injected** by the caller — same shape
        # as ``backend=``. If absent, SkillsMiddleware is omitted entirely
        # (no implicit default, no singleton). The ABC
        # ``BaseSkillsBackend`` enforces the ``.backend``, ``.routes()`` and
        # ``.clear()`` contract at type-check / instantiation time, mirroring
        # how ``BaseDeepAgentBackend`` does it for the main backend.
        self._skills_backend_wrapper: BaseSkillsBackend | None = skills_backend
        self._skills_backend: BackendProtocol | None = skills_backend.backend if skills_backend is not None else None
        # Resolve skill sources eagerly (pure parsing, no I/O): explicit ctor
        # argument only — configuration flows in through the caller, which
        # reads it from ``Config`` and passes it here (same as memory).
        if skills_paths is not None:
            self._resolved_skills_sources: list[str | tuple[str, str]] = _resolve_skills_sources(skills_paths)
        else:
            self._resolved_skills_sources = []
        if self._resolved_skills_sources and self._skills_backend is None:
            # Surfacing this loudly avoids the silent-no-skills failure mode of
            # earlier drafts where sources were configured but no backend was
            # wired, so the bot saw no skill metadata at all.
            logger.warning(
                "skills_paths is configured (%d source(s)) but no skills_backend "
                "was passed; SkillsMiddleware will be omitted and skills will "
                "not be loaded. Pass skills_backend=SkillsFilesystemDeepAgentBackend(...) "
                "to enable skills.",
                len(self._resolved_skills_sources),
            )

        # Memory is **explicitly injected** by the caller — same shape as
        # ``backend=`` and ``skills_backend=``. If absent, MemoryMiddleware is
        # omitted entirely (no implicit default, no singleton). The ABC
        # ``BaseMemoryBackend`` enforces the ``.backend``, ``.routes()`` and
        # ``.clear()`` contract at type-check / instantiation time, mirroring
        # how ``BaseSkillsBackend`` does it for skills.
        self._memory_backend: BaseMemoryBackend | None = memory_backend
        self._memory_add_cache_control = memory_add_cache_control
        # Memory sources resolve in order: explicit ctor argument (advanced use),
        # then — for per-chat memory — the injected wrapper's own chat-scoped
        # source path (the wrapper owns the per-chat AGENTS.md).
        if memory_paths is not None:
            self._resolved_memory_paths: list[str] = list(memory_paths)
        elif memory_backend is not None:
            # Per-chat memory: the wrapper owns the chat-scoped AGENTS.md path.
            self._resolved_memory_paths = [memory_backend.source]  # type: ignore[attr-defined]
        else:
            self._resolved_memory_paths = []
        if self._resolved_memory_paths and self._memory_backend is None:
            # Surfacing this loudly avoids the silent-no-memory failure mode:
            # sources configured but no backend wired, so the agent sees no
            # memory content at all.
            logger.warning(
                "memory_paths is configured (%d source(s)) but no memory_backend "
                "was passed; MemoryMiddleware will be omitted and memory will "
                "not be loaded. Pass memory_backend=MemoryFilesystemDeepAgentBackend(...) "
                "to enable memory.",
                len(self._resolved_memory_paths),
            )

        # Wrap the main backend with routes provided by the injected skill and
        # memory backend wrappers so the agent's FilesystemMiddleware
        # (read_file, write_file, edit_file, ls, glob, grep) can reach files at
        # the configured paths. Without this, the per-chat backend's
        # virtual_mode=True root rejects any path outside the chat workspace:
        # read_file calls to skill or memory paths return "file not found", and
        # write_file/edit_file silently land in in-memory state instead of the
        # real file on disk. The path→backend mapping is storage-specific
        # knowledge owned by each wrapper's ``routes()`` factory — the agent
        # only merges the route dicts.
        routes: dict[str, BackendProtocol] = {}
        if self._resolved_skills_sources and self._skills_backend_wrapper is not None:
            for route_prefix, route_backend in self._skills_backend_wrapper.routes(
                self._resolved_skills_sources
            ).items():
                if route_prefix in routes:
                    logger.debug("CompositeBackend route %s already present; reusing it", route_prefix)
                    continue
                routes[route_prefix] = route_backend
        if self._resolved_memory_paths and self._memory_backend is not None:
            for route_prefix, route_backend in self._memory_backend.routes(self._resolved_memory_paths).items():
                if route_prefix in routes:
                    logger.debug("CompositeBackend route %s already present; reusing it", route_prefix)
                    continue
                routes[route_prefix] = route_backend
        if routes:
            self._backend = CompositeBackend(default=self._backend, routes=routes)

    async def initialize_async_resources(self) -> None:
        """Initialize async resources and create deep agent with all tools."""
        await super().initialize_async_resources()

        tools = await get_all_tools(
            self._mcp_manager, self.bot_config, document_storage=self.documents_storage, custom_tools=self.tools
        )

        instructions_text = self._system_instructions[0].content if self._system_instructions else ""

        middleware: list[Any] = [
            FilesystemMiddleware(backend=self._backend),
            TodoListMiddleware(),
        ]
        # SkillsMiddleware is only included when both skill sources and an
        # explicit skills backend wrapper are present. The wrapper is constructed
        # by the caller (typically main.py instantiates one
        # SkillsFilesystemDeepAgentBackend and shares it across every chat).
        # Constructing SkillsMiddleware explicitly (instead of passing
        # `skills=[...]` to create_deep_agent) keeps the skills backend
        # independent of the agent's main filesystem backend.
        if self._resolved_skills_sources and self._skills_backend is not None:
            middleware.append(SkillsMiddleware(backend=self._skills_backend, sources=self._resolved_skills_sources))
        # MemoryMiddleware is only included when both memory sources and an
        # explicit memory backend wrapper are present. The wrapper is
        # constructed by the caller (typically main.py instantiates one
        # MemoryFilesystemDeepAgentBackend and shares it across every chat).
        # Constructing MemoryMiddleware explicitly (mirroring SkillsMiddleware)
        # keeps the memory backend independent of the agent's main filesystem
        # backend; the injected wrapper supplies the backend and the langgraph
        # store.
        if self._resolved_memory_paths and self._memory_backend is not None:
            middleware.append(
                MemoryMiddleware(
                    backend=self._memory_backend.backend,
                    sources=self._resolved_memory_paths,
                    add_cache_control=self._memory_add_cache_control,
                )
            )

        create_kwargs: dict[str, Any] = {
            "model": self.llm,
            "tools": tools,
            "system_prompt": instructions_text,
            "middleware": middleware,
            "subagents": [],
        }
        if self._resolved_memory_paths and self._memory_backend is not None:
            # The langgraph store (persistent agent state) is owned by the
            # memory backend wrapper (a storage concern — skills had no store).
            create_kwargs["store"] = self._memory_backend.store  # type: ignore[attr-defined]
        self.agent = create_deep_agent(**create_kwargs)
        logging.debug(f"Deep agent created with {len(tools)} tools")

    def _base_messages(self) -> list[BaseMessage]:
        """Only keep the AIMessage priming, skipping the SystemMessage (already in system_prompt)."""
        if self.system_instructions and len(self.system_instructions) > 1:
            return [self.system_instructions[1]]
        return []

    async def clean_context(self) -> None:
        await super().clean_context()
        # Skills are operator-provided, not per-chat state — do not clear the
        # skills backend here. The main workspace backend and the per-chat
        # memory backend ARE chat state: /flushcontext wipes both, using the
        # same guarded-wrapper mechanism (the memory backend's clear() deletes
        # the chat's memory directory).
        if self._backend_wrapper:
            await self._backend_wrapper.clear()
        if self._memory_backend:
            await self._memory_backend.clear()
