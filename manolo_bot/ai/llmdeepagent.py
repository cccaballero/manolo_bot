import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware, SkillsMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.ai.tools import get_all_tools
from manolo_bot.storage.deep_agent_backends.base import (
    BaseDeepAgentBackend,
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

    When ``skills_paths`` is configured, the agent's main filesystem
    backend is wrapped with a :class:`CompositeBackend` that routes
    each configured skill source to its own sandboxed
    :class:`FilesystemBackend` rooted at that source. This lets the
    agent's runtime ``read_file`` / ``ls`` tools access skill files at
    their absolute paths — the per-chat backend's ``virtual_mode=True``
    root would otherwise reject any path outside the chat workspace.
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
        # ``BaseSkillsBackend`` enforces the ``.backend`` and ``.clear()``
        # contract at type-check / instantiation time, mirroring how
        # ``BaseDeepAgentBackend`` does it for the main backend.
        self._skills_backend: BackendProtocol | None = skills_backend.backend if skills_backend is not None else None
        # Resolve skill sources eagerly (pure parsing, no I/O): prefer the
        # explicit ctor argument, fall back to bot_config, else empty.
        if skills_paths is not None:
            self._resolved_skills_sources: list[str | tuple[str, str]] = _resolve_skills_sources(skills_paths)
        elif bot_config.deep_agent_skills_paths:
            self._resolved_skills_sources = _resolve_skills_sources(bot_config.deep_agent_skills_paths)
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

        # Wrap the main backend with skill-source routes so the agent's
        # FilesystemMiddleware (read_file, ls, glob, grep) can access skill files
        # at the configured paths. Without this, the per-chat backend's
        # virtual_mode=True root rejects any path outside the chat workspace and
        # the bot's read_file calls to skill paths return "file not found".
        # Each route gets its own FilesystemBackend rooted at the source with
        # virtual_mode=True so CompositeBackend's prefix-stripping lands the
        # resolved path back inside the source directory.
        if self._resolved_skills_sources:
            skill_routes: dict[str, BackendProtocol] = {}
            for source in self._resolved_skills_sources:
                skill_path = source[0] if isinstance(source, tuple) else source
                route_prefix = skill_path.rstrip("/") + "/"
                skill_routes[route_prefix] = FilesystemBackend(
                    root_dir=skill_path.rstrip("/"),
                    virtual_mode=True,
                )
            self._backend = CompositeBackend(default=self._backend, routes=skill_routes)

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

        create_kwargs: dict[str, Any] = {
            "model": self.llm,
            "tools": tools,
            "system_prompt": instructions_text,
            "middleware": middleware,
            "subagents": [],
        }
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
        # skills backend here. Clearing a FilesystemDeepAgentBackend used for
        # skills would delete the skills directory; clearing a MemoryDeepAgentBackend
        # would orphan the in-memory skill cache from re-discovery. SkillsMiddleware
        # owns its own per-session lifecycle via the `before_agent` hook.
        if self._backend_wrapper:
            await self._backend_wrapper.clear()
