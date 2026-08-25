"""Filesystem-based backend for the deep agent's MemoryMiddleware.

Memory is per-chat, durable long-term knowledge (a single ``AGENTS.md`` file)
loaded into the agent's system prompt via ``MemoryMiddleware``. Each chat gets
its own independent memory directory under ``memory_root`` — no information
leaks between chats. The agent writes new knowledge back to memory via
``edit_file``, so the file persists across sessions.

Unlike the skills backend (global), this class is scoped by
``(bot_uuid, chat_id)`` exactly like :class:`FilesystemDeepAgentBackend`.

The wrapper exposes **two distinct** :class:`FilesystemBackend` instances
because no single ``virtual_mode`` serves both consumers:

* ``backend`` (``virtual_mode=False``) is handed to ``MemoryMiddleware``,
  whose sources are RAW ABSOLUTE paths (``<chat_memory_dir>/AGENTS.md``).
  With ``virtual_mode=True`` those absolute paths would be appended under
  the root (``<chat_memory_dir>/<chat_memory_dir>/AGENTS.md``) and silently
  not found — hence the middleware-facing backend must be
  ``virtual_mode=False`` so absolute paths pass through unchanged.
  Containment is guaranteed by construction (the root is derived from the
  validated ``bot_uuid``/``chat_id`` via the Layer-1/Layer-2 security model)
  and by the fact this backend is handed ONLY to ``MemoryMiddleware``, which
  accesses exactly the one chat-scoped source the wrapper derives.
* ``routes()`` returns a separate ``virtual_mode=True`` instance for the
  agent's main ``CompositeBackend``: ``_route_for_path`` strips the route
  prefix and hands the target a key WITH a leading slash (``"/AGENTS.md"``),
  which ``virtual_mode=True`` correctly resolves under the chat root — a
  ``virtual_mode=False`` target would resolve it against the filesystem root
  (wrong and unsafe). This mirrors how the skills wrapper separates its
  middleware-facing backend from its ``routes()`` targets.

On construction the chat memory directory is created (if missing) and seeded
with a minimal NON-EMPTY ``AGENTS.md`` template. The template must be plain
markdown text — deepagents' ``MemoryMiddleware._format_agent_memory`` skips
empty sources (HTML comments are stripped, so a comment-only file counts as
empty) and would render "(No memory loaded)" while hiding the source path
from the agent.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence

from deepagents.backends.filesystem import FilesystemBackend
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from manolo_bot.storage.deep_agent_backends.base import _BOT_UUID_PATTERN, BaseMemoryBackend

# Seeded into every chat's memory file. Must be NON-EMPTY plain markdown:
# deepagents' MemoryMiddleware._format_agent_memory skips empty sources and
# strips HTML comments, so a comment-only file would render "(No memory
# loaded)" and hide the source path from the agent.
MEMORY_TEMPLATE = "# Memory\n\nLearnings from this conversation are recorded here.\n"


class MemoryFilesystemDeepAgentBackend(BaseMemoryBackend):
    """Filesystem-backed, per-chat implementation of :class:`BaseMemoryBackend`.

    One instance is constructed per chat (typically inside
    :meth:`main.instance_llm_bot`, alongside the main
    :class:`FilesystemDeepAgentBackend`) and passed to that chat's
    ``LLMDeepAgent``. Memory is scoped by ``(bot_uuid, chat_id)`` so chats
    never share or leak memory.

    Security model — defense in depth, two layers (mirrors
    :class:`FilesystemDeepAgentBackend`):

    * **Layer 1 — input validation at construction time** (fail-fast).
      ``bot_uuid`` must match ``[A-Za-z0-9_-]+``; ``memory_root`` must be a
      non-empty absolute path. A misconfigured operator gets a ``ValueError``
      at startup, before any agent can run.
    * **Layer 2 — containment check at use time** (defense in depth).
      ``_safe_memory_path()`` resolves the chat memory dir and verifies it is
      still under ``memory_root``, catching symlink-based escapes that pass
      Layer 1.

    :param bot_uuid: Bot identifier (validated against ``[A-Za-z0-9_-]+``).
    :param chat_id: Chat identifier; scopes the memory directory.
    :param memory_root: Absolute root directory under which each chat's
        memory lives at ``memory_root/bot_uuid/chat_id``.
    :param store: Optional langgraph :class:`BaseStore` forwarded to
        ``create_deep_agent(store=...)``. Defaults to a process-local
        :class:`InMemoryStore` created at construction time — the default
        lives here, in the storage module, never inside ``LLMDeepAgent``.

    Example:

        .. code-block:: python

            from manolo_bot.storage.deep_agent_backends.memory_filesystem_backend import (
                MemoryFilesystemDeepAgentBackend,
            )

            # One instance per chat — pass it to that chat's agent.
            memory_backend = MemoryFilesystemDeepAgentBackend(
                bot_uuid="bot-1", chat_id=1001, memory_root="/var/lib/manolo_bot/memory"
            )
            llm_bot = LLMDeepAgent(
                ...,
                memory_backend=memory_backend,
            )
    """

    def __init__(self, bot_uuid: str, chat_id: int, memory_root: str, store: BaseStore | None = None) -> None:
        # Layer 1: input validation (mirrors FilesystemDeepAgentBackend).
        if not isinstance(bot_uuid, str) or not _BOT_UUID_PATTERN.fullmatch(bot_uuid):
            raise ValueError(
                f"Invalid bot_uuid {bot_uuid!r}: must match [A-Za-z0-9_-]+ "
                "(refuses '/', '..', and other path-traversal characters)."
            )
        if not isinstance(memory_root, str) or not memory_root or not os.path.isabs(memory_root):
            raise ValueError(f"memory_root must be a non-empty absolute path; got {memory_root!r}.")
        self.bot_uuid = bot_uuid
        self.chat_id = chat_id
        self._memory_root = memory_root
        # The default store lives HERE (storage module), not in LLMDeepAgent —
        # the agent never constructs implicit defaults.
        self._store = store if store is not None else InMemoryStore()
        # Layer 2 containment check happens inside _safe_memory_path().
        memory_path = self._safe_memory_path()
        # Middleware-facing backend: virtual_mode=False so MemoryMiddleware's
        # raw ABSOLUTE source paths pass through unchanged (virtual_mode=True
        # would append them under the root and silently miss the file). Never
        # exposed to agent tools; only MemoryMiddleware reads through it.
        self._backend = FilesystemBackend(root_dir=memory_path, virtual_mode=False)
        # Route target for the agent's main CompositeBackend: a SEPARATE
        # instance with virtual_mode=True so the leading-slash stripped key
        # ("/AGENTS.md") resolves under the chat root — CompositeBackend
        # always strips the route prefix before delegating.
        self._route_backend = FilesystemBackend(root_dir=memory_path, virtual_mode=True)
        self._seed_memory()

    def _safe_memory_path(self) -> str:
        """Compute the chat memory dir and verify it resolves under ``memory_root``.

        Returns the resolved path. Raises ``ValueError`` if the chat memory
        dir escapes the root — caller should treat this as a hard error and
        not attempt ``shutil.rmtree`` or any other filesystem op.

        This is Layer 2 of the security model. ``os.path.realpath`` follows
        symlinks, so a root that itself contains a symlink pointing outside is
        caught here even though Layer 1 (which only inspects ``bot_uuid``)
        would have let it through.
        """
        memory_path = os.path.join(self._memory_root, self.bot_uuid, str(self.chat_id))
        resolved_root = os.path.realpath(self._memory_root)
        resolved_memory = os.path.realpath(memory_path)
        try:
            common = os.path.commonpath([resolved_root, resolved_memory])
        except ValueError as exc:
            raise ValueError(f"memory_path {memory_path!r} not under memory_root {self._memory_root!r}: {exc}") from exc
        if common != resolved_root:
            raise ValueError(
                f"Refusing to operate on memory_path {memory_path!r}: resolved "
                f"outside memory_root {self._memory_root!r}."
            )
        return resolved_memory

    def _seed_memory(self) -> None:
        """Create the chat memory dir (if missing) and seed a non-empty AGENTS.md.

        deepagents' ``MemoryMiddleware._format_agent_memory`` skips empty
        sources (HTML comments are stripped, so a comment-only file counts as
        empty) and would render "(No memory loaded)" while hiding the source
        path from the agent — the template is therefore plain markdown text.
        Creation races are tolerated: if the file already exists (another
        instance won the race), it is left untouched.
        """
        memory_path = self._safe_memory_path()
        os.makedirs(memory_path, exist_ok=True)
        agents_file = os.path.join(memory_path, "AGENTS.md")
        if not os.path.exists(agents_file):
            try:
                with open(agents_file, "w", encoding="utf-8") as f:
                    f.write(MEMORY_TEMPLATE)
            except OSError:
                # Best-effort seeding; MemoryMiddleware tolerates a missing file.
                pass

    @property
    def backend(self) -> FilesystemBackend:
        """The middleware-facing :class:`FilesystemBackend` (``virtual_mode=False``).

        Handed to ``MemoryMiddleware``, whose sources are raw absolute paths
        that must pass through unchanged. Never exposed to agent tools; see
        the module docstring for the two-instance rationale.
        """
        return self._backend

    @property
    def source(self) -> str:
        """The chat-scoped memory file path loaded by ``MemoryMiddleware``."""
        return os.path.join(self._safe_memory_path(), "AGENTS.md")

    @property
    def store(self) -> BaseStore:
        """The langgraph :class:`BaseStore` forwarded to ``create_deep_agent(store=...)``.

        This property is the one addition over :class:`BaseSkillsBackend`:
        ``MemoryMiddleware`` / ``create_deep_agent`` need a langgraph store
        for persistent agent state, and skills did not — so the store is a
        storage concern owned by this wrapper, not by ``LLMDeepAgent``.
        """
        return self._store

    def routes(self, sources: Sequence[str]) -> dict[str, FilesystemBackend]:
        """Build CompositeBackend routes for the chat memory file.

        The chat memory directory becomes a single route prefix ``<dir>/``
        mapped to a SEPARATE :class:`FilesystemBackend` with
        ``virtual_mode=True`` (distinct from the middleware-facing
        ``backend``): ``CompositeBackend`` strips the route prefix and hands
        the target a leading-slash key (``"/AGENTS.md"``), which
        ``virtual_mode=True`` resolves under the chat root — a
        ``virtual_mode=False`` target would resolve it against the filesystem
        root (wrong and unsafe). This is what lets the agent's runtime
        ``read_file`` / ``write_file`` / ``edit_file`` tools reach the real
        seeded ``AGENTS.md`` on disk for durable write-back. The ``sources``
        argument is accepted for ABC compatibility; the route is always the
        chat's own memory directory.
        """
        memory_dir = self._safe_memory_path()
        route_prefix = memory_dir.rstrip("/") + "/"
        return {route_prefix: self._route_backend}

    async def clear(self) -> None:
        """Delete the chat's memory directory (destructive).

        Memory is per-chat state — ``/flushcontext`` wipes the chat's memory
        along with its workspace. Layer 2 containment check runs first so a
        tampered ``_memory_root`` or a symlink escape can never delete
        directories outside the configured root.
        """
        memory_path = self._safe_memory_path()
        if os.path.exists(memory_path):
            shutil.rmtree(memory_path)
