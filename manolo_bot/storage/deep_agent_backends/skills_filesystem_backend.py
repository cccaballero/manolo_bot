"""Filesystem-based backend for the deep agent's SkillsMiddleware.

Skills are operator-provided, versioned, auditable capability bundles described
by ``SKILL.md`` files. They are *global* (shared across all chats and bot
instances in the same process) and must never be cleared by ``clean_context``
— clearing a filesystem-backed skills store would delete operator content
when a user runs ``/flushcontext``.

Unlike :class:`FilesystemDeepAgentBackend`, this class is intentionally
**not** scoped by ``(bot_uuid, chat_id)``. It wraps ``FilesystemBackend``
with ``virtual_mode=False`` so the configured skill source paths
(``DEEP_AGENT_SKILLS_PATHS``) can be at any absolute location on the host
filesystem, not just inside a per-chat workspace subdirectory.

``virtual_mode=False`` is safe here because this instance is handed exclusively
to ``SkillsMiddleware``, which only calls ``ls()`` and ``download_files()``
(read-only) at the configured source paths. No write/edit/delete is ever
invoked through this backend.
"""

from collections.abc import Sequence

from deepagents.backends.filesystem import FilesystemBackend

from manolo_bot.storage.deep_agent_backends.base import BaseSkillsBackend


class SkillsFilesystemDeepAgentBackend(BaseSkillsBackend):
    """Filesystem-backed implementation of :class:`BaseSkillsBackend`.

    One instance is intended to be reused across every agent in the process:
    :mod:`manolo_bot.main` constructs a single instance at module load and
    passes it to every ``LLMDeepAgent`` it creates.

    :param workspace_path: Optional root directory for relative skill source
        paths. Absolute paths in ``DEEP_AGENT_SKILLS_PATHS`` are unaffected.
        Defaults to ``None`` (uses the process current working directory).

    Example:

        .. code-block:: python

            from manolo_bot.storage.deep_agent_backends.skills_filesystem_backend import (
                SkillsFilesystemDeepAgentBackend,
            )

            # Shared — reuse across every chat in the process.
            skills_backend = SkillsFilesystemDeepAgentBackend()
            llm_bot = LLMDeepAgent(
                ...,
                skills_paths=["/etc/manolo_bot/skills", ("$HOME/.local/share/manolo_bot/skills", "User")],
                skills_backend=skills_backend,
            )
    """

    def __init__(self, workspace_path: str | None = None) -> None:
        # virtual_mode=False so absolute skill source paths resolve directly
        # on the host filesystem, regardless of `workspace_path`.
        self._backend = FilesystemBackend(root_dir=workspace_path, virtual_mode=False)

    @property
    def backend(self) -> FilesystemBackend:
        """The underlying :class:`deepagents.backends.filesystem.FilesystemBackend` instance."""
        return self._backend

    def routes(self, sources: Sequence[str | tuple[str, str]]) -> dict[str, FilesystemBackend]:
        """Build CompositeBackend routes for the given skill source paths.

        Each source (bare path or ``(path, label)`` tuple) becomes a route
        prefix ``<path>/`` mapped to a sandboxed :class:`FilesystemBackend`
        rooted at that source with ``virtual_mode=True``. ``CompositeBackend``
        strips the prefix before delegating, so the agent's runtime
        ``read_file`` / ``ls`` / ``glob`` / ``grep`` tools reach skill files at
        their absolute paths — the per-chat backend's ``virtual_mode=True``
        root would otherwise reject any path outside the chat workspace.
        Repeated prefixes are deduped.
        """
        routes: dict[str, FilesystemBackend] = {}
        for source in sources:
            path = source[0] if isinstance(source, tuple) else source
            route_prefix = path.rstrip("/") + "/"
            if route_prefix in routes:
                continue
            routes[route_prefix] = FilesystemBackend(
                root_dir=path.rstrip("/"),
                virtual_mode=True,
            )
        return routes

    async def clear(self) -> None:
        """No-op.

        Skills are operator-provided, not per-chat state. They are intentionally
        not cleared by ``LLMDeepAgent.clean_context()`` — wiping operator content
        when a user runs ``/flushcontext`` would be destructive and surprising.
        If a caller really wants to drop skills, they can replace the singleton
        with a fresh instance.
        """
        return None
