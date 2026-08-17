import os
import re
import shutil

from deepagents.backends.filesystem import FilesystemBackend

from manolo_bot.storage.deep_agent_backends.base import BaseDeepAgentBackend

# Path-traversal characters are forbidden in bot_uuid. The character class is
# chosen to match realistic identifiers (alphanumeric, dash, underscore) while
# excluding ``/``, ``.``, and everything else that could escape the chat
# directory under ``workspace_path``. ``bot_uuid`` is operator-provided via
# ``BOT_UUID`` — a typo or attacker-controlled env var must not be able to
# trick ``/flushcontext`` (which calls ``clear()`` → ``shutil.rmtree``) into
# deleting directories outside the configured workspace.
_BOT_UUID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class FilesystemDeepAgentBackend(BaseDeepAgentBackend):
    """
    Filesystem-based implementation of the deep agent filesystem backend.

    Builds a per-chat filesystem path from ``bot_uuid`` and ``chat_id``,
    mirroring the pattern used by :class:`FileDocumentsStorage`.

    Security model — defense in depth, two layers:

    * **Layer 1 — input validation at construction time** (fail-fast).
      ``bot_uuid`` is required to match ``[A-Za-z0-9_-]+`` (refuses ``/``,
      ``..``, control characters, etc.). ``workspace_path`` is required to
      be an absolute path. A misconfigured operator gets a ``ValueError``
      at startup, before any agent can run.
    * **Layer 2 — containment check at use time** (defense in depth).
      ``_build_backend()`` and ``clear()`` resolve the constructed
      ``chat_path`` and verify it is still under ``workspace_path``. This
      catches symlink-based escapes that pass Layer 1 (the workspace root
      itself contains a symlink pointing outside) and runtime mutations
      of ``_workspace_path``.

    Note: ``workspace_path`` should be a directory only the bot process
    can read. The default (``/tmp/manolo_bot/workspace``) lives under the
    system temp directory, which on Linux is world-readable; on shared
    hosts set ``DEEP_AGENT_WORKSPACE_PATH`` to a private path
    (e.g. ``~/.local/share/manolo_bot/workspace`` with mode ``0700``).
    """

    def __init__(self, bot_uuid: str, chat_id: int, workspace_path: str, virtual_mode: bool = True) -> None:
        # Layer 1: input validation.
        if not isinstance(bot_uuid, str) or not _BOT_UUID_PATTERN.fullmatch(bot_uuid):
            raise ValueError(
                f"Invalid bot_uuid {bot_uuid!r}: must match [A-Za-z0-9_-]+ "
                "(refuses '/', '..', and other path-traversal characters)."
            )
        if not isinstance(workspace_path, str) or not workspace_path or not os.path.isabs(workspace_path):
            raise ValueError(f"workspace_path must be a non-empty absolute path; got {workspace_path!r}.")
        self._workspace_path = workspace_path
        self._virtual_mode = virtual_mode
        super().__init__(bot_uuid, chat_id)

    def _safe_chat_path(self) -> str:
        """Compute the chat path and verify it resolves under ``workspace_path``.

        Returns the resolved path. Raises ``ValueError`` if the chat path
        escapes the workspace — caller should treat this as a hard error
        and not attempt ``shutil.rmtree`` or any other filesystem op.

        This is Layer 2 of the security model. ``os.path.realpath`` follows
        symlinks, so a workspace whose root itself contains a symlink
        pointing outside is caught here even though Layer 1 (which only
        inspects ``bot_uuid``) would have let it through.
        """
        chat_path = os.path.join(self._workspace_path, self.bot_uuid, str(self.chat_id))
        resolved_workspace = os.path.realpath(self._workspace_path)
        resolved_chat = os.path.realpath(chat_path)
        try:
            common = os.path.commonpath([resolved_workspace, resolved_chat])
        except ValueError as exc:
            raise ValueError(
                f"chat_path {chat_path!r} not under workspace_path {self._workspace_path!r}: {exc}"
            ) from exc
        if common != resolved_workspace:
            raise ValueError(
                f"Refusing to operate on chat_path {chat_path!r}: resolved "
                f"outside workspace_path {self._workspace_path!r}."
            )
        return resolved_chat

    def _build_backend(self) -> FilesystemBackend:
        chat_path = self._safe_chat_path()
        return FilesystemBackend(root_dir=chat_path, virtual_mode=self._virtual_mode)

    @property
    def backend(self) -> FilesystemBackend:
        return self._backend

    async def clear(self) -> None:
        # Layer 2 catches a tampered ``_workspace_path`` or a workspace
        # whose realpath resolves outside the operator-intended root.
        chat_path = self._safe_chat_path()
        if os.path.exists(chat_path):
            shutil.rmtree(chat_path)
