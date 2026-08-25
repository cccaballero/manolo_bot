import abc
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

# Path-traversal characters are forbidden in bot_uuid. The character class is
# chosen to match realistic identifiers (alphanumeric, dash, underscore) while
# excluding ``/``, ``.``, and everything else that could escape the chat
# directory under the configured root. ``bot_uuid`` is operator-provided via
# ``BOT_UUID`` — a typo or attacker-controlled env var must not be able to
# trick ``/flushcontext`` (which calls ``clear()`` → ``shutil.rmtree``) into
# deleting directories outside the configured root.
_BOT_UUID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class BaseDeepAgentBackend(abc.ABC):
    """
    Abstract base class for deep agent filesystem backends.

    Handles the bot_uuid and chat_id scoping, delegating file operations
    to the underlying deepagents backend (StateBackend, FilesystemBackend, etc.).
    """

    def __init__(self, bot_uuid: str, chat_id: int) -> None:
        self.bot_uuid = bot_uuid
        self.chat_id = chat_id
        self._backend = self._build_backend()

    @abc.abstractmethod
    def _build_backend(self) -> "BackendProtocol":
        """
        Build the underlying deepagents backend instance.

        :return: A BackendProtocol-compatible backend instance.
        """
        pass

    @property
    def backend(self) -> "BackendProtocol":
        """
        Returns the underlying deepagents backend instance.

        :return: The BackendProtocol instance used for file operations.
        """
        return self._backend

    @abc.abstractmethod
    async def clear(self) -> None:
        """
        Clears the backend state for the current chat.

        For memory backends, this removes the cached StateBackend instance.
        For filesystem backends, this removes the chat directory.
        """
        pass


class BaseSkillsBackend(abc.ABC):
    """Abstract base class for deep-agent skills backends.

    Skills are operator-provided, versioned, auditable capability bundles
    described by ``SKILL.md`` files. They are *global* (shared across all
    chats and bot instances in the same process) and must never be cleared
    by ``LLMDeepAgent.clean_context()``.

    This class is intentionally **not** scoped by ``(bot_uuid, chat_id)`` —
    it parallels :class:`BaseDeepAgentBackend` but for the *global* (not
    per-chat) namespace. Subclasses provide a single backend instance that
    every chat shares.

    Library users may subclass this to provide custom storage (Redis, S3,
    in-memory, etc.) — same extension model as :class:`BaseDeepAgentBackend`.

    .. note::
       ``routes()`` is a breaking addition to this ABC vs. the previously
       merged skills shape (issue #70) — acceptable, this is a
       project-internal library.
    """

    @property
    @abc.abstractmethod
    def backend(self) -> "BackendProtocol":
        """The underlying :class:`BackendProtocol` instance used by ``SkillsMiddleware``."""
        ...

    @abc.abstractmethod
    def routes(self, sources: Sequence[str | tuple[str, str]]) -> dict[str, "BackendProtocol"]:
        """Return CompositeBackend route entries (prefix → BackendProtocol) so the
        agent's runtime tools can reach skill content.

        Keys are virtual path prefixes ending in ``"/"``; values are backends
        rooted at the routed location — ``CompositeBackend`` strips the route
        prefix before delegating, so each value must resolve the stripped path
        inside its own root. Non-filesystem implementations may map prefixes to
        ``StoreBackend``-style backends, or return ``{}`` if skill content is
        served exclusively through ``SkillsMiddleware``.
        """
        ...

    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear the backend state.

        Skills are operator-provided, not per-chat state. Implementations
        should typically be a no-op — wiping operator content when a user
        runs ``/flushcontext`` would be destructive and surprising.
        """
        ...


class BaseMemoryBackend(abc.ABC):
    """Abstract base class for deep-agent memory backends.

    Memory is per-chat, durable long-term knowledge (a single ``AGENTS.md``
    file per chat) loaded into the agent's system prompt via
    ``MemoryMiddleware``. It is scoped by ``(bot_uuid, chat_id)`` exactly
    like :class:`BaseDeepAgentBackend` — each chat gets its own independent
    memory so no information leaks between chats. The agent writes new
    knowledge back to memory via ``edit_file``, so the file persists across
    sessions.

    Subclasses provide the per-chat backend instance; the constructor
    contract is ``(bot_uuid, chat_id, ...)`` like :class:`BaseDeepAgentBackend`.
    ``clear()`` semantics are implementation-defined: the filesystem
    implementation deletes the chat's memory directory (``/flushcontext``
    wipes chat memory along with the workspace).

    Library users may subclass this to provide custom storage (Redis, S3,
    in-memory, etc.) — same extension model as :class:`BaseDeepAgentBackend`.

    .. note::
       ``routes()`` is a breaking addition to this ABC vs. the previously
       merged memory shape (issue #71) — acceptable, this is a
       project-internal library.
    """

    @property
    @abc.abstractmethod
    def backend(self) -> "BackendProtocol":
        """The underlying :class:`BackendProtocol` instance used by ``MemoryMiddleware``."""
        ...

    @abc.abstractmethod
    def routes(self, sources: Sequence[str]) -> dict[str, "BackendProtocol"]:
        """Return CompositeBackend route entries (prefix → BackendProtocol) so the
        agent's runtime tools can reach memory files.

        Keys are virtual path prefixes ending in ``"/"``; values are backends
        rooted at the routed location — ``CompositeBackend`` strips the route
        prefix before delegating, so each value must resolve the stripped path
        inside its own root. Unlike skills, memory routes must support WRITE
        (durable ``edit_file`` write-back), not just read. Non-filesystem
        implementations may map prefixes to ``StoreBackend``-style backends, or
        return ``{}`` if memory is served exclusively through ``MemoryMiddleware``.
        """
        ...

    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear the backend state.

        Memory is per-chat state. The filesystem implementation deletes the
        chat's memory directory; other implementations define their own
        semantics (e.g. dropping the chat's memory keys).
        """
        ...
