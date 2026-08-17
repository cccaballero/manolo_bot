import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol


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
    """

    @property
    @abc.abstractmethod
    def backend(self) -> "BackendProtocol":
        """The underlying :class:`BackendProtocol` instance used by ``SkillsMiddleware``."""
        ...

    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear the backend state.

        Skills are operator-provided, not per-chat state. Implementations
        should typically be a no-op — wiping operator content when a user
        runs ``/flushcontext`` would be destructive and surprising.
        """
        ...
