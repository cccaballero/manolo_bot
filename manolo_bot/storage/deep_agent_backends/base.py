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