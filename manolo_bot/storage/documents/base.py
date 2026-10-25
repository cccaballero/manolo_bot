import abc
from abc import abstractmethod


class BaseDocumentStorage(abc.ABC):
    """
    Abstract base class for document storage.

    Provides the interface for persisting and retrieving extracted document text.
    Documents are grouped by an opaque storage key (a Telegram chat id, a Slack
    channel, or any other conversation identifier).
    """

    def __init__(self, bot_uuid: str) -> None:
        self.bot_uuid = bot_uuid

    @abstractmethod
    async def store(self, chat_id: str | int, filename: str, text: str) -> None:
        """
        Stores the extracted text of a document.

        :param chat_id: The storage key of the conversation.
        :param filename: The name of the document.
        :param text: The extracted text.
        """
        pass

    @abstractmethod
    async def retrieve(self, chat_id: str | int, filename: str) -> str | None:
        """
        Retrieves the extracted text of a document.

        :param chat_id: The storage key of the conversation.
        :param filename: The name of the document.
        :return: The extracted text or None if not found.
        """
        pass

    @abstractmethod
    async def clear(self, chat_id: str | int) -> None:
        """
        Clears all stored documents for a specific storage key.

        :param chat_id: The storage key of the conversation.
        """
        pass

    @abstractmethod
    async def list_documents(self, chat_id: str | int) -> list[str]:
        """
        Lists all stored documents for a specific storage key.

        :param chat_id: The storage key of the conversation.
        :return: A list of filenames.
        """
        pass
