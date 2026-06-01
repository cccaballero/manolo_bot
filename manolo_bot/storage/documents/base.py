import abc
from abc import abstractmethod


class BaseDocumentStorage(abc.ABC):
    """
    Abstract base class for document storage.

    Provides the interface for persisting and retrieving extracted document text.
    """

    def __init__(self, bot_uuid: str) -> None:
        self.bot_uuid = bot_uuid

    @abstractmethod
    async def store(self, chat_id: int, filename: str, text: str) -> None:
        """
        Stores the extracted text of a document.

        :param chat_id: The ID of the chat.
        :param filename: The name of the document.
        :param text: The extracted text.
        """
        pass

    @abstractmethod
    async def retrieve(self, chat_id: int, filename: str) -> str | None:
        """
        Retrieves the extracted text of a document.

        :param chat_id: The ID of the chat.
        :param filename: The name of the document.
        :return: The extracted text or None if not found.
        """
        pass

    @abstractmethod
    async def clear(self, chat_id: int) -> None:
        """
        Clears all stored documents for a specific chat.

        :param chat_id: The ID of the chat.
        """
        pass

    @abstractmethod
    async def list_documents(self, chat_id: int) -> list[str]:
        """
        Lists all stored documents for a specific chat.

        :param chat_id: The ID of the chat.
        :return: A list of filenames.
        """
        pass
