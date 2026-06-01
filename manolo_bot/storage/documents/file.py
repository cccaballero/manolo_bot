import os
import tempfile
from pathlib import Path

from manolo_bot.storage.documents.base import BaseDocumentStorage


class FileDocumentStorage(BaseDocumentStorage):
    """
    File-based implementation of document storage.
    """

    def __init__(self, bot_uuid: str, base_path: str | None = None) -> None:
        """
        Initializes the file document storage.

        :param bot_uuid: The UUID of the bot.
        :param base_path: The base path for storage. Defaults to system temp dir.
        """
        super().__init__(bot_uuid)
        if base_path is None:
            base_path = os.path.join(tempfile.gettempdir(), "manolo_bot", "documents")
        self.base_path = Path(base_path) / bot_uuid

    def _get_chat_path(self, chat_id: int) -> Path:
        """
        Gets the path for a specific chat.

        :param chat_id: The ID of the chat.
        :return: The Path object for the chat directory.
        """
        path = (self.base_path / str(chat_id)).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_safe_path(self, chat_path: Path, filename: str) -> Path:
        """
        Gets a safe path for a file, preventing directory traversal.

        :param chat_path: The base path for the chat.
        :param filename: The name of the file.
        :return: The safe Path object.
        :raises ValueError: If the path is insecure.
        """
        # Prevent directory traversal
        file_path = (chat_path / f"{filename}.txt").resolve()
        if not str(file_path).startswith(str(chat_path)):
            raise ValueError(f"Insecure filename or path: {filename}")
        return file_path

    async def store(self, chat_id: int, filename: str, text: str) -> None:
        """
        Stores the extracted text of a document in the filesystem.

        :param chat_id: The ID of the chat.
        :param filename: The name of the document.
        :param text: The extracted text.
        """
        chat_path = self._get_chat_path(chat_id)
        file_path = self._get_safe_path(chat_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    async def retrieve(self, chat_id: int, filename: str) -> str | None:
        """
        Retrieves the extracted text of a document from the filesystem.

        :param chat_id: The ID of the chat.
        :param filename: The name of the document.
        :return: The extracted text or None if not found or path is insecure.
        """
        chat_path = self._get_chat_path(chat_id)
        try:
            file_path = self._get_safe_path(chat_path, filename)
        except ValueError:
            return None

        if file_path.exists():
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        return None

    async def clear(self, chat_id: int) -> None:
        """
        Clears all stored documents for a specific chat from the filesystem.

        :param chat_id: The ID of the chat.
        """
        chat_path = self._get_chat_path(chat_id)
        if chat_path.exists():
            for file in chat_path.glob("*.txt"):
                if file.resolve().parent == chat_path:
                    file.unlink()

    async def list_documents(self, chat_id: int) -> list[str]:
        """
        Lists all stored documents for a specific chat in the filesystem.

        :param chat_id: The ID of the chat.
        :return: A list of filenames.
        """
        chat_path = self._get_chat_path(chat_id)
        if not chat_path.exists():
            return []
        return [f.stem for f in chat_path.glob("*.txt") if f.resolve().parent == chat_path]
