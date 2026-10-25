import os
import tempfile
from pathlib import Path

from manolo_bot.storage.documents.base import BaseDocumentStorage
from manolo_bot.storage.keys import normalize_storage_key, storage_path_segment


class FileDocumentsStorage(BaseDocumentStorage):
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
        self.base_path = (Path(base_path) / bot_uuid).resolve()

    def _get_chat_path(self, chat_id: str | int) -> Path:
        """
        Gets the directory for a storage key.

        The key becomes a single, filesystem-safe path segment, and the resolved
        directory is checked to still sit under the documents base.

        :param chat_id: The opaque storage key of the conversation.
        :return: The Path object for the storage key's directory.
        :raises ValueError: If the key is empty or escapes the documents base.
        """
        key = normalize_storage_key(chat_id)
        path = (self.base_path / storage_path_segment(key)).resolve()
        if path != self.base_path and self.base_path not in path.parents:
            raise ValueError(f"Storage key {key!r} resolves outside the documents base path {self.base_path}")

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
        if file_path.parent != chat_path:
            raise ValueError(f"Insecure filename or path: {filename}")
        return file_path

    async def store(self, chat_id: str | int, filename: str, text: str) -> None:
        """
        Stores the extracted text of a document in the filesystem.

        :param chat_id: The opaque storage key of the conversation.
        :param filename: The name of the document.
        :param text: The extracted text.
        """
        chat_path = self._get_chat_path(chat_id)
        file_path = self._get_safe_path(chat_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    async def retrieve(self, chat_id: str | int, filename: str) -> str | None:
        """
        Retrieves the extracted text of a document from the filesystem.

        :param chat_id: The opaque storage key of the conversation.
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

    async def clear(self, chat_id: str | int) -> None:
        """
        Clears all stored documents for a specific storage key from the filesystem.

        :param chat_id: The opaque storage key of the conversation.
        """
        chat_path = self._get_chat_path(chat_id)
        if chat_path.exists():
            for file in chat_path.glob("*.txt"):
                if file.resolve().parent == chat_path:
                    file.unlink()

    async def list_documents(self, chat_id: str | int) -> list[str]:
        """
        Lists all stored documents for a specific storage key in the filesystem.

        :param chat_id: The opaque storage key of the conversation.
        :return: A list of filenames.
        """
        chat_path = self._get_chat_path(chat_id)
        if not chat_path.exists():
            return []
        return [f.stem for f in chat_path.glob("*.txt") if f.resolve().parent == chat_path]
