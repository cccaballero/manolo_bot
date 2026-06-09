from manolo_bot.storage.documents.base import BaseDocumentStorage
from manolo_bot.storage.messages.redis_storage import RedisDBHelper


def get_documents_key(bot_uuid: str, chat_id: int) -> str:
    """
    Gets the Redis key for the document filename set of a chat.

    :param bot_uuid: The UUID of the bot.
    :param chat_id: The ID of the chat.
    :return: The Redis key.
    """
    return f"{bot_uuid}:documents:{chat_id}"


def get_document_key(bot_uuid: str, chat_id: int, filename: str) -> str:
    """
    Gets the Redis key for a stored document.

    :param bot_uuid: The UUID of the bot.
    :param chat_id: The ID of the chat.
    :param filename: The name of the document.
    :return: The Redis key.
    """
    return f"{bot_uuid}:documents:{chat_id}:{filename}"


class RedisDocumentsStorage(BaseDocumentStorage):
    """
    Redis-based implementation of document storage.
    """

    def __init__(self, db: RedisDBHelper, bot_uuid: str) -> None:
        """
        Initializes the Redis document storage.

        :param db: The RedisDBHelper instance.
        :param bot_uuid: The UUID of the bot.
        """
        super().__init__(bot_uuid)
        self.client = db.client

    async def store(self, chat_id: int, filename: str, text: str) -> None:
        """
        Stores the extracted text of a document in Redis.

        :param chat_id: The ID of the chat.
        :param filename: The name of the document.
        :param text: The extracted text.
        """
        await self.client.set(get_document_key(self.bot_uuid, chat_id, filename), text)
        await self.client.sadd(get_documents_key(self.bot_uuid, chat_id), filename)

    async def retrieve(self, chat_id: int, filename: str) -> str | None:
        """
        Retrieves the extracted text of a document from Redis.

        :param chat_id: The ID of the chat.
        :param filename: The name of the document.
        :return: The extracted text or None if not found.
        """
        raw_document = await self.client.get(get_document_key(self.bot_uuid, chat_id, filename))
        if raw_document is None:
            return None

        if isinstance(raw_document, bytes):
            return raw_document.decode("utf-8")

        return raw_document

    async def clear(self, chat_id: int) -> None:
        """
        Clears all stored documents for a specific chat from Redis.

        :param chat_id: The ID of the chat.
        """
        documents_key = get_documents_key(self.bot_uuid, chat_id)
        filenames = await self.client.smembers(documents_key)

        keys_to_delete = [documents_key]
        for filename in filenames:
            if isinstance(filename, bytes):
                filename = filename.decode("utf-8")
            keys_to_delete.append(get_document_key(self.bot_uuid, chat_id, filename))

        if keys_to_delete:
            await self.client.delete(*keys_to_delete)

    async def list_documents(self, chat_id: int) -> list[str]:
        """
        Lists all stored documents for a specific chat in Redis.

        :param chat_id: The ID of the chat.
        :return: A list of filenames.
        """
        filenames = await self.client.smembers(get_documents_key(self.bot_uuid, chat_id))

        return [filename.decode("utf-8") if isinstance(filename, bytes) else filename for filename in filenames]
