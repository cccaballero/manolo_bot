from manolo_bot.storage.documents.base import BaseDocumentStorage
from manolo_bot.storage.keys import escape_key_component, normalize_storage_key
from manolo_bot.storage.messages.redis_storage import RedisDBHelper


def get_documents_key(bot_uuid: str, chat_id: str | int) -> str:
    """
    Gets the Redis key for the document filename set of a storage key.

    :param bot_uuid: The UUID of the bot.
    :param chat_id: The opaque storage key of the conversation.
    :return: The Redis key.
    """
    key = escape_key_component(normalize_storage_key(chat_id))
    return f"{bot_uuid}:documents:{key}"


def get_document_key(bot_uuid: str, chat_id: str | int, filename: str) -> str:
    """
    Gets the Redis key for a stored document.

    :param bot_uuid: The UUID of the bot.
    :param chat_id: The opaque storage key of the conversation.
    :param filename: The name of the document.
    :return: The Redis key.
    """
    key = escape_key_component(normalize_storage_key(chat_id))
    return f"{bot_uuid}:documents:{key}:{escape_key_component(filename)}"


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

    async def store(self, chat_id: str | int, filename: str, text: str) -> None:
        """
        Stores the extracted text of a document in Redis.

        :param chat_id: The opaque storage key of the conversation.
        :param filename: The name of the document.
        :param text: The extracted text.
        """
        key = normalize_storage_key(chat_id)
        await self.client.set(get_document_key(self.bot_uuid, key, filename), text)
        await self.client.sadd(get_documents_key(self.bot_uuid, key), filename)

    async def retrieve(self, chat_id: str | int, filename: str) -> str | None:
        """
        Retrieves the extracted text of a document from Redis.

        :param chat_id: The opaque storage key of the conversation.
        :param filename: The name of the document.
        :return: The extracted text or None if not found.
        """
        key = normalize_storage_key(chat_id)
        raw_document = await self.client.get(get_document_key(self.bot_uuid, key, filename))
        if raw_document is None:
            return None

        if isinstance(raw_document, bytes):
            return raw_document.decode("utf-8")

        return raw_document

    async def clear(self, chat_id: str | int) -> None:
        """
        Clears all stored documents for a specific storage key from Redis.

        :param chat_id: The opaque storage key of the conversation.
        """
        key = normalize_storage_key(chat_id)
        documents_key = get_documents_key(self.bot_uuid, key)
        filenames = await self.client.smembers(documents_key)

        keys_to_delete = [documents_key]
        for filename in filenames:
            if isinstance(filename, bytes):
                filename = filename.decode("utf-8")
            keys_to_delete.append(get_document_key(self.bot_uuid, key, filename))

        if keys_to_delete:
            await self.client.delete(*keys_to_delete)

    async def list_documents(self, chat_id: str | int) -> list[str]:
        """
        Lists all stored documents for a specific storage key in Redis.

        :param chat_id: The opaque storage key of the conversation.
        :return: A list of filenames.
        """
        key = normalize_storage_key(chat_id)
        filenames = await self.client.smembers(get_documents_key(self.bot_uuid, key))

        return [filename.decode("utf-8") if isinstance(filename, bytes) else filename for filename in filenames]
