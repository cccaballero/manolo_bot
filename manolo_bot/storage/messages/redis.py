import logging

import redis

from manolo_bot.storage.messages.base import (
    BaseDBHelper,
    BaseMessagesStorage,
    StorageMessage,
    convert_json_to_message,
    get_messages_key,
)


class RedisDBHelper(BaseDBHelper):
    """
    Helper class for Redis database operations.
    """

    def __init__(self, db_url) -> None:
        """
        Initializes the Redis database helper.

        :param db_url: The URL of the Redis database.
        """
        self.pool = redis.asyncio.ConnectionPool.from_url(db_url)
        self.client = None

    async def disconnect(self) -> None:
        """
        Disconnects from the Redis database.
        """
        logging.debug("Disconnecting from the database")
        await self.client.aclose()

    async def connect(self) -> None:
        """
        Connects to the Redis database.
        """
        logging.debug("Connecting to the database")
        self.client = redis.asyncio.Redis.from_pool(self.pool)


class RedisMessagesStorage(BaseMessagesStorage):
    """
    Redis-based implementation of message storage.
    """

    def __init__(self, db: RedisDBHelper, bot_uuid: str, chat_id: int) -> None:
        """
        Initializes the Redis messages storage.

        :param db: The RedisDBHelper instance.
        :param bot_uuid: The UUID of the bot.
        :param chat_id: The ID of the chat.
        """
        super().__init__(bot_uuid, chat_id)
        self.client = db.client

    async def refresh_messages(self) -> None:
        """
        Updates the messages list from the Redis database.
        """
        key = get_messages_key(self.bot_uuid, self.chat_id)
        raw_messages = await self.client.lrange(key, 0, -1)
        self._messages = [StorageMessage(message=convert_json_to_message(raw_message)) for raw_message in raw_messages]

    async def clear_messages(self) -> None:
        """
        Clears all messages from the Redis database for the current chat.
        """
        await self.client.delete(get_messages_key(self.bot_uuid, self.chat_id))
        self._messages = []

    async def commit(self) -> None:
        """
        Include new messages and remove deleted messages from the Redis database.
        """
        key = get_messages_key(self.bot_uuid, self.chat_id)
        for storage_message in self._messages:
            if storage_message.new:
                await self.client.rpush(key, storage_message.message.model_dump_json())
            elif storage_message.deleted:
                await self.client.lrem(key, 1, storage_message.message.model_dump_json())
        # await self.refresh_messages()
