import logging

import redis

from manolo_bot.storage.base import (
    BaseDBHelper,
    BaseMessagesStorage,
    StorageMessage,
    convert_json_to_message,
    get_messages_key,
)


class RedisDBHelper(BaseDBHelper):
    def __init__(self, db_url) -> None:
        self.pool = redis.asyncio.ConnectionPool.from_url(db_url)
        self.client = None

    async def disconnect(self) -> None:
        logging.debug("Disconnecting from the database")
        await self.client.aclose()

    async def connect(self) -> None:
        logging.debug("Connecting to the database")
        self.client = redis.asyncio.Redis.from_pool(self.pool)


class RedisMessagesStorage(BaseMessagesStorage):
    def __init__(self, db: RedisDBHelper, bot_uuid: str, chat_id: int) -> None:
        super().__init__(bot_uuid, chat_id)
        self.client = db.client

    async def refresh_messages(self) -> None:
        """
        Updates the messages list from the database.
        """
        key = get_messages_key(self.bot_uuid, self.chat_id)
        raw_messages = await self.client.lrange(key, 0, -1)
        self._messages = [StorageMessage(message=convert_json_to_message(raw_message)) for raw_message in raw_messages]

    async def clear_messages(self) -> None:
        await self.client.delete(get_messages_key(self.bot_uuid, self.chat_id))
        self._messages = []

    async def commit(self) -> None:
        """
        Include new messages and remove deleted messages from the database.
        """
        key = get_messages_key(self.bot_uuid, self.chat_id)
        for storage_message in self._messages:
            if storage_message.new:
                await self.client.rpush(key, storage_message.message.model_dump_json())
            elif storage_message.deleted:
                await self.client.lrem(key, 1, storage_message.message.model_dump_json())
        # await self.refresh_messages()
