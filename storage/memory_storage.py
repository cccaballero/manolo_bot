from storage.base import BaseMessagesStorage, StorageMessage

_chats = {}


class MemoryMessagesStorage(BaseMessagesStorage):
    def __init__(self, bot_uuid: str, chat_id: int) -> None:
        super().__init__(bot_uuid, chat_id)

    async def refresh_messages(self) -> None:
        """
        Updates the messages list from the database.
        """
        self._messages = [StorageMessage(message=message) for message in _chats.get(self.chat_id, [])]

    async def clear_messages(self) -> None:
        _chats[self.chat_id] = []
        self._messages = []

    async def commit(self) -> None:
        """
        Include new messages and remove deleted messages from the database.
        """
        if not _chats.get(self.chat_id, None):
            _chats[self.chat_id] = []
        for storage_message in self._messages:
            if storage_message.new:
                _chats[self.chat_id].append(storage_message.message)
            elif storage_message.deleted:
                _chats[self.chat_id].remove(storage_message.message)
        await self.refresh_messages()
