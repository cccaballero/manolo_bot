from manolo_bot.storage.messages.base import BaseMessagesStorage, StorageMessage

_chats = {}


class MemoryMessagesStorage(BaseMessagesStorage):
    """
    In-memory implementation of message storage.
    """

    def __init__(self, bot_uuid: str, chat_id: int) -> None:
        """
        Initializes the memory messages storage.

        :param bot_uuid: The UUID of the bot.
        :param chat_id: The ID of the chat.
        """
        super().__init__(bot_uuid, chat_id)

    async def refresh_messages(self) -> None:
        """
        Updates the messages list from the memory storage.
        """
        self._messages = [StorageMessage(message=message) for message in _chats.get(self.chat_id, [])]

    async def clear_messages(self) -> None:
        """
        Clears all messages from the memory storage for the current chat.
        """
        _chats[self.chat_id] = []
        self._messages = []

    async def commit(self) -> None:
        """
        Include new messages and remove deleted messages from the memory storage.
        """
        if not _chats.get(self.chat_id, None):
            _chats[self.chat_id] = []
        for storage_message in self._messages:
            if storage_message.new:
                _chats[self.chat_id].append(storage_message.message)
            elif storage_message.deleted:
                try:
                    _chats[self.chat_id].remove(storage_message.message)
                except ValueError:
                    # Message might have been removed already
                    pass
        await self.refresh_messages()
