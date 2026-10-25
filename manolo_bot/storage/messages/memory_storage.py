from manolo_bot.storage.messages.base import BaseMessagesStorage, StorageMessage

#: Process-global message store. Keyed by ``(bot_uuid, chat_id)`` so two bots
#: never share history even when they happen to use the same chat key.
_chats: dict[tuple[str, str], list] = {}


class MemoryMessagesStorage(BaseMessagesStorage):
    """
    In-memory implementation of message storage.
    """

    def __init__(self, bot_uuid: str, chat_id: str | int) -> None:
        """
        Initializes the memory messages storage.

        :param bot_uuid: The UUID of the bot.
        :param chat_id: The storage key of the conversation (an opaque
            non-empty string; numeric ids normalize to their decimal form).
        """
        super().__init__(bot_uuid, chat_id)

    async def refresh_messages(self) -> None:
        """
        Updates the messages list from the memory storage.
        """
        self._messages = [StorageMessage(message=message) for message in _chats.get((self.bot_uuid, self.chat_id), [])]

    async def clear_messages(self) -> None:
        """
        Clears all messages from the memory storage for the current chat.
        """
        _chats[(self.bot_uuid, self.chat_id)] = []
        self._messages = []

    async def commit(self) -> None:
        """
        Include new messages and remove deleted messages from the memory storage.

        The persisted list is rebuilt from the in-memory non-deleted messages so
        that insertion order is preserved — including summaries inserted at the
        front via ``set_summary``.
        """
        _chats[(self.bot_uuid, self.chat_id)] = [sm.message for sm in self._messages if not sm.deleted]
        await self.refresh_messages()
