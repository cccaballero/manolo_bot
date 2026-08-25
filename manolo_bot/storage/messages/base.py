import abc
import json
from abc import abstractmethod
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

#: Marker prefix that distinguishes the auto-generated conversation summary
#: (stored as a leading SystemMessage) from any other system message.
SUMMARY_PREFIX = "CONVERSATION SUMMARY: "


def get_messages_key(bot_uuid: str, chat_id: int) -> str:
    """
    Generates a key for storing messages in a database based on bot UUID and chat ID.
    """
    return f"{bot_uuid}:{chat_id}"


def convert_json_to_message(json_message: str) -> BaseMessage:
    """
    Converts a JSON string representation of a message into a BaseMessage instance.
    """
    message = json.loads(json_message)
    message_type = message.get("type")
    if message_type == "system":
        return SystemMessage(**message)
    elif message_type == "human":
        return HumanMessage(**message)
    elif message_type == "ai":
        return AIMessage(**message)
    else:
        return BaseMessage(**message)


@dataclass
class StorageMessage:
    message: BaseMessage
    deleted: bool = False
    new: bool = False


class BaseDBHelper(abc.ABC):
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnects from the database.
        """
        pass

    async def connect(self) -> None:
        """
        Connects to the database.
        """
        pass


class BaseMessagesStorage(abc.ABC):
    """
    Abstract base class for message storage.

    Provides the interface for persisting and retrieving chat messages.
    """

    def __init__(self, bot_uuid: str, chat_id: int) -> None:
        self.bot_uuid = bot_uuid
        self.chat_id = chat_id
        self._messages: list[StorageMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        """
        Returns a list of non-deleted messages.
        """
        return [storage_message.message for storage_message in self._messages if not storage_message.deleted]

    def get_summary(self) -> str | None:
        """
        Returns the content of the persisted conversation summary, if any.

        The summary is stored as a ``SystemMessage`` (flagged with
        :data:`SUMMARY_PREFIX`) at the front of the message list. This method
        scans non-deleted messages and returns the first one that carries the
        marker prefix, so it keeps working even if a backend reorders messages
        (e.g. after a refresh).

        :return: The summary text without the marker prefix, or None if absent.
        """
        for storage_message in self._messages:
            if storage_message.deleted:
                continue
            message = storage_message.message
            content = message.content
            if isinstance(message, SystemMessage) and isinstance(content, str) and content.startswith(SUMMARY_PREFIX):
                return content[len(SUMMARY_PREFIX) :]
        return None

    def set_summary(self, text: str) -> None:
        """
        Replaces the conversation summary with the given text.

        Any existing summary message is marked as deleted, then a new
        ``SystemMessage`` (``new=True``) is inserted at the front of the
        message list, before the first non-deleted message, so it is the first
        message returned by :attr:`messages`. Both Memory and Redis backends
        persist it transparently as a regular message.

        Note: inserting the summary shifts the non-deleted indices used by
        :meth:`delete_message`. Callers should always operate on fresh state
        (i.e. query ``messages``/``get_summary`` after any insertion).

        :param text: The summary text.
        """
        for storage_message in self._messages:
            if storage_message.deleted:
                continue
            message = storage_message.message
            content = message.content
            if isinstance(message, SystemMessage) and isinstance(content, str) and content.startswith(SUMMARY_PREFIX):
                storage_message.deleted = True

        insert_index = 0
        for i, storage_message in enumerate(self._messages):
            if not storage_message.deleted:
                insert_index = i
                break
        self._messages.insert(
            insert_index,
            StorageMessage(message=SystemMessage(content=f"{SUMMARY_PREFIX}{text}"), new=True),
        )

    @abstractmethod
    async def refresh_messages(self) -> None:
        """
        Updates the messages list from the database asynchronously.
        """
        pass

    def add_message(self, message: BaseMessage) -> None:
        """
        Adds a new message.
        """
        self._messages.append(StorageMessage(message=message, new=True))

    def delete_message(self, index: int) -> None:
        """
        Deletes a message from the storage by index.

        The index refers to the position among *non-deleted* messages (the
        same indexing used by :attr:`messages`). Note that inserting a summary
        via :meth:`set_summary` shifts these indices, so callers should operate
        on fresh state.
        """
        i = 0
        for storage_message in self._messages:
            if storage_message.deleted:
                continue
            if i == index:
                storage_message.deleted = True
                break
            if not storage_message.deleted:
                i += 1

    @abstractmethod
    async def clear_messages(self) -> None:
        """
        Clears all messages from the storage.
        """
        pass

    @abstractmethod
    async def commit(self) -> None:
        """
        Include new messages and remove deleted messages from the database asynchronously.
        """
        pass
