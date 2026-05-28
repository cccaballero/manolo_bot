import abc
import json
from abc import abstractmethod
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


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
