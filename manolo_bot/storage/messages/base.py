import abc
import json
from abc import abstractmethod
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

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
    elif message_type == "tool":
        return ToolMessage(**message)
    elif message_type == "function":
        return FunctionMessage(**message)
    else:
        return BaseMessage(**message)


def _is_ai_tool_call(message: BaseMessage) -> bool:
    """True if the message is an AI message carrying tool calls."""
    return isinstance(message, AIMessage) and bool(getattr(message, "tool_calls", None))


def _is_tool_result(message: BaseMessage) -> bool:
    """True if the message is a tool/function result that pairs with a prior AI message."""
    return isinstance(message, ToolMessage | FunctionMessage)


def expand_tool_block(messages: list[BaseMessage], idx: int) -> tuple[int, int]:
    """
    Return the ``(start, end)`` inclusive span of the atomic AI->Tool* block
    containing ``messages[idx]``.

    A block is an ``AIMessage`` with ``tool_calls`` followed by one or more
    consecutive tool/function result messages. A lone AI message with tool
    calls and no results, or an orphan run of tool results with no leading AI
    message, forms a degenerate block. Any other message maps to ``(idx, idx)``.

    Out-of-range ``idx`` (or an empty list) is passed through as ``(idx, idx)``.

    Truncation/deletion callers must drop (or keep) the whole span atomically:
    deleting a single message inside a block orphans its pair and causes
    provider 400 errors on the next turn.
    """
    if not messages or idx < 0 or idx >= len(messages):
        return (idx, idx)
    message = messages[idx]
    if _is_ai_tool_call(message):
        start = idx
    elif _is_tool_result(message):
        start = idx
        while start - 1 >= 0 and _is_tool_result(messages[start - 1]):
            start -= 1
        if start - 1 >= 0 and _is_ai_tool_call(messages[start - 1]):
            start -= 1
        else:
            # Orphan tool-result run with no leading AI tool-call message:
            # the atomic unit is the consecutive run itself.
            end = idx
            while end + 1 < len(messages) and _is_tool_result(messages[end + 1]):
                end += 1
            return (start, end)
    else:
        return (idx, idx)
    end = start + 1
    while end < len(messages) and _is_tool_result(messages[end]):
        end += 1
    return (start, end - 1)


def find_safe_drop_index(messages: list[BaseMessage], candidate: int) -> int:
    """
    Map a candidate drop index to a pair-safe drop index.

    If ``candidate`` falls inside an atomic AI->Tool* block (see
    :func:`expand_tool_block`), the start of that block is returned so the
    caller can expand to the whole span and drop it atomically instead of
    splitting the pair. Otherwise ``candidate`` is returned unchanged.

    Intended AI-lane pattern for drop-oldest truncation::

        msgs = storage.messages
        safe = find_safe_drop_index(msgs, candidate)
        start, end = expand_tool_block(msgs, safe)
        for i in range(end, start - 1, -1):
            storage.delete_message(i)

    (Deleting highest-index-first keeps the lower non-deleted indices valid.)
    Out-of-range ``candidate`` (or an empty list) is returned unchanged.
    """
    if not messages or candidate < 0 or candidate >= len(messages):
        return candidate
    start, _ = expand_tool_block(messages, candidate)
    return start


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
