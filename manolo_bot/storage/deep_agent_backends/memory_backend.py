import asyncio

from deepagents.backends import StateBackend

from manolo_bot.storage.deep_agent_backends.base import BaseDeepAgentBackend

# Keyed by (bot_uuid, chat_id) to avoid cross-bot leakage when a single
# process serves multiple bots. The chat_id alone is not unique across bots.
_instances: dict[tuple[str, int], StateBackend] = {}
_instances_lock = asyncio.Lock()


class MemoryDeepAgentBackend(BaseDeepAgentBackend):
    """
    In-memory implementation of the deep agent filesystem backend.

    Uses a module-level dict of StateBackend instances keyed by
    (bot_uuid, chat_id), mirroring the pattern used by MemoryMessagesStorage
    while preventing cross-bot data leakage in multi-tenant deployments.
    """

    @staticmethod
    def _key(bot_uuid: str, chat_id: int) -> tuple[str, int]:
        return (bot_uuid, chat_id)

    def _build_backend(self) -> StateBackend:
        key = self._key(self.bot_uuid, self.chat_id)
        backend = _instances.get(key)
        if backend is None:
            backend = StateBackend()
            _instances[key] = backend
        return backend

    async def clear(self) -> None:
        """
        Clears the backend state for the current chat.

        Removes the cached StateBackend instance for this (bot_uuid, chat_id)
        pair. If a new instance is requested after clearing, a fresh
        StateBackend is created.
        """
        async with _instances_lock:
            _instances.pop(self._key(self.bot_uuid, self.chat_id), None)
