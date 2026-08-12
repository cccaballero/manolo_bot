from deepagents.backends import StateBackend

from manolo_bot.storage.deep_agent_backends.base import BaseDeepAgentBackend

_instances: dict[int, StateBackend] = {}


class MemoryDeepAgentBackend(BaseDeepAgentBackend):
    """
    In-memory implementation of the deep agent filesystem backend.

    Uses a module-level dict of StateBackend instances keyed by chat_id,
    mirroring the pattern used by MemoryMessagesStorage.
    """

    def _build_backend(self) -> StateBackend:
        if self.chat_id not in _instances:
            _instances[self.chat_id] = StateBackend()
        return _instances[self.chat_id]

    async def clear(self) -> None:
        _instances.pop(self.chat_id, None)
