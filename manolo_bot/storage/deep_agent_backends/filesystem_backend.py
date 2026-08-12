import os
import shutil

from deepagents.backends.filesystem import FilesystemBackend

from manolo_bot.storage.deep_agent_backends.base import BaseDeepAgentBackend


class FilesystemDeepAgentBackend(BaseDeepAgentBackend):
    """
    Filesystem-based implementation of the deep agent filesystem backend.

    Builds a per-chat filesystem path from bot_uuid and chat_id,
    mirroring the pattern used by FileDocumentsStorage.
    """

    def __init__(self, bot_uuid: str, chat_id: int, workspace_path: str, virtual_mode: bool = True) -> None:
        self._workspace_path = workspace_path
        self._virtual_mode = virtual_mode
        super().__init__(bot_uuid, chat_id)

    def _build_backend(self) -> FilesystemBackend:
        chat_path = os.path.join(self._workspace_path, self.bot_uuid, str(self.chat_id))
        return FilesystemBackend(root_dir=chat_path, virtual_mode=self._virtual_mode)

    @property
    def backend(self) -> FilesystemBackend:
        return self._backend

    async def clear(self) -> None:
        chat_path = os.path.join(self._workspace_path, self.bot_uuid, str(self.chat_id))
        if os.path.exists(chat_path):
            shutil.rmtree(chat_path)
