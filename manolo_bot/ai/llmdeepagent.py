import logging
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.ai.tools import get_all_tools
from manolo_bot.storage.deep_agent_backends.base import BaseDeepAgentBackend
from manolo_bot.storage.documents.base import BaseDocumentStorage
from manolo_bot.storage.messages.base import BaseMessagesStorage

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol


class LLMDeepAgent(LLMAgent):
    """
    Advanced Telegram LLM Chat Bot using LangChain Deep Agents harness.

    Extends LLMAgent with the full deep agents stack:
    - To-do list planning (TodoListMiddleware)
    - Virtual filesystem (FilesystemMiddleware + StateBackend)
    - Sub-agents (via create_deep_agent subagents parameter)

    The deep agent harness has its own internal system prompt (planning,
    filesystem, sub-agent instructions). The bot's character/persona
    instructions from :attr:`system_instructions` are passed as the
    ``system_prompt`` string so they are merged with the harness prompt:
    ``bot_instructions + "\\n\\n" + harness_base_prompt``.

    The virtual filesystem backend is injected from outside
    (see :meth:`main.instance_llm_bot`), keeping the class decoupled
    from chat-specific path logic.
    """

    bind_tools_on_init = False

    def __init__(
        self,
        llm: BaseChatModel,
        bot_config: BotConfig,
        system_instructions: list[BaseMessage],
        messages_storage: BaseMessagesStorage,
        tools: list[BaseTool] | None = None,
        documents_storage: BaseDocumentStorage | None = None,
        system_instructions_mapping=None,
        backend: BaseDeepAgentBackend | None = None,
    ) -> None:
        super().__init__(
            llm,
            bot_config,
            system_instructions,
            messages_storage,
            tools=tools,
            documents_storage=documents_storage,
            system_instructions_mapping=system_instructions_mapping,
        )
        self._backend_wrapper = backend
        self._backend: BackendProtocol = backend.backend if backend else StateBackend()
        self.agent = None

    async def initialize_async_resources(self) -> None:
        """Initialize async resources and create deep agent with all tools."""
        await super().initialize_async_resources()

        tools = await get_all_tools(
            self._mcp_manager, self.bot_config, document_storage=self.documents_storage, custom_tools=self.tools
        )

        instructions_text = self._system_instructions[0].content if self._system_instructions else ""

        self.agent = create_deep_agent(
            model=self.llm,
            tools=tools,
            system_prompt=instructions_text,
            middleware=[
                FilesystemMiddleware(backend=self._backend),
                TodoListMiddleware(),
            ],
            subagents=[],
        )
        logging.debug(f"Deep agent created with {len(tools)} tools")

    def _base_messages(self) -> list[BaseMessage]:
        """Only keep the AIMessage priming, skipping the SystemMessage (already in system_prompt)."""
        if self.system_instructions and len(self.system_instructions) > 1:
            return [self.system_instructions[1]]
        return []

    async def clean_context(self) -> None:
        await super().clean_context()
        if self._backend_wrapper:
            await self._backend_wrapper.clear()
