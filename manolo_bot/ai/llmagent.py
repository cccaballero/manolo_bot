import base64
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import aiohttp
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.document_loaders import DocumentLoader, UnsupportedFileError
from manolo_bot.ai.llmbot import FileTooLargeError, LLMBot
from manolo_bot.storage.documents.base import BaseDocumentStorage
from manolo_bot.storage.messages.base import BaseMessagesStorage

if TYPE_CHECKING:
    from manolo_bot.ai.history_policy import BaseHistoryPolicy
    from manolo_bot.rag.base import BaseRAGBackend
    from manolo_bot.rag.sources import RAGSource


class LLMAgent(LLMBot):
    """
    Advanced Telegram LLM Chat Bot using a LangGraph-based agent.

    This bot can use tools and dynamically integrate with MCP servers.
    """

    bind_tools_on_init = False

    def _base_messages(self) -> list[BaseMessage]:
        """Hook for subclasses to customise the messages prepended before every agent call."""
        return self.system_instructions

    def __init__(
        self,
        llm: BaseChatModel,
        bot_config: BotConfig,
        system_instructions: list[BaseMessage],
        messages_storage: BaseMessagesStorage,
        tools: list[BaseTool] | None = None,
        documents_storage: BaseDocumentStorage | None = None,
        system_instructions_mapping=None,
        rag_backend: "BaseRAGBackend | None" = None,
        rag_sources: "Sequence[RAGSource] | None" = None,
        history_policy: "BaseHistoryPolicy | None" = None,
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
        # Don't create agent yet - wait for async initialization
        self.agent = None
        # First-class RAG backend (appended as a retriever tool at init time,
        # separate from the `tools=` custom-tools channel).
        self._rag_backend = rag_backend
        # Structured sources for tool naming/scoping; the library channel owns
        # records explicitly (env strings never reach the agents).
        self._rag_sources = rag_sources
        # History policy for agent loop traces (lazy import to avoid cycles;
        # None keeps backward compat by defaulting to FinalOnlyPolicy).
        if history_policy is None:
            from manolo_bot.ai.history_policy import FinalOnlyPolicy

            history_policy = FinalOnlyPolicy()
        self.history_policy = history_policy

    def _stage_agent_trace(self, full: list[BaseMessage], sent_len: int) -> BaseMessage:
        """Apply history policy to the agent tail and stage it for persistence.

        ``full`` is ``result["messages"]`` and ``tail = full[sent_len:]`` is
        the new loop trace. Persists all selected messages EXCEPT the last
        one and returns the last: the final message is left for
        ``LLMBot.postprocess_response`` to persist (it owns the text-only
        ``AIMessage`` write in main.py flow), avoiding a double-persist.
        Falls back to ``full[-1]`` when the policy selects nothing.
        """
        tail = full[sent_len:]
        selected = self.history_policy.select(tail)
        if not selected:
            return full[-1]
        for m in selected[:-1]:
            self.messages_storage.add_message(m)
        return selected[-1]

    def _resolve_rag_tools(self, tools: list[BaseTool]) -> list[BaseTool]:
        """Append global + per-source RAG retriever tools (each skips on MCP clash)."""
        if self._rag_backend is None:
            return tools
        from manolo_bot.rag.base import RAG_TOOL_NAME
        from manolo_bot.rag.prompting import build_rag_tool_description

        sources = list(self._rag_sources or [])
        resolved = tools
        if any(getattr(t, "name", None) == RAG_TOOL_NAME for t in resolved):
            logging.info("RAG tool %r already provided (e.g. by MCP); skipping backend tool", RAG_TOOL_NAME)
        else:
            description = build_rag_tool_description(sources)
            resolved = resolved + [self._rag_backend.as_tool(RAG_TOOL_NAME, description)]
        for tool in self._rag_backend.as_source_tools(sources):
            if any(getattr(t, "name", None) == tool.name for t in resolved):
                logging.info("RAG tool %r already provided (e.g. by MCP); skipping backend tool", tool.name)
                continue
            resolved = resolved + [tool]
        return resolved

    async def initialize_async_resources(self) -> None:
        """Initialize async resources and create agent with all tools."""
        await super().initialize_async_resources()

        # Create agent with all tools (custom + MCP)
        from manolo_bot.ai.tools import get_all_tools

        # Use the tools passed in __init__ if available, otherwise get default ones
        tools = await get_all_tools(
            self._mcp_manager, self.bot_config, document_storage=self.documents_storage, custom_tools=self.tools
        )
        tools = self._resolve_rag_tools(tools)

        self.agent = create_agent(
            model=self.llm,
            tools=tools,
        )
        logging.debug(f"Agent created with {len(tools)} tools")

    # is probably better to not use the agent for this
    # def generate_feedback_message(self, prompt: str, max_length: int = 200) -> str:
    #     logging.debug("Generating feedback message")
    #
    #     response = self.agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    #
    #     # Clean up the response if needed
    #     feedback_message = response["messages"][-1].content.strip()
    #
    #     # Ensure the message isn't too long
    #     if len(feedback_message) > max_length:
    #         feedback_message = feedback_message[: max_length - 3] + "..."
    #
    #     logging.debug(f"Generated feedback message: {feedback_message}")
    #     return feedback_message

    async def answer_message(self, chat_id: int, message: str) -> BaseMessage:
        self.messages_storage.add_message(HumanMessage(content=message))
        await self.truncate_chat_context()

        config = self._get_langchain_config(chat_id)
        sent = self._base_messages() + self.messages_storage.messages
        sent_len = len(sent)
        result = await self.agent.ainvoke(
            {"messages": sent},
            config=config,
        )
        full = result["messages"]
        return self._stage_agent_trace(full, sent_len)

    async def answer_image_message(self, chat_id: int, text: str, image: str) -> BaseMessage:
        """
        Answer an image message.
        :param chat_id: Chat ID
        :param text: Text to answer
        :param image: Image to answer
        :return: Response
        """
        logging.debug(f"Image message: {text}")

        try:
            async with aiohttp.ClientSession() as session:
                timeout = self._get_session_timeout()

                async with session.get(image, timeout=timeout, headers=self._download_headers()) as response:
                    response.raise_for_status()
                    image_bytes = await response.read()
                    image_data = base64.b64encode(image_bytes).decode("utf-8")

                llm_message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ]
                )
                self.messages_storage.add_message(llm_message)
                await self.truncate_chat_context()
                config = self._get_langchain_config(chat_id)
                # Preserve existing behavior: image path omits _base_messages().
                sent_messages = self.messages_storage.messages
                sent_len = len(sent_messages)
                result = await self.agent.ainvoke({"messages": sent_messages}, config=config)
                full = result["messages"]
                response = self._stage_agent_trace(full, sent_len)
        except (aiohttp.ClientError, Exception) as e:
            if isinstance(e, aiohttp.ClientError):
                logging.error(f"Failed to get image: {image}")
            logging.exception(e)
            response = BaseMessage(content="NO_ANSWER", type="text")

        logging.debug(f"Image message response: {response}")
        return response

    async def answer_voice_message(self, chat_id: int, text: str, audio: str) -> BaseMessage:
        """
        Answer a voice message.
        :param chat_id: Chat ID
        :param text: Text to answer
        :param audio: Audio to answer
        :return: Response
        """
        logging.debug(f"Voice message: {text}")

        try:
            async with aiohttp.ClientSession() as session:
                audio_bytes = await self._download_file(audio, session, size_limit=self.bot_config.max_voice_size)
                audio_data = base64.b64encode(audio_bytes).decode("utf-8")

                llm_message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "media",
                            "mime_type": "audio/ogg",
                            "data": audio_data,
                        },
                    ]
                )
                self.messages_storage.add_message(llm_message)
                await self.truncate_chat_context()
                config = self._get_langchain_config(chat_id)
                sent = self._base_messages() + self.messages_storage.messages
                sent_len = len(sent)
                result = await self.agent.ainvoke({"messages": sent}, config=config)
                full = result["messages"]
                response = self._stage_agent_trace(full, sent_len)
        except FileTooLargeError:
            error_prompt = (
                f"Generate a brief, friendly response in {self.bot_config.preferred_language} "
                f"explaining that the voice message is too long and you cannot process it. "
                f"Keep it under 150 characters and maintain your character's style."
            )
            feedback = await self.generate_feedback_message(error_prompt, chat_id=chat_id)
            response = AIMessage(content=feedback)
        except (aiohttp.ClientError, Exception) as e:
            if isinstance(e, aiohttp.ClientError):
                logging.error(f"Failed to get audio: {audio}")
            logging.exception(e)
            response = BaseMessage(content="NO_ANSWER", type="text")

        logging.debug(f"Voice message response: {response}")
        return response

    async def answer_document_message(self, chat_id: int, text: str, document_url: str, filename: str) -> BaseMessage:
        """
        Answer a document message using the agent.

        :param chat_id: Chat ID
        :param text: Text to answer
        :param document_url: Document URL
        :param filename: Original filename
        :return: Response
        """
        logging.debug(f"Document message: {filename}")

        try:
            doc_key, _ = await self._process_and_store_document(chat_id, document_url, filename)

            # Add a pointer message to history
            pointer_message = HumanMessage(
                content=f"User uploaded a document: {filename}. "
                f"Use the read_document tool with filename '{doc_key}' to access it."
            )
            self.messages_storage.add_message(pointer_message)
            await self.truncate_chat_context()

            # We don't stuff the prompt for the agent, we just add the user message
            # and the agent will use the tool if needed.
            self.messages_storage.add_message(HumanMessage(content=text))

            config = self._get_langchain_config(chat_id)
            sent = self._base_messages() + self.messages_storage.messages
            sent_len = len(sent)
            result = await self.agent.ainvoke({"messages": sent}, config=config)
            full = result["messages"]
            response = self._stage_agent_trace(full, sent_len)
        except UnsupportedFileError:
            extension = filename.split(".")[-1].lower()
            supported = ", ".join([ext.upper() for ext in DocumentLoader.SUPPORTED_EXTENSIONS])
            error_prompt = (
                f"Generate a brief, friendly response in {self.bot_config.preferred_language} "
                f"explaining that you cannot read .{extension} files yet. "
                f"Mention that you support {supported}. "
                f"Keep it under 150 characters and maintain your character's style."
            )
            feedback = await self.generate_feedback_message(error_prompt, chat_id=chat_id)
            response = AIMessage(content=feedback)

        except FileTooLargeError:
            error_prompt = (
                f"Generate a brief, friendly response in {self.bot_config.preferred_language} "
                f"explaining that the document is too large and you cannot process it. "
                f"Keep it under 150 characters and maintain your character's style."
            )
            feedback = await self.generate_feedback_message(error_prompt, chat_id=chat_id)
            response = AIMessage(content=feedback)

        except (aiohttp.ClientError, Exception) as e:
            if isinstance(e, aiohttp.ClientError):
                logging.error(f"Failed to get document: {document_url}")
            logging.exception(e)
            response = BaseMessage(content="NO_ANSWER", type="text")

        logging.debug(f"Document message response: {response}")
        return response
