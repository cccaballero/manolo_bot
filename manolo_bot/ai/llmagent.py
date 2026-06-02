import base64
import logging

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


class LLMAgent(LLMBot):
    """
    Advanced Telegram LLM Chat Bot using a LangGraph-based agent.

    This bot can use tools and dynamically integrate with MCP servers.
    """

    bind_tools_on_init = False

    def __init__(
        self,
        llm: BaseChatModel,
        bot_config: BotConfig,
        system_instructions: list[BaseMessage],
        messages_storage: BaseMessagesStorage,
        tools: list[BaseTool] | None = None,
        document_storage: BaseDocumentStorage | None = None,
        system_instructions_mapping=None,
    ) -> None:
        super().__init__(
            llm,
            bot_config,
            system_instructions,
            messages_storage,
            tools=tools,
            document_storage=document_storage,
            system_instructions_mapping=system_instructions_mapping,
        )
        # Don't create agent yet - wait for async initialization
        self.agent = None

    async def initialize_async_resources(self) -> None:
        """Initialize async resources and create agent with all tools."""
        await super().initialize_async_resources()

        # Create agent with all tools (custom + MCP)
        from manolo_bot.ai.tools import get_all_tools

        # Use the tools passed in __init__ if available, otherwise get default ones
        tools = await get_all_tools(
            self._mcp_manager, self.bot_config, document_storage=self.document_storage, custom_tools=self.tools
        )

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
        self.truncate_chat_context()

        config = self._get_langchain_config(chat_id)
        ai_msg = await self.agent.ainvoke(
            {"messages": self.system_instructions + self.messages_storage.messages},
            config=config,
        )
        return ai_msg["messages"][-1]

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

                async with session.get(image, timeout=timeout) as response:
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
                self.truncate_chat_context()
                config = self._get_langchain_config(chat_id)
                response = (await self.agent.ainvoke({"messages": self.messages_storage.messages}, config=config))[
                    "messages"
                ][-1]
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
                timeout = self._get_session_timeout()

                async with session.get(audio, timeout=timeout) as response:
                    response.raise_for_status()
                    audio_bytes = await response.read()
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
                self.truncate_chat_context()
                config = self._get_langchain_config(chat_id)
                response = (
                    await self.agent.ainvoke(
                        {"messages": self.system_instructions + self.messages_storage.messages}, config=config
                    )
                )["messages"][-1]
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
            self.truncate_chat_context()

            # We don't stuff the prompt for the agent, we just add the user message
            # and the agent will use the tool if needed.
            self.messages_storage.add_message(HumanMessage(content=text))

            config = self._get_langchain_config(chat_id)
            response = (
                await self.agent.ainvoke(
                    {"messages": self.system_instructions + self.messages_storage.messages}, config=config
                )
            )["messages"][-1]
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
