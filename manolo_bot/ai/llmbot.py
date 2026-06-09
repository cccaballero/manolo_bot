import base64
import copy
import logging
import re
import secrets
from types import TracebackType
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import aiohttp
from google.genai.types import HarmBlockThreshold, HarmCategory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from manolo_bot.ai.config import BotConfig, LLMConfig
from manolo_bot.ai.document_loaders import DocumentLoader, UnsupportedFileError, clean_text
from manolo_bot.ai.tools import get_tool, get_tools
from manolo_bot.storage.documents.utils import generate_document_key
from manolo_bot.storage.messages.base import BaseMessagesStorage

if TYPE_CHECKING:
    from manolo_bot.ai.mcp_manager import MCPManager
    from manolo_bot.storage.documents.base import BaseDocumentStorage


class FileTooLargeError(ValueError):
    """Exception raised when a file exceeds the allowed size."""

    pass


class LLMBuilder:
    """Factory class for creating LangChain Chat Model instances."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self.llm_config = llm_config

    def _get_rate_limiter(self) -> InMemoryRateLimiter:
        return InMemoryRateLimiter(
            requests_per_second=self.llm_config.rate_limiter_requests_per_second,
            check_every_n_seconds=self.llm_config.rate_limiter_check_every_n_seconds,
            max_bucket_size=self.llm_config.rate_limiter_max_bucket_size,
        )

    def _get_chat_ollama(self) -> ChatOllama:
        return ChatOllama(model=self.llm_config.ollama_model)

    def _get_chat_google_generativeai(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.llm_config.google_api_model,
            safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            rate_limiter=self._get_rate_limiter(),
        )

    def _get_chat_openai(self) -> ChatOpenAI:
        api_key = self.llm_config.openai_api_key if self.llm_config.openai_api_key else "not-needed"
        base_url = self.llm_config.openai_api_base_url
        model = self.llm_config.openai_api_model
        params = {
            "openai_api_key": api_key,
        }
        if base_url:
            params["base_url"] = base_url
        if model:
            params["model"] = model
        return ChatOpenAI(rate_limiter=self._get_rate_limiter(), **params)

    def get_llm(self) -> BaseChatModel:
        """
        Creates and returns an instance of the configured LLM.

        :return: A LangChain BaseChatModel instance.
        :raises Exception: If no LLM configuration is found.
        """
        if self.llm_config.ollama_model:
            llm = self._get_chat_ollama()
        elif self.llm_config.google_api_key:
            llm = self._get_chat_google_generativeai()
        elif self.llm_config.openai_api_key or self.llm_config.openai_api_base_url:
            llm = self._get_chat_openai()
        else:
            raise Exception("No LLM backend data found")
        return llm


class LLMBot:
    """
    Base class for a Telegram LLM Chat Bot.

    Handles interaction with the LLM, message processing, and context management.
    """

    bind_tools_on_init = True

    def __init__(
        self,
        llm: BaseChatModel,
        bot_config: BotConfig,
        system_instructions: list[BaseMessage],
        messages_storage: BaseMessagesStorage,
        tools: list[BaseTool] | None = None,
        documents_storage: "BaseDocumentStorage | None" = None,
        system_instructions_mapping=None,
    ) -> None:
        self.bot_config = bot_config
        self._system_instructions = system_instructions
        # self.messages_buffer = messages_buffer
        self.llm = llm
        self.messages_storage: BaseMessagesStorage = messages_storage
        # self._load_llm()
        self.tools = tools
        self.documents_storage = documents_storage
        self._mcp_manager: MCPManager | None = None
        self._async_resources_initialized = False
        self.system_instructions_mapping = system_instructions_mapping or {}

        if self.bind_tools_on_init and self.bot_config.use_tools:
            self._load_tools()

    @property
    def system_instructions(self) -> list[BaseMessage]:
        """Get the system instructions mapping."""
        instructions = copy.deepcopy(self._system_instructions)
        instructions_content = instructions[0].content
        for old, new in self.system_instructions_mapping.items():
            instructions_content = instructions_content.replace(old, new(self))
        instructions[0].content = instructions_content
        return instructions

    def _get_langchain_config(self, chat_id: int) -> RunnableConfig:
        """Helper to create LangChain config with metadata and tags."""
        bot_uuid = self.bot_config.bot_uuid
        user_id = self.bot_config.user_id
        return RunnableConfig(
            tags=[f"bot:{bot_uuid}", f"user:{user_id}"],
            metadata={
                "bot_uuid": bot_uuid,
                "bot_username": self.bot_config.bot_username,
                "user_id": user_id,
                "chat_id": chat_id,
            },
        )

    def _get_session_timeout(self) -> aiohttp.ClientTimeout:
        """Get the timeout for aiohttp sessions."""
        return aiohttp.ClientTimeout(total=self.bot_config.web_content_request_timeout)

    async def _download_file(self, url: str, session: aiohttp.ClientSession, size_limit: int = 0) -> bytes:
        """
        Downloads a file from a URL with a size limit.
        Checks Content-Length header first, then streams in chunks.

        :param url: URL to download
        :param session: aiohttp ClientSession
        :param size_limit: Maximum allowed size in bytes, 0 for no limit
        :return: File content as bytes
        :raises FileTooLargeError: If the file exceeds the maximum allowed size
        """
        timeout = self._get_session_timeout()
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()

            # 1. Header Check (First layer of security)
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if size_limit and int(content_length) > size_limit:
                        raise FileTooLargeError(f"File is too large ({content_length} bytes)")
                except ValueError:
                    # If Content-Length is not an integer, we ignore it and rely on the stream check
                    pass

            # 2. Chunked Download (Second layer of security)
            file_content = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                file_content.extend(chunk)
                if size_limit and len(file_content) > size_limit:
                    raise FileTooLargeError(f"File exceeds maximum allowed size ({len(file_content)} bytes)")

            return bytes(file_content)

    async def initialize_async_resources(self) -> None:
        """Initialize all async resources (MCP, etc.)."""
        if self._async_resources_initialized:
            return

        logging.debug("Initializing async resources...")

        # Initialize MCP if enabled
        if self.bot_config.enable_mcp:
            logging.info("Initializing MCP...")
            try:
                from manolo_bot.ai.mcp_manager import MCPManager

                # TODO: We probably don't want to initialize MCP in each call, maybe we can cache this somehow?
                self._mcp_manager = MCPManager(self.bot_config)
                await self._mcp_manager.connect()
                logging.info("MCP initialized successfully")

                # Reload tools to include MCP tools
                if self.bot_config.use_tools:
                    await self._reload_tools_with_mcp()

            except Exception as e:
                logging.warning(
                    f"MCP initialization failed, continuing without MCP: {e}",
                    exc_info=True,
                )
                self._mcp_manager = None

        self._async_resources_initialized = True

    # async def cleanup(self) -> None:
    #     """Clean up all async resources."""
    #     if not self._async_resources_initialized:
    #         return
    #
    #     logging.debug("Cleaning up async resources...")
    #
    #     # Clean up MCP if it was initialized
    #     if self._mcp_manager is not None:
    #         try:
    #             await self._mcp_manager.close()
    #             logging.info("MCP resources cleaned up successfully")
    #         except Exception as e:
    #             logging.error(f"Error cleaning up MCP resources: {e}", exc_info=True)
    #
    #     self._async_resources_initialized = False

    async def _reload_tools_with_mcp(self) -> None:
        """Reload tools including MCP tools."""
        from manolo_bot.ai.tools import get_all_tools

        tools = await get_all_tools(
            self._mcp_manager, self.bot_config, document_storage=self.documents_storage, custom_tools=self.tools
        )
        self.llm = self.llm.bind_tools(tools)
        logging.debug(f"Reloaded {len(tools)} tools (including MCP)")

    async def close(self) -> None:
        """Close all async resources."""
        if self._mcp_manager:
            await self._mcp_manager.disconnect()

        logging.debug("Async resources closed")

    async def __aenter__(self) -> "LLMBot":
        """Async context manager entry - initialize async resources."""
        await self.initialize_async_resources()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit - cleanup resources."""
        await self.close()

    def _extract_url(self, text: str) -> str | None:
        """
        Extract the URL from the text.
        :param text: Text to extract the URL from
        :return: URL if found, None otherwise
        """
        url = re.search(
            r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|%[0-9a-fA-F][0-9a-fA-F])+",
            text,
        )
        return url.group(0) if url else None

    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from the text.
        :param text: Text to remove URLs from
        :return: Text without URLs
        """
        return re.sub(
            r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|%[0-9a-fA-F][0-9a-fA-F])+",
            "",
            text,
        )

    def truncate_chat_context(self) -> None:
        """
        Truncate the chat context if it is too long.
        """
        while self.count_tokens(self.messages_storage.messages) > self.bot_config.context_max_tokens:
            self.messages_storage.delete_message(0)
            logging.debug(f"Chat context truncated for chat {self.messages_storage.chat_id}")

    async def clean_context(self) -> None:
        """
        Clean the chat context.
        """
        await self.messages_storage.clear_messages()
        if self.documents_storage:
            await self.documents_storage.clear(self.messages_storage.chat_id)
        logging.debug(f"Chat context and documents cleaned for chat {self.messages_storage.chat_id}")

    async def answer_message(self, chat_id: int, message: str) -> BaseMessage:
        """
        Processes a text message and returns the LLM's response.

        :param chat_id: The ID of the chat.
        :param message: The text of the message to process.
        :return: The response message from the LLM.
        """
        self.messages_storage.add_message(HumanMessage(content=message))
        self.truncate_chat_context()
        config = self._get_langchain_config(chat_id)
        ai_msg = await self.llm.ainvoke(self.system_instructions + self.messages_storage.messages)
        if ai_msg.tool_calls:
            self.messages_storage.add_message(ai_msg)
            for tool_call in ai_msg.tool_calls:
                selected_tool = get_tool(tool_call["name"], self.bot_config, self.documents_storage)
                tool_msg = await selected_tool.ainvoke(tool_call, config=config)
                self.messages_storage.add_message(tool_msg)
            ai_msg = await self.llm.ainvoke(self.system_instructions + self.messages_storage.messages, config=config)
        return ai_msg

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
                image_bytes = await self._download_file(image, session)
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
                response = await self.llm.ainvoke(
                    self.messages_storage.messages,
                    config=self._get_langchain_config(chat_id),
                )
        except (aiohttp.ClientError, Exception) as e:
            if isinstance(e, aiohttp.ClientError):
                logging.error(f"Failed to get image: {image}")
            logging.exception(e)
            response = BaseMessage(content="NO_ANSWER", type="text")

        logging.debug(f"Image message response: {response}")
        return response

    async def answer_voice_message(self, chat_id: int, text: str, audio: str):
        """
        Answer a voice message.
        :param chat_id: Chat ID
        :param audio: Voice message audio
        """
        logging.debug(f"Voice message: {audio}")

        try:
            async with aiohttp.ClientSession() as session:
                audio_bytes = await self._download_file(audio, session)
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
                            # "data": audio_bytes,
                            "data": audio_data,
                            # "file_uri": audio,
                        },
                    ]
                )
                self.messages_storage.add_message(llm_message)
                self.truncate_chat_context()
                response = await self.llm.ainvoke(
                    self.system_instructions + self.messages_storage.messages,
                    config=self._get_langchain_config(chat_id),
                )
        except (aiohttp.ClientError, Exception) as e:
            if isinstance(e, aiohttp.ClientError):
                logging.error(f"Failed to get audio: {audio}")
            logging.exception(e)
            response = BaseMessage(content="NO_ANSWER", type="text")

        logging.debug(f"Voice message response: {response}")
        return response

    async def _process_and_store_document(self, chat_id: int, document_url: str, filename: str) -> tuple[str, str]:
        """
        Downloads, processes, and stores a document.

        :param chat_id: Chat ID
        :param document_url: URL to download the document
        :param filename: Original filename
        :return: A tuple of (doc_key, extracted_text)
        :raises UnsupportedFileError: If the format is not supported
        :raises ValueError: If the document is too large
        """
        extension = filename.split(".")[-1].lower()
        if extension not in DocumentLoader.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(f"Extension .{extension} is not supported")

        async with aiohttp.ClientSession() as session:
            file_content = await self._download_file(
                document_url, session, size_limit=self.bot_config.max_document_size
            )

        # Extract text
        loader = DocumentLoader()
        extracted_text = loader.extract_text(file_content, filename)
        extracted_text = clean_text(extracted_text)

        # Generate a unique key for the document to avoid collisions
        doc_key = generate_document_key(filename)

        # Store document text
        if self.documents_storage:
            await self.documents_storage.store(chat_id, doc_key, extracted_text)

        return doc_key, extracted_text

    async def answer_document_message(self, chat_id: int, text: str, document_url: str, filename: str) -> BaseMessage:
        """
        Answer a document message.

        :param chat_id: Chat ID
        :param text: Text to answer (user prompt/caption)
        :param document_url: URL to download the document
        :param filename: Original filename
        :return: Response
        """
        logging.debug(f"Document message: {filename}")

        try:
            doc_key, extracted_text = await self._process_and_store_document(chat_id, document_url, filename)

            # For LLMBot (simple), we "stuff" the context for this turn only
            # but we don't save the massive text to chat history
            prompt = (
                f"The user uploaded a document: {filename} (ID: {doc_key})\n"
                f"--- DOCUMENT START ---\n"
                f"{extracted_text}\n"
                f"--- DOCUMENT END ---\n\n"
                f"User message: {text}"
            )

            # Add a pointer message to history instead of full text
            pointer_message = HumanMessage(
                content=f"User uploaded a document: {filename}. "
                f"Use the read_document tool with filename '{doc_key}' to access it."
            )
            self.messages_storage.add_message(pointer_message)
            self.truncate_chat_context()

            # We invoke with the stuffed prompt for this specific turn
            messages = self.system_instructions + self.messages_storage.messages[:-1] + [HumanMessage(content=prompt)]

            response = await self.llm.ainvoke(
                messages,
                config=self._get_langchain_config(chat_id),
            )

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

        except Exception as e:
            logging.error(f"Error processing document {filename}: {e}")
            logging.exception(e)
            response = BaseMessage(content="NO_ANSWER", type="text")

        return response

    async def postprocess_response(self, response: BaseMessage, message_text: str, chat_id: int) -> dict | None:
        """
        Postprocess the response from the LLM.
        :param response: Response from the LLM
        :param message_text: Text of the user message
        :param chat_id: Chat ID
        return: Final response data
        """

        # response.content is sometimes a list instead of a string, TODO: find why this happens and fix it
        if isinstance(response.content, list):
            response_content = ""
            for i, content_item in enumerate(response.content):
                if isinstance(content_item, str):
                    response_content += content_item
                else:
                    response_content += content_item.get("text", "")
                if i + 1 != len(response.content):
                    response_content += "\n\n"
        else:
            response_content = response.content

        final_response = None
        if response_content.startswith("GENERATE_IMAGE"):
            logging.debug(f"GENERATE_IMAGE response, generating image for chat {chat_id}")
            image = await self.generate_image(response_content[len("GENERATE_IMAGE ") :])
            if image:
                final_response = {
                    "type": "image",
                    "data": image,
                }
        elif "WEBCONTENT_RESUME" in response_content:
            logging.debug(f"WEBCONTENT_RESUME response, generating web content abstract for chat {chat_id}")
            response_content = await self.answer_webcontent(message_text, response_content)
            # TODO: find a way to graciously handle failed web content requests
            response_content = response_content if response_content else "😐"
            final_response = {"type": "text", "data": response_content}
        elif "WEBCONTENT_OPINION" in response_content:
            logging.debug(f"WEBCONTENT_OPINION response, generating web content opinion for chat {chat_id}")
            response_content = await self.answer_webcontent(message_text, response_content)
            # TODO: find a way to graciously handle failed web content requests
            response_content = response_content if response_content else "😐"
            final_response = {"type": "text", "data": response_content}
        elif "NO_ANSWER" not in response_content:
            logging.debug(f"Response for chat {chat_id}")
            final_response = {"type": "text", "data": response_content}
        else:
            logging.debug(f"NO_ANSWER response for chat {chat_id}")
            final_response = {
                "type": "text",
                "data": secrets.choice(["😐", "😶", "😳", "😕", "😑"]),
            }

        self.messages_storage.add_message(AIMessage(content=response_content))

        return final_response

    async def answer_webcontent(self, message_text: str, response_content: str) -> str | None:
        """
        Answer a web content message.
        :param message_text: Text to answer
        :param response_content: Response content
        :param chat_id: Chat ID
        :return: New response content if the call was successful, None otherwise
        """
        url = self._extract_url(response_content)
        try:
            if url:
                logging.debug(f"Obtaining web content for {url} using pseudotool")

                loader = WebBaseLoader(web_path=url)

                # alazy_load returns an async iterator of Document objects
                docs = []
                async for doc in loader.alazy_load():
                    docs.append(doc)

                template = self._remove_urls(message_text) + "\n" + '"{text}"'
                prompt = PromptTemplate.from_template(template)
                logging.debug(f"Web content prompt: {prompt}")

                self.truncate_chat_context()

                # TODO: Add full chat context
                stuff_chain = create_stuff_documents_chain(
                    llm=self.llm, prompt=prompt, document_variable_name="text", output_parser=StrOutputParser()
                )

                # The key should match the document_variable_name parameter
                response = await stuff_chain.ainvoke({"text": docs})
                logging.debug(f"Web content response: {response}")
                return response
            else:
                logging.debug(f"No URL found for web content: {message_text}")
        except aiohttp.ClientError as e:
            logging.error("Connection error connecting to web content")
            logging.exception(e)
            error_prompt = (
                f"Generate a brief response in {self.bot_config.preferred_language} "
                f"explaining that you couldn't connect to the webpage {url}. "
                f"Suggest checking the URL or trying again later. "
                f"Keep your response under 150 characters and maintain your character's style."
            )
            return await self.generate_feedback_message(error_prompt)
        except TimeoutError as e:
            logging.error("Timeout error connecting to web content")
            logging.exception(e)
            error_prompt = (
                f"Generate a brief response in {self.bot_config.preferred_language} "
                f"explaining that the webpage {url} took too long to respond. "
                f"Suggest it might be unavailable or too large. "
                f"Keep your response under 150 characters and maintain your character's style."
            )
            return await self.generate_feedback_message(error_prompt)
        except Exception as e:
            logging.error("Error connecting to web content")
            logging.exception(e)
            error_prompt = (
                f"Generate a brief response in {self.bot_config.preferred_language} "
                f"explaining that you had trouble processing the webpage {url}. "
                f"Suggest trying again later or trying a different URL. "
                f"Keep your response under 150 characters and maintain your character's style."
            )
            return await self.generate_feedback_message(error_prompt)
        return None

    async def call_sdapi(self, prompt: str) -> dict | None:
        """
        Call the StableDiffusion API.
        :param prompt: The prompt to send to the StableDiffusion API.
        :return: The response from the StableDiffusion API.
        """
        if self.bot_config.sdapi_url:
            try:
                params = self.bot_config.sdapi_params.copy()
                params["prompt"] = prompt
                if self.bot_config.sdapi_negative_prompt:
                    params["negative_prompt"] = self.bot_config.sdapi_negative_prompt

                # Use aiohttp for async HTTP requests
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        urljoin(self.bot_config.sdapi_url, "/sdapi/v1/txt2img"),
                        json=params,
                    ) as response:
                        if response.status == 200:
                            return await response.json()
            except Exception as e:
                logging.error("Failed to call SDAPI")
                logging.exception(e)
        return None

    async def generate_image(self, prompt: str) -> str | None:
        """
        Generate an image.
        :param prompt: Prompt to generate the image
        :return: Image representation in base64 format if the call was successful, None otherwise
        """
        logging.debug(f"Generate image: {prompt}")
        response = await self.call_sdapi(prompt)
        if response and "images" in response:
            return response["images"][0]
        return None

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        """
        Count the number of tokens in the messages.
        :param messages: List of messages
        :return: Number of tokens
        """
        extra_tokens = 0
        context_text = ""
        for message in messages:
            if isinstance(message.content, list):
                for item in message.content:
                    if item.get("type") == "text":
                        context_text += "\n " + item.get("text")
                    elif item.get("type") == "image_url":
                        # TODO: Use an LLM-based method to get the image token count.
                        extra_tokens += 258  # using gemini image context size
            else:
                context_text += "\n " + message.content

        return self.llm.get_num_tokens(context_text) + extra_tokens

    async def generate_feedback_message(self, prompt: str, max_length: int = 200, chat_id: int | None = None) -> str:
        """
        Generate a feedback message using the LLM.

        :param prompt: Prompt to generate the feedback message
        :param max_length: Maximum length of the feedback message
        :param chat_id: Optional chat ID for metadata
        :return: Generated feedback message
        """
        logging.debug("Generating feedback message")

        # Create a simple message list with just the prompt
        messages = [HumanMessage(content=prompt)]
        config = self._get_langchain_config(chat_id) if chat_id else None
        response = await self.llm.ainvoke(messages, config=config)

        # Clean up the response if needed
        feedback_message = response.content.strip()

        # Ensure the message isn't too long
        if len(feedback_message) > max_length:
            feedback_message = feedback_message[: max_length - 3] + "..."

        logging.debug(f"Generated feedback message: {feedback_message}")
        return feedback_message

    def _get_time_from_wpm(self, text: str, wpm: float) -> float:
        """
        Get the time it takes to write a text with a given WPM.
        :param text: Text to write
        :param wpm: Words per minute
        :return: Time in seconds
        """
        return (len(text.split()) / wpm) * 60

    def _load_tools(self) -> None:
        tools_to_bind = (
            self.tools
            if self.tools is not None
            else get_tools(self.bot_config, document_storage=self.documents_storage)
        )
        self.llm = self.llm.bind_tools(tools_to_bind)  # add wikipedia?
