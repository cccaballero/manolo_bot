import base64
import logging
import re
import secrets
from types import TracebackType
from typing import TYPE_CHECKING

import aiohttp
from google.genai.types import HarmBlockThreshold, HarmCategory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from ai.config import BotConfig, LLMConfig
from ai.tools import get_tool, get_tools
from storage.base import BaseMessagesStorage

if TYPE_CHECKING:
    from ai.mcp_manager import MCPManager


class LLMBuilder:
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
    def __init__(
        self,
        llm: BaseChatModel,
        bot_config: BotConfig,
        system_instructions: list[BaseMessage],
        messages_storage: BaseMessagesStorage,
    ) -> None:
        self.bot_config = bot_config
        self.system_instructions = system_instructions
        # self.messages_buffer = messages_buffer
        self.llm = llm
        self.messages_storage: BaseMessagesStorage = messages_storage
        # self._load_llm()
        self._mcp_manager: MCPManager | None = None
        self._async_resources_initialized = False

        # if self.bot_config.use_tools:
        #     self._load_tools()
        self._load_tools()

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

    async def initialize_async_resources(self) -> None:
        """Initialize all async resources (MCP, etc.)."""
        if self._async_resources_initialized:
            return

        logging.debug("Initializing async resources...")

        # Initialize MCP if enabled
        if self.bot_config.enable_mcp:
            logging.info("Initializing MCP...")
            try:
                from ai.mcp_manager import MCPManager

                # TODO: We probably don't want to initialize MCP in each call, maybe we can cache this somehow?
                self._mcp_manager = MCPManager(self.bot_config)
                await self._mcp_manager.connect()
                logging.info("MCP initialized successfully")

                # Reload tools to include MCP tools
                # if self.bot_config.use_tools:
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
        from ai.tools import get_all_tools

        tools = await get_all_tools(self._mcp_manager, self.bot_config)
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
        logging.debug(f"Chat context cleaned for chat {self.messages_storage.chat_id}")

    async def answer_message(self, chat_id: int, message: str) -> BaseMessage:
        self.messages_storage.add_message(HumanMessage(content=message))
        self.truncate_chat_context()
        config = self._get_langchain_config(chat_id)
        ai_msg = await self.llm.ainvoke(self.system_instructions + self.messages_storage.messages)
        if ai_msg.tool_calls:
            self.messages_storage.add_message(ai_msg)
            for tool_call in ai_msg.tool_calls:
                selected_tool = get_tool(tool_call["name"])
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
                    "content": image,
                }
        elif "WEBCONTENT_RESUME" in response_content:
            logging.debug(f"WEBCONTENT_RESUME response, generating web content abstract for chat {chat_id}")
            response_content = await self.answer_webcontent(message_text, response_content, chat_id)
            # TODO: find a way to graciously handle failed web content requests
            response_content = response_content if response_content else "😐"
            final_response = {"type": "text", "data": response_content}
        elif "WEBCONTENT_OPINION" in response_content:
            logging.debug(f"WEBCONTENT_OPINION response, generating web content opinion for chat {chat_id}")
            response_content = await self.answer_webcontent(message_text, response_content, chat_id)
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
                "text": secrets.choice(["😐", "😶", "😳", "😕", "😑"]),
            }

        self.messages_storage.add_message(AIMessage(content=response_content))

        return final_response

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
        config = self._get_langchain_config(chat_id) if chat_id else {}
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
        self.llm = self.llm.bind_tools(get_tools(bot_config=self.bot_config))  # add wikipedia?
