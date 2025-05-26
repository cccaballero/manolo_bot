import base64
import logging
import random
import re
from typing import Any
from urllib.parse import urljoin

import requests
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from requests import ConnectTimeout, ReadTimeout, RequestException

from ai.tools import get_tool, get_tools
from config import Config


class LLMBot:
    def __init__(self, config: Config, system_instructions: list[BaseMessage]):
        self.config = config
        self.system_instructions = system_instructions
        # self.messages_buffer = messages_buffer
        self.llm = None
        self.chats: dict = {}  #  {'chat_id': {"messages": []}}
        self._load_llm()

        if self.config.use_tools:
            self._load_tools()

    def _get_rate_limiter(self):
        return InMemoryRateLimiter(
            requests_per_second=self.config.rate_limiter_requests_per_second,
            check_every_n_seconds=self.config.rate_limiter_check_every_n_seconds,
            max_bucket_size=self.config.rate_limiter_max_bucket_size,
        )

    def _get_chat_ollama(self):
        return ChatOllama(model=self.config.ollama_model)

    def _get_chat_google_generativeai(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.config.google_api_model,
            safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            rate_limiter=self._get_rate_limiter(),
        )

    def _get_chat_openai(self) -> ChatOpenAI:
        api_key = self.config.openai_api_key if self.config.openai_api_key else "not-needed"
        base_url = self.config.openai_api_base_url
        model = self.config.openai_api_model
        params = {
            "openai_api_key": api_key,
        }
        if base_url:
            params["base_url"] = base_url
        if model:
            params["model"] = model
        return ChatOpenAI(temperature=0.0, rate_limiter=self._get_rate_limiter(), **params)

    def _extract_url(self, text: str) -> str | None:
        """
        Extract the URL from the text.
        :param text: Text to extract the URL from
        :return: URL if found, None otherwise
        """
        url = re.search(r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|%[0-9a-fA-F][0-9a-fA-F])+", text)
        return url.group(0) if url else None

    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from the text.
        :param text: Text to remove URLs from
        :return: Text without URLs
        """
        return re.sub(r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|%[0-9a-fA-F][0-9a-fA-F])+", "", text)

    def _load_llm(self):
        if self.config.ollama_model:
            self.llm = self._get_chat_ollama()
        elif self.config.google_api_key:
            self.llm = self._get_chat_google_generativeai()
        elif self.config.openai_api_key or self.config.openai_api_base_url:
            self.llm = self._get_chat_openai()
        else:
            raise Exception("No LLM backend data found")

    def add_chat(self, chat_id: int) -> None:
        """
        Add a new chat to the list of chats.
        :param chat_id: Chat ID
        """
        if chat_id not in self.chats:
            self.chats[chat_id] = {
                "messages": [],
            }

    def truncate_chat_context(self, chat_id: int) -> None:
        """
        Truncate the chat context if it is too long.
        :param chat_id: Chat ID
        """
        while self.count_tokens(self.chats[chat_id]["messages"]) > self.config.context_max_tokens:
            self.chats[chat_id]["messages"] = self.chats[chat_id]["messages"][1:]
            logging.debug(f"Chat context truncated for chat {chat_id}")

    def call_sdapi(self, prompt: str) -> dict[str, Any] | None:
        """
        Call the StableDiffusion API.
        :param prompt: The prompt to send to the StableDiffusion API.
        :return: The response from the StableDiffusion API.
        """
        if self.config.sdapi_url:
            try:
                params = self.config.sdapi_params.copy()
                params["prompt"] = prompt
                if self.config.sdapi_negative_prompt:
                    params["negative_prompt"] = self.config.sdapi_negative_prompt
                response = requests.post(urljoin(self.config.sdapi_url, "/sdapi/v1/txt2img"), json=params)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                logging.error("Failed to call SDAPI")
                logging.exception(e)
        return None

    def clean_context(self, chat_id: int) -> None:
        """
        Clean the chat context.
        :param chat_id: Chat ID
        """
        self.chats[chat_id]["messages"] = []
        logging.debug(f"Chat context cleaned for chat {chat_id}")

    def answer_message(self, chat_id: int, message: str) -> BaseMessage:
        self.chats[chat_id]["messages"].append(HumanMessage(content=message))
        self.truncate_chat_context(chat_id)
        ai_msg = self.llm.invoke(self.system_instructions + self.chats[chat_id]["messages"])
        if ai_msg.tool_calls:
            self.chats[chat_id]["messages"].append(ai_msg)
            for tool_call in ai_msg.tool_calls:
                selected_tool = get_tool(tool_call["name"])
                tool_msg = selected_tool.invoke(tool_call)
                self.chats[chat_id]["messages"].append(tool_msg)
            ai_msg = self.llm.invoke(self.system_instructions + self.chats[chat_id]["messages"])
        return ai_msg

    def answer_image_message(self, chat_id: int, text: str, image: str) -> BaseMessage:
        """
        Answer an image message.
        :param chat_id: Chat ID
        :param text: Text to answer
        :param image: Image to answer
        :return: Response
        """
        logging.debug(f"Image message: {text}")

        try:
            image_data = base64.b64encode(requests.get(image).content).decode("utf-8")
            llm_message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": text,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ]
            )
            self.chats[chat_id]["messages"].append(llm_message)
            self.truncate_chat_context(chat_id)
            response = self.llm.invoke(self.chats[chat_id]["messages"])
        except (RequestException, Exception) as e:
            if isinstance(e, RequestException):
                logging.error(f"Failed to get image: {image}")
            logging.exception(e)
            response = BaseMessage(content="NO_ANSWER", type="text")

        logging.debug(f"Image message response: {response}")
        return response

    def postprocess_response(self, response: BaseMessage, message_text: str, chat_id: int) -> dict | None:
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
            for content_item in response.content:
                response_content += f"\n\n{content_item}"
        else:
            response_content = response.content

        final_response = None
        if response_content.startswith("GENERATE_IMAGE"):
            logging.debug(f"GENERATE_IMAGE response, generating image for chat {chat_id}")
            image = self.generate_image(response_content[len("GENERATE_IMAGE ") :])
            if image:
                final_response = {
                    "type": "image",
                    "content": image,
                }
        elif "WEBCONTENT_RESUME" in response_content:
            logging.debug(f"WEBCONTENT_RESUME response, generating web content abstract for chat {chat_id}")
            response_content = self.answer_webcontent(message_text, response_content, chat_id)
            # TODO: find a way to graciously handle failed web content requests
            response_content = response_content if response_content else "😐"
            final_response = {"type": "text", "data": response_content}
        elif "WEBCONTENT_OPINION" in response_content:
            logging.debug(f"WEBCONTENT_OPINION response, generating web content opinion for chat {chat_id}")
            response_content = self.answer_webcontent(message_text, response_content, chat_id)
            # TODO: find a way to graciously handle failed web content requests
            response_content = response_content if response_content else "😐"
            final_response = {"type": "text", "data": response_content}
        elif "NO_ANSWER" not in response_content:
            logging.debug(f"Response for chat {chat_id}")
            final_response = {"type": "text", "data": response_content}
        else:
            logging.debug(f"NO_ANSWER response for chat {chat_id}")
            final_response = {"type": "text", "text": random.choice(["😐", "😶", "😳", "😕", "😑"])}

        self.chats[chat_id]["messages"].append(AIMessage(content=response_content))

        return final_response

    def generate_image(self, prompt: str) -> str | None:
        """
        Generate an image.
        :param prompt: Prompt to generate the image
        :return: Image representation in base64 format if the call was successful, None otherwise
        """
        logging.debug(f"Generate image: {prompt}")
        response = self.call_sdapi(prompt)
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

    def answer_webcontent(self, message_text: str, response_content: str, chat_id: int) -> str | None:
        """
        Answer a web content message.
        :param message_text: Text to answer
        :param response_content: Response content
        :return: New response content if the call was successful, None otherwise
        """
        try:
            url = self._extract_url(response_content)
            if url:
                logging.debug(f"Obtaining web content for {url}")
                
                # Configure WebBaseLoader with timeout settings
                loader = WebBaseLoader(url)
                loader.requests_kwargs = {
                    'timeout': self.config.web_content_request_timeout
                }
                
                docs = loader.load()
                
                template = self._remove_urls(message_text) + "\n" + '"{text}"'
                prompt = PromptTemplate.from_template(template)
                logging.debug(f"Web content prompt: {prompt}")

                self.truncate_chat_context(chat_id)

                # TODO: Add full chat context
                stuff_chain = create_stuff_documents_chain(
                    llm=self.llm, prompt=prompt, document_variable_name="text", output_parser=StrOutputParser()
                )

                # The key should match the document_variable_name parameter
                response = stuff_chain.invoke({"text": docs})
                logging.debug(f"Web content response: {response}")
                return response
            else:
                logging.debug(f"No URL found for web content: {message_text}")
        except ConnectionError as e:
            logging.error("Connection error connecting to web content")
            logging.exception(e)
            error_prompt = f"Generate a brief response in {self.config.preferred_language} explaining that you couldn't connect to the webpage {url}. Suggest checking the URL or trying again later. Keep your response under 150 characters and maintain your character's style."
            return self.generate_feedback_message(error_prompt)
        except ReadTimeout as e:
            logging.error("Read timeout error connecting to web content")
            logging.exception(e)
            error_prompt = f"Generate a brief response in {self.config.preferred_language} explaining that the webpage {url} took too long to send data. Suggest it might be unavailable or too large. Keep your response under 150 characters and maintain your character's style."
            return self.generate_feedback_message(error_prompt)
        except ConnectTimeout as e:
            logging.error("Timeout error connecting to web content")
            logging.exception(e)
            error_prompt = f"Generate a brief response in {self.config.preferred_language} explaining that the webpage {url} took too long to respond. Suggest it might be unavailable or too large. Keep your response under 150 characters and maintain your character's style."
            return self.generate_feedback_message(error_prompt)
        except Exception as e:
            logging.error("Error connecting to web content")
            logging.exception(e)
            error_prompt = f"Generate a brief response in {self.config.preferred_language} explaining that you had trouble processing the webpage {url}. Suggest trying again later or trying a different URL. Keep your response under 150 characters and maintain your character's style."
            return self.generate_feedback_message(error_prompt)
        return None

    def generate_feedback_message(self, prompt: str, max_length: int = 200) -> str:
        """
        Generate a feedback message using the LLM.

        :param prompt: Prompt to generate the feedback message
        :param max_length: Maximum length of the feedback message
        :return: Generated feedback message
        """
        logging.debug("Generating feedback message")

        # Create a simple message list with just the prompt
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

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

    def _load_tools(self):
        self.llm = self.llm.bind_tools(get_tools())  # add wikipedia?
