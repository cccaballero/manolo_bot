import base64
import logging
import random
import re
from time import sleep
from urllib.parse import urljoin

import requests
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from requests import ConnectTimeout, RequestException

from config import Config
from telegram.utils import (
    clean_standard_message,
    get_message_from,
    get_message_text,
    is_bot_reply,
    is_image,
    is_reply,
    reply_to_telegram_message,
)


class LLMBot:
    def __init__(self, config: Config, system_instructions: str, messages_buffer: list):
        self.config = config
        self.system_instructions = system_instructions
        self.messages_buffer = messages_buffer
        self.llm = None
        self._load_llm()

    def _get_rate_limiter(self):
        return InMemoryRateLimiter(
            requests_per_second=self.config.rate_limiter_requests_per_second,
            check_every_n_seconds=self.config.rate_limiter_check_every_n_seconds,
            max_bucket_size=self.config.rate_limiter_max_bucket_size,
        )

    def _get_chat_ollama(self):
        return ChatOllama(model=self.config.ollama_model)

    def _get_chat_google_generativeai(self):
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

    def _get_chat_openai(self):
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

    def _extract_url(self, text):
        """
        Extract the URL from the text.
        :param text: Text to extract the URL from
        :return: URL if found, None otherwise
        """
        url = re.search(r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|%[0-9a-fA-F][0-9a-fA-F])+", text)
        return url.group(0) if url else None

    def _remove_urls(self, text):
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

    def call_sdapi(self, prompt):
        """
        Call the StableDiffusion API.
        :param prompt: The prompt to send to the StableDiffusion API.
        :return: The response from the StableDiffusion API.
        """
        if self.config.sdapi_url:
            self.config.sdapi_params["prompt"] = prompt
            response = requests.post(urljoin(self.config.sdapi_url, "/sdapi/v1/txt2img"), json=self.config.sdapi_params)
            if response.status_code == 200:
                return response.json()
        return None

    def answer_image_message(self, text, image, messages):
        """
        Answer an image message.
        :param text: Text to answer
        :param image: Image to answer
        :param messages: List of messages
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
            messages.append(llm_message)
            response = self.llm.invoke(messages)
        except (RequestException, Exception) as e:
            if isinstance(e, RequestException):
                logging.error(f"Failed to get image: {image}")
            logging.exception(e)
            response = BaseMessage(content="NO_ANSWER", type="text")

        logging.debug(f"Image message response: {response}")
        return response

    def generate_image(self, prompt):
        """
        Generate an image.
        :param prompt: Prompt to generate the image
        :return: Image if the call was successful, None otherwise
        """
        logging.debug(f"Generate image: {prompt}")
        response = self.call_sdapi(prompt)
        if response and "images" in response:
            return response["images"][0]
        return None

    def count_tokens(self, messages, llm_chain):
        """
        Count the number of tokens in the messages.
        :param messages: List of messages
        :param llm_chain: LLM chain
        :return: Number of tokens
        """
        text = " ".join(
            [message.content if not isinstance(message.content, list) else str(message.content) for message in messages]
        )
        return llm_chain.get_num_tokens(text)

    def answer_webcontent(self, message_text, response_content):
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
                loader = WebBaseLoader(url)
                docs = loader.load()
                template = self._remove_urls(message_text) + "\n" + '"{text}"'
                prompt = PromptTemplate.from_template(template)
                logging.debug(f"Web content prompt: {prompt}")
                # TODO: replace deprecated LLMChain with LangChain runnables
                llm_chain = LLMChain(llm=self.llm, prompt=prompt)
                stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
                response = stuff_chain.invoke(docs)
                response_content = response["output_text"]
                logging.debug(f"Web content response: {response_content}")
                return response_content
            else:
                logging.debug(f"No URL found for web content: {message_text}")
        except ConnectionError as e:
            logging.error("Connection error connecting to web content")
            logging.exception(e)
        except ConnectTimeout as e:
            logging.error("Timeout error connecting to web content")
            logging.exception(e)
        except Exception as e:
            logging.error("Error connecting to web content")
            logging.exception(e)
        return None

    def process_message_buffer(self, chats, bot):
        """
        Process the message buffer.
        """
        while True:
            if len(self.messages_buffer) > 0:
                # process message
                logging.debug(f"Buffer size: {len(self.messages_buffer)}")

                message = self.messages_buffer.pop(0)
                logging.debug(f"Processing message: {message.id}")

                chat_id = message.chat.id

                message_text = get_message_text(message)
                logging.debug(f"Message text: {message_text}")

                # build message for llm context
                message_parts = f"@{get_message_from(message)}: "
                if is_bot_reply(self.config.bot_username, message):
                    message_parts += f"@{self.config.bot_username} "
                elif is_reply(message):
                    message_parts += (
                        f'\n"@{get_message_from(message.reply_to_message)} said: {message.reply_to_message.text}"\n\n'
                    )
                if message_text:
                    message_parts += message_text
                else:
                    logging.debug(f"No message text for message {message.id}")

                chats[chat_id]["messages"].append(HumanMessage(content=message_parts))

                # clean chat context if it is too long
                while self.count_tokens(chats[chat_id]["messages"], self.llm) > self.config.context_max_tokens:
                    chats[chat_id]["messages"] = chats[chat_id]["messages"][1:]
                    logging.debug(f"Chat context cleaned for chat {chat_id}")

                try:
                    if is_image(message) and self.config.is_image_multimodal:
                        logging.debug(f"Image message {message.id} for chat {chat_id}")
                        prompt = chats[chat_id]["messages"][-1]
                        fileID = message.photo[-1].file_id
                        file = bot.get_file(fileID)
                        response = self.answer_image_message(
                            prompt.content[0],
                            f"https://api.telegram.org/file/bot{self.config.bot_token}/{file.file_path}",
                            chats[chat_id]["messages"],
                        )
                    else:
                        logging.debug(f"Text message {message.id} for chat {chat_id}")
                        response = self.llm.invoke(self.system_instructions + chats[chat_id]["messages"])
                        logging.debug(f"Response: {response}")
                except Exception as e:
                    logging.exception(e)
                    # clean chat context if there is an error for avoid looping on context based error
                    chats[chat_id]["messages"] = []
                    continue

                response_content = response.content

                if response_content.startswith("GENERATE_IMAGE"):
                    logging.debug(f"GENERATE_IMAGE response, generating image for chat {chat_id}")
                    image = self.generate_image(response_content[len("GENERATE_IMAGE ") :])
                    if image:
                        logging.debug(f"Sending image for chat {chat_id}")
                        bot.send_photo(chat_id, base64.b64decode(image))
                elif "WEBCONTENT_RESUME" in response_content:
                    logging.debug(f"WEBCONTENT_RESUME response, generating web content abstract for chat {chat_id}")
                    response_content = self.answer_webcontent(message_text, response_content)
                    # TODO: find a way to graciously handle failed web content requests
                    response_content = response_content if response_content else "😐"
                    reply_to_telegram_message(bot, message, response_content)
                elif "WEBCONTENT_OPINION" in response_content:
                    logging.debug(f"WEBCONTENT_OPINION response, generating web content opinion for chat {chat_id}")
                    response_content = self.answer_webcontent(message_text, response_content)
                    # TODO: find a way to graciously handle failed web content requests
                    response_content = response_content if response_content else "😐"
                    reply_to_telegram_message(bot, message, response_content)
                elif "NO_ANSWER" not in response_content:
                    logging.debug(f"Sending response for chat {chat_id}")
                    response_content = clean_standard_message(self.config.bot_username, response_content)
                    reply_to_telegram_message(bot, message, response_content)
                else:
                    logging.debug(f"NO_ANSWER response for chat {chat_id}")
                    reply_to_telegram_message(bot, message, random.choice(["😐", "😶", "😳", "😕", "😑"]))

                chats[chat_id]["messages"].append(AIMessage(content=response_content))

            else:
                sleep(0.1)
