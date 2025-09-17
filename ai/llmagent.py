import base64
import logging

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from requests import RequestException

from ai.llmbot import LLMBot
from ai.tools import get_tools
from config import Config


class LLMAgent(LLMBot):
    def __init__(self, config: Config, system_instructions: list[BaseMessage]):
        super().__init__(config, system_instructions)

        self.agent = create_react_agent(
            model=self.llm,
            tools=get_tools(),
            # A static prompt that never changes
            # prompt=self.system_instructions[0],
        )

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

    def answer_message(self, chat_id: int, message: str) -> BaseMessage:
        self.chats[chat_id]["messages"].append(HumanMessage(content=message))
        self.truncate_chat_context(chat_id)

        ai_msg = self.agent.invoke({"messages": self.system_instructions + self.chats[chat_id]["messages"]})
        return ai_msg["messages"][-1]

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
            # Use the shared session to download the image
            response = self._session.get(image, timeout=self.config.web_content_request_timeout)
            response.raise_for_status()
            image_data = base64.b64encode(response.content).decode("utf-8")

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
            response = self.agent.invoke({"messages": self.chats[chat_id]["messages"]})["messages"][-1]
        except (RequestException, Exception) as e:
            if isinstance(e, RequestException):
                logging.error(f"Failed to get image: {image}")
            logging.exception(e)
            response = BaseMessage(content="NO_ANSWER", type="text")

        logging.debug(f"Image message response: {response}")
        return response
