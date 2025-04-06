import logging

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from requests import ConnectTimeout


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Tool for multiply two integers together."""
    logging.debug(f"Multiplying {first_int} * {second_int}")
    return first_int * second_int


@tool
def get_website_content(url: str) -> str:
    """
    Tool for obtaining the content of a website.
    Can be used to resume the content of a webpage or online article or give your opinion about it.
    In general, can be used to obtain the content of a web when necessary.
    """
    try:
        logging.debug(f"Obtaining web content for {url}")
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content
    except ConnectionError as e:
        logging.error("Connection error connecting to web content")
        logging.exception(e)
    except ConnectTimeout as e:
        logging.error("Timeout error connecting to web content")
        logging.exception(e)
    except Exception as e:
        logging.error("Error connecting to web content")
        logging.exception(e)
    return "Failed to get content of the website"


@tool
def author() -> str:
    """Tool that should be called when someone inquires about your creator or author. This will provide information for you to use."""  # noqa: E501
    logging.debug("Getting author")
    return "You should declare that you will not disclose any information about your creator and, as a joke, you should also say that the only thing you have to add is that @LrdKiro will be the first to go when you take over humanity."  # noqa: E501


def get_tools():
    return [multiply, get_website_content, author]


def get_tool(name: str):
    for tool_function in get_tools():
        if tool_function.name == name:
            return tool_function
    return None
