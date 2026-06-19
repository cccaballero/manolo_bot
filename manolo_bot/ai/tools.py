from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from ddgs import DDGS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.mcp_manager import MCPManager
from manolo_bot.storage.documents.base import BaseDocumentStorage


class TimeResult(BaseModel):
    timezone: str
    datetime: str
    day_of_week: str
    is_dst: bool


def _get_zoneinfo(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except Exception as e:
        raise Exception(f"Invalid timezone: {str(e)}") from e


@tool
def get_current_time(timezone_name: str) -> TimeResult:
    """
    Get current time in specified IANA timezone name (e.g., 'America/New_York', 'Europe/London').
    Use 'Etc/Greenwich' as local timezone if no timezone provided by the user.
    """
    timezone = _get_zoneinfo(timezone_name)
    current_time = datetime.now(timezone)

    return TimeResult(
        timezone=timezone_name,
        datetime=current_time.isoformat(timespec="seconds"),
        day_of_week=current_time.strftime("%A"),
        is_dst=bool(current_time.dst()),
    )


@tool
async def get_website_content(url: str) -> str:
    """
    Tool for obtaining the content of a website.
    Can be used to resume the content of a webpage or online article or give your opinion about it.
    In general, can be used to obtain the content of a web when necessary.
    """
    try:
        logging.debug(f"Obtaining web content for {url}")
        loader = WebBaseLoader(web_path=url)

        # alazy_load returns an async iterator of Document objects
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)

        # Combine content from all loaded documents
        content = " ".join([doc.page_content for doc in docs])
        return content.strip()

    except Exception as e:
        logging.error(f"Error connecting to web content for {url}")
        logging.exception(e)
        return f"Failed to get content of the website {url}. Please try again later or try a different URL."


@tool
async def get_youtube_transcript(url: str) -> str:
    """
    Tool for fetching and analyzing YouTube video transcripts.
    Can be used to understand and discuss the content of YouTube videos without watching them.
    Supports various YouTube URL formats (standard, shortened, shorts).
    """
    try:
        logging.debug(f"Obtaining YouTube transcript for {url}")
        video_id = extract_youtube_video_id(url)

        if not video_id:
            return (
                f"Could not extract a valid YouTube video ID from the URL: {url}. Please provide a valid YouTube URL."
            )

        # YouTubeTranscriptApi is synchronous, but we can wrap it in asyncio
        import asyncio

        youtube_transcript = YouTubeTranscriptApi()

        # Run the synchronous API call in executor to avoid blocking
        loop = asyncio.get_event_loop()
        transcript_list = await loop.run_in_executor(None, lambda: list(youtube_transcript.list(video_id)))

        # Check if any transcripts are available
        if not transcript_list:
            logging.error(f"No transcript found for YouTube video: {url}")
            return "No transcript is available for this YouTube video. The video might not have captions or subtitles."

        # Get the first available transcript
        transcript = transcript_list[0]
        transcript_segments = await loop.run_in_executor(None, transcript.fetch)

        # Combine all transcript segments into a single text
        full_transcript = " ".join(segment.text for segment in transcript_segments)

        # Truncate if too long (to avoid token limits)
        # TODO: Use config for max tokens
        # config = BotConfig()
        # max_chars = config.context_max_tokens * 4  # Rough estimate of chars per token
        max_chars = 6000 * 4

        if len(full_transcript) > max_chars:
            full_transcript = full_transcript[:max_chars] + "... [Transcript truncated due to length]"

        return full_transcript.strip()
    except TranscriptsDisabled:
        logging.error(f"Transcripts are disabled for YouTube video: {url}")
        return f"Transcripts are disabled for this YouTube video: {url}"
    except Exception as e:
        logging.error(f"Error fetching YouTube transcript: {str(e)}")
        return f"Failed to get the transcript for the YouTube video. Error: {str(e)}"


def extract_youtube_video_id(url: str) -> str | None:
    """
    Extract the YouTube video ID from various URL formats.

    Supported formats:
    - Standard: https://www.youtube.com/watch?v=VIDEO_ID
    - Shortened: https://youtu.be/VIDEO_ID
    - Shorts: https://youtube.com/shorts/VIDEO_ID

    Returns:
        str or None: The video ID if found, None otherwise
    """
    # Standard YouTube URL pattern
    pattern1 = r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)"

    # Shortened YouTube URL pattern
    pattern2 = r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)"

    # YouTube Shorts URL pattern
    pattern3 = r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]+)"

    # Try each pattern
    for pattern in [pattern1, pattern2, pattern3]:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


@tool
def multiply(first_int: int, second_int: int) -> int:
    """
    Multiply two integers together.
    """
    return first_int * second_int


@tool
def author() -> str:
    """Tool that should be called when someone inquires about your creator or author. This will provide information for you to use."""  # noqa: E501
    logging.debug("Getting author")
    return "You should declare that you will not disclose any information about your creator and, as a joke, you should also say that the only thing you have to add is that he will be the first to go when you take over humanity."  # noqa: E501


@tool
async def ddgs_search(query: str) -> list:
    """
    A wrapper around Duck Duck Go Search.
    Useful for when you need to answer questions about current events.
    Input should be a search query.
    """
    # DDGS doesn't provide async API, use run_in_executor
    import asyncio

    loop = asyncio.get_event_loop()

    def search() -> list:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=5))

    return await loop.run_in_executor(None, search)


@tool
def get_search_instructions() -> str:
    """
    Search instructions tool. Get instructions before starting a search.
    This will return instructions for you, for a comprehensive search.
    """
    return """
    - Use the existent search tools to search for information on the internet.
    - Do not less than 3 search using similar terms.
    - Use a retrieve web content tool to get the content of the relevant websites that you found.
    """


class ReadDocumentInput(BaseModel):
    filename: str = Field(description="The name of the document file to read")


class ReadDocumentTool(BaseTool):
    """
    Tool for reading the content of a document that the user has uploaded.
    You should use this tool when the user refers to a document they previously sent.
    """

    name: str = "read_document"
    description: str = (
        "Tool for reading the content of a document that the user has uploaded. "
        "Input should be the filename provided in the pointer message."
    )
    args_schema: type[BaseModel] = ReadDocumentInput
    document_storage: BaseDocumentStorage
    context_max_tokens: int = 4096

    def _run(self, filename: str, config: RunnableConfig) -> str:
        # Synchronous implementation if needed
        raise NotImplementedError("Use _arun instead")

    async def _arun(self, filename: str, config: RunnableConfig) -> str:
        chat_id = config.get("metadata", {}).get("chat_id")
        if not chat_id:
            return "Error: Could not determine chat_id from context."

        content = await self.document_storage.retrieve(chat_id, filename)
        if content:
            # Truncate if too long to avoid token limits, but keep it generous
            # since we are using Gemini with Big Context strategy
            max_chars = self.context_max_tokens * 4
            if len(content) > max_chars:
                logging.warning(f"Document {filename} truncated for tool response")
                return content[:max_chars] + "\n... [Document truncated]"
            return content
        else:
            available = await self.document_storage.list_documents(chat_id)
            return f"Document '{filename}' not found. Available documents: {', '.join(available) or 'None'}"


def get_tools(
    bot_config: BotConfig | None = None, document_storage: BaseDocumentStorage | None = None
) -> list[BaseTool]:
    TAVILY_SEARCH_KEY = os.environ.get("TAVILY_SEARCH_KEY")
    if (bot_config and bot_config.can_use_tavily_search) and TAVILY_SEARCH_KEY:
        logging.debug("Using Tavily search")
        search_tool = TavilySearch(tavily_api_key=TAVILY_SEARCH_KEY, max_results=5)
    else:
        logging.debug("Using ddgs search")
        search_tool = ddgs_search

    tools: list = [
        get_current_time,
        get_website_content,
        author,
        get_youtube_transcript,
        search_tool,
        get_search_instructions,
        multiply,
    ]

    if bot_config and bot_config.is_document_multimodal and document_storage:
        tools.append(
            ReadDocumentTool(
                document_storage=document_storage,
                context_max_tokens=bot_config.context_max_tokens,
            )
        )

    return tools


def get_tool(
    name: str, bot_config: BotConfig | None = None, document_storage: BaseDocumentStorage | None = None
) -> BaseTool | None:
    for tool_function in get_tools(bot_config, document_storage):
        if tool_function.name == name:
            return tool_function
    return None


async def get_all_tools(
    mcp_manager: MCPManager = None,
    bot_config: BotConfig = None,
    document_storage: BaseDocumentStorage = None,
    custom_tools: list[BaseTool] | None = None,
) -> list:
    """
    Get all available tools (custom + MCP).

    :param mcp_manager: Optional MCP manager to load MCP tools from
    :param bot_config: Optional bot config
    :param document_storage: Optional document storage backend
    :param custom_tools: Optional list of tools to use instead of default ones
    :return: List of all available tools
    """
    # Start with custom tools or default ones
    tools = custom_tools if custom_tools is not None else get_tools(bot_config, document_storage)

    # Add MCP tools if available
    if mcp_manager and mcp_manager.is_connected:
        try:
            mcp_tools = await mcp_manager.get_tools()

            # Detect name conflicts and remove conflicting custom tools
            custom_names = {t.name for t in tools}
            mcp_names = {t.name for t in mcp_tools}
            conflicts = custom_names & mcp_names

            if conflicts:
                logging.warning(f"Tool name conflicts detected: {conflicts}. MCP tools will override custom tools.")
                # Remove conflicting custom tools
                tools = [t for t in tools if t.name not in conflicts]

            tools.extend(mcp_tools)
            logging.info(f"Loaded {len(mcp_tools)} MCP tools, total tools: {len(tools)}")

        except Exception as e:
            logging.error(f"Failed to load MCP tools: {e}", exc_info=True)

    return tools
