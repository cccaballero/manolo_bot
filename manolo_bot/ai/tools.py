from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from ddgs import DDGS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilyExtract, TavilySearch
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


# Tavily Extract is optional for both fetching tools below. It fetches and
# extracts server-side, so it works when the bot's IP is blocked by the target
# (403 / YouTube IpBlocked) and handles binary documents (PDF, ...) that the
# HTML-only WebBaseLoader path cannot. The direct implementations stay as the
# fallback: these flags mean "prefer Tavily", never "Tavily only".
_TAVILY_EXTRACT_TIMEOUT_SECONDS = 30
# YouTube extracts carry the transcript as "[mm:ss] text" lines. A video
# without captions yields only title/channel/description, which must not be
# presented to the model as a transcript.
_YOUTUBE_TIMESTAMP_RE = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")


async def _tavily_extract(url: str, api_key: str) -> tuple[str | None, str]:
    """Extract text from ``url`` through Tavily Extract.

    Returns ``(text, reason)``. ``text`` is ``None`` when Tavily could not
    serve the request (failed result, HTTP error, timeout or any other
    exception) and the caller should use the direct path instead. ``reason``
    is a short, log-safe explanation; the API key is never included.
    """
    try:
        extractor = TavilyExtract(tavily_api_key=api_key, format="text", extract_depth="basic")
        raw = await asyncio.wait_for(extractor.ainvoke({"urls": [url]}), timeout=_TAVILY_EXTRACT_TIMEOUT_SECONDS)
    except Exception as e:
        return None, f"request failed ({type(e).__name__}: {e})"

    # TavilyExtract returns {"error": exc} for unexpected failures and a plain
    # string when the tool-level ToolException is handled.
    if not isinstance(raw, dict):
        return None, f"returned an error: {raw}"
    if raw.get("error"):
        return None, f"returned an error: {raw['error']}"

    results = raw.get("results") or []
    if not results:
        failed = raw.get("failed_results") or []
        return None, f"returned no results (failed_results={failed})"

    content = (results[0].get("raw_content") or "").strip()
    if not content:
        return None, "returned empty content"
    return content, ""


def _truncate_content(content: str, max_chars: int, notice: str) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + notice


class WebContentInput(BaseModel):
    url: str = Field(description="The URL of the website or document to fetch")


class WebContentTool(BaseTool):
    """
    Tool for obtaining the content of a website.
    Can be used to resume the content of a webpage or online article or give your opinion about it.
    In general, can be used to obtain the content of a web when necessary.
    """

    name: str = "get_website_content"
    description: str = (
        "Tool for obtaining the content of a website. "
        "Can be used to resume the content of a webpage or online article or give your opinion about it. "
        "In general, can be used to obtain the content of a web when necessary."
    )
    args_schema: type[BaseModel] = WebContentInput
    context_max_tokens: int = 6000
    tavily_api_key: str = ""
    use_tavily: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # URL document support (PDF, ...) only exists through Tavily Extract;
        # advertising it on the direct path makes the model retry files that
        # WebBaseLoader cannot read.
        if self.use_tavily and self.tavily_api_key:
            self.description = f"{self.description} Also works for documents served from a URL, such as PDFs."

    def _run(self, url: str) -> str:
        # Synchronous implementation if needed
        raise NotImplementedError("Use _arun instead")

    async def _arun(self, url: str) -> str:
        if self.use_tavily and self.tavily_api_key:
            content, reason = await _tavily_extract(url, self.tavily_api_key)
            if content is not None:
                logging.info(f"Tavily Extract served web content for {url}")
                return _truncate_content(
                    content,
                    self.context_max_tokens * 4,
                    "... [Content truncated due to length]",
                )
            logging.warning(f"Tavily Extract failed for {url} ({reason}); falling back to direct fetch")

        return await self._fetch_direct(url)

    async def _fetch_direct(self, url: str) -> str:
        try:
            logging.debug(f"Serving web content for {url} via WebBaseLoader (direct path)")
            loader = WebBaseLoader(web_path=url)

            # alazy_load returns an async iterator of Document objects
            docs = []
            async for doc in loader.alazy_load():
                docs.append(doc)

            # Combine content from all loaded documents
            content = " ".join([doc.page_content for doc in docs])
            return _truncate_content(
                content.strip(),
                self.context_max_tokens * 4,
                "... [Content truncated due to length]",
            )

        except Exception as e:
            logging.error(f"Error connecting to web content for {url}")
            logging.exception(e)
            return f"Failed to get content of the website {url}. Please try again later or try a different URL."


class YouTubeTranscriptInput(BaseModel):
    url: str = Field(description="The YouTube video URL to fetch the transcript for")


class YouTubeTranscriptTool(BaseTool):
    """
    Tool for fetching and analyzing YouTube video transcripts.
    Can be used to understand and discuss the content of YouTube videos without watching them.
    Supports various YouTube URL formats (standard, shortened, shorts).
    """

    name: str = "get_youtube_transcript"
    description: str = (
        "Tool for fetching and analyzing YouTube video transcripts. "
        "Can be used to understand and discuss the content of YouTube videos without watching them. "
        "Supports various YouTube URL formats (standard, shortened, shorts)."
    )
    args_schema: type[BaseModel] = YouTubeTranscriptInput
    context_max_tokens: int = 6000
    tavily_api_key: str = ""
    use_tavily: bool = False

    def _run(self, url: str) -> str:
        # Synchronous implementation if needed
        raise NotImplementedError("Use _arun instead")

    async def _arun(self, url: str) -> str:
        if self.use_tavily and self.tavily_api_key:
            content, reason = await _tavily_extract(url, self.tavily_api_key)
            if content is not None and _YOUTUBE_TIMESTAMP_RE.search(content):
                logging.info(f"Tavily Extract served YouTube transcript for {url}")
                return self._truncate(content)
            if content is not None:
                reason = "extracted content has no [mm:ss] transcript markers"
            logging.warning(f"Tavily Extract did not serve the transcript for {url} ({reason}); falling back")

        try:
            logging.debug(f"Serving YouTube transcript for {url} via YouTubeTranscriptApi (direct path)")
            video_id = extract_youtube_video_id(url)

            if not video_id:
                return (
                    f"Could not extract a valid YouTube video ID from the URL: {url}. "
                    "Please provide a valid YouTube URL."
                )

            # YouTubeTranscriptApi is synchronous, but we can wrap it in asyncio
            youtube_transcript = YouTubeTranscriptApi()

            # Run the synchronous API call in executor to avoid blocking
            loop = asyncio.get_event_loop()
            transcript_list = await loop.run_in_executor(None, lambda: list(youtube_transcript.list(video_id)))

            # Check if any transcripts are available
            if not transcript_list:
                logging.error(f"No transcript found for YouTube video: {url}")
                return (
                    "No transcript is available for this YouTube video. The video might not have captions or subtitles."
                )

            # Get the first available transcript
            transcript = transcript_list[0]
            transcript_segments = await loop.run_in_executor(None, transcript.fetch)

            # Combine all transcript segments into a single text
            full_transcript = " ".join(segment.text for segment in transcript_segments)
            return self._truncate(full_transcript.strip())
        except TranscriptsDisabled:
            logging.error(f"Transcripts are disabled for YouTube video: {url}")
            return f"Transcripts are disabled for this YouTube video: {url}"
        except Exception as e:
            logging.error(f"Error fetching YouTube transcript: {str(e)}")
            return f"Failed to get the transcript for the YouTube video. Error: {str(e)}"

    def _truncate(self, transcript: str) -> str:
        # Truncate if too long (to avoid token limits)
        # Rough estimate of chars per token
        max_chars = self.context_max_tokens * 4
        return _truncate_content(transcript, max_chars, "... [Transcript truncated due to length]")


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
    tavily_api_key = bot_config.tavily_api_key if bot_config else ""

    if (bot_config and bot_config.can_use_tavily_search) and tavily_api_key:
        logging.debug("Using Tavily search")
        search_tool = TavilySearch(tavily_api_key=tavily_api_key, max_results=5)
    else:
        logging.debug("Using ddgs search")
        search_tool = ddgs_search

    if bot_config:
        use_tavily_extract = bot_config.can_use_tavily_extract
        use_tavily_extract_for_youtube = bot_config.can_use_tavily_extract_for_youtube
        if (use_tavily_extract or use_tavily_extract_for_youtube) and not tavily_api_key:
            logging.warning(
                "Tavily Extract is enabled but no Tavily API key is configured "
                "(TAVILY_API_KEY); using the direct fetch paths."
            )

        web_tool: BaseTool = WebContentTool(
            context_max_tokens=bot_config.context_max_tokens,
            tavily_api_key=tavily_api_key,
            use_tavily=use_tavily_extract,
        )
        youtube_tool: BaseTool = YouTubeTranscriptTool(
            context_max_tokens=bot_config.context_max_tokens,
            tavily_api_key=tavily_api_key,
            use_tavily=use_tavily_extract_for_youtube,
        )
    else:
        web_tool = WebContentTool()
        youtube_tool = YouTubeTranscriptTool()

    tools: list = [
        get_current_time,
        web_tool,
        author,
        youtube_tool,
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
