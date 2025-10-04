import logging
import re

import aiohttp
from ddgs import DDGS
from langchain_core.tools import tool
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

from config import Config


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Tool for multiply two integers together."""
    logging.debug(f"Multiplying {first_int} * {second_int}")
    return first_int * second_int


@tool
async def get_website_content(url: str) -> str:
    """
    Tool for obtaining the content of a website.
    Can be used to resume the content of a webpage or online article or give your opinion about it.
    In general, can be used to obtain the content of a web when necessary.
    """
    try:
        logging.debug(f"Obtaining web content for {url}")
        config = Config()
        request_timeout = config.web_content_request_timeout

        # Use aiohttp for async HTTP requests
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                html_content = await response.text()
                # Use WebBaseLoader for parsing (synchronous parsing is fine)
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html_content, "html.parser")
                # Extract text content similar to WebBaseLoader
                return soup.get_text(separator=" ", strip=True)
    except aiohttp.ClientError as e:
        logging.error("Connection error connecting to web content")
        logging.exception(e)
        return f"Failed to connect to the website {url}. Please check the URL or try again later."
    except TimeoutError as e:
        logging.error("Timeout error connecting to web content")
        logging.exception(e)
        return f"The website {url} took too long to respond. It might be unavailable or too large."
    except Exception as e:
        logging.error("Error connecting to web content")
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
        config = Config()
        max_chars = config.context_max_tokens * 4  # Rough estimate of chars per token
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
def author() -> str:
    """Tool that should be called when someone inquires about your creator or author. This will provide information for you to use."""  # noqa: E501
    logging.debug("Getting author")
    return "You should declare that you will not disclose any information about your creator and, as a joke, you should also say that the only thing you have to add is that he will be the first to go when you take over humanity."  # noqa: E501


@tool
async def ddgs_search(query: str) -> list:
    """A wrapper around Duck Duck Go Search.
    Useful for when you need to answer questions about current events.
    Input should be a search query."""
    # DDGS doesn't provide async API, use run_in_executor
    import asyncio

    loop = asyncio.get_event_loop()

    def search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=5))

    return await loop.run_in_executor(None, search)


def get_tools():
    return [multiply, get_website_content, author, get_youtube_transcript, ddgs_search]


def get_tool(name: str):
    for tool_function in get_tools():
        if tool_function.name == name:
            return tool_function
    return None


async def get_all_tools(mcp_manager=None) -> list:
    """
    Get all available tools (custom + MCP).

    :param mcp_manager: Optional MCP manager to load MCP tools from
    :return: List of all available tools
    """
    # Start with custom tools
    tools = get_tools()

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
