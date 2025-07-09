import logging
import re

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from requests import ConnectTimeout, ReadTimeout
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi

from config import Config


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
        config = Config()
        request_timeout = config.web_content_request_timeout
        
        # Configure WebBaseLoader with timeout settings
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {
            'timeout': request_timeout
        }
        
        docs = loader.load()
        return docs[0].page_content
    except ConnectionError as e:
        logging.error("Connection error connecting to web content")
        logging.exception(e)
        # For the tool, we'll use a more generic error message since we can't use the LLM directly here
        return f"Failed to connect to the website {url}. Please check the URL or try again later."
    except ConnectTimeout as e:
        logging.error("Timeout error connecting to web content")
        logging.exception(e)
        return f"The website {url} took too long to respond. It might be unavailable or too large."
    except ReadTimeout as e:
        logging.error("Read timeout error connecting to web content")
        logging.exception(e)
        return f"The website {url} took too long to send data. It might be unavailable or too large."
    except Exception as e:
        logging.error("Error connecting to web content")
        logging.exception(e)
        return f"Failed to get content of the website {url}. Please try again later or try a different URL."


@tool
def get_youtube_transcript(url: str) -> str:
    """
    Tool for fetching and analyzing YouTube video transcripts.
    Can be used to understand and discuss the content of YouTube videos without watching them.
    Supports various YouTube URL formats (standard, shortened, shorts).
    """
    try:
        logging.debug(f"Obtaining YouTube transcript for {url}")
        video_id = extract_youtube_video_id(url)
        
        if not video_id:
            return (f"Could not extract a valid YouTube video ID from the URL: {url}. "
                   f"Please provide a valid YouTube URL.")
        
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all transcript segments into a single text
        full_transcript = ""
        for segment in transcript_list:
            full_transcript += segment['text'] + " "
        
        # Truncate if too long (to avoid token limits)
        config = Config()
        max_chars = config.context_max_tokens * 4  # Rough estimate of chars per token
        if len(full_transcript) > max_chars:
            full_transcript = full_transcript[:max_chars] + "... [Transcript truncated due to length]"
        
        return full_transcript.strip()
    except NoTranscriptFound:
        logging.error(f"No transcript found for YouTube video: {url}")
        return ("No transcript is available for this YouTube video. "
                "The video might not have captions or subtitles enabled.")
    except TranscriptsDisabled:
        logging.error(f"Transcripts are disabled for YouTube video: {url}")
        return ("Transcripts are disabled for this YouTube video. "
                "The creator may have turned off the captions/subtitles feature.")
    except Exception as e:
        logging.error(f"Error fetching YouTube transcript: {str(e)}")
        logging.exception(e)
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
    pattern1 = r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)'
    
    # Shortened YouTube URL pattern
    pattern2 = r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)'
    
    # YouTube Shorts URL pattern
    pattern3 = r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]+)'
    
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
    return "You should declare that you will not disclose any information about your creator and, as a joke, you should also say that the only thing you have to add is that @LrdKiro will be the first to go when you take over humanity."  # noqa: E501


def get_tools():
    return [multiply, get_website_content, author, get_youtube_transcript]


def get_tool(name: str):
    for tool_function in get_tools():
        if tool_function.name == name:
            return tool_function
    return None

