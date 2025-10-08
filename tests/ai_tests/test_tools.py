import unittest
import unittest.mock
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import aiohttp

from ai.tools import (
    TimeResult,
    ddgs_search,
    extract_youtube_video_id,
    get_current_time,
    get_search_instructions,
    get_tool,
    get_website_content,
    get_youtube_transcript,
    multiply,
)


class TestSearchInstructionsTool(unittest.TestCase):
    def test_get_search_instructions_returns_expected_format(self):
        # Act
        result = get_search_instructions.invoke({})

        # Assert
        self.assertIsInstance(result, str)
        self.assertIn("search tools", result.lower())
        self.assertIn("web content", result.lower())

    def test_get_search_instructions_registered_correctly(self):
        # Act
        tool = get_tool("get_search_instructions")

        # Assert
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "get_search_instructions")
        self.assertIn("search instructions", tool.description.lower())


class TestLlmBot(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Mock the Config class for all tests
        self.config_patcher = patch("ai.tools.Config")
        self.mock_config_class = self.config_patcher.start()
        self.mock_config = self.mock_config_class.return_value
        # Set default timeout values
        self.mock_config.web_content_request_timeout = 10

    def tearDown(self):
        self.config_patcher.stop()

    def test_multiply__positive_integers(self):
        # Arrange
        first_int = 5
        second_int = 7
        expected_result = 35

        # Act
        result = multiply.invoke({"first_int": first_int, "second_int": second_int})

        # Assert
        self.assertEqual(result, expected_result)

    @patch("ai.tools.aiohttp.ClientSession")
    async def test_get_website_content__successful_content_retrieval(self, mock_session_class):
        # Arrange
        mock_session = unittest.mock.MagicMock()  # Not AsyncMock, as the session itself is not async
        mock_response = unittest.mock.AsyncMock()
        mock_response.text = unittest.mock.AsyncMock(return_value="<html><body>Test content</body></html>")

        # Mock the response context manager
        mock_response_context_manager = unittest.mock.MagicMock()  # Not AsyncMock for the CM object itself
        mock_response_context_manager.__aenter__ = unittest.mock.AsyncMock(return_value=mock_response)
        mock_response_context_manager.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_session.get = unittest.mock.MagicMock(
            return_value=mock_response_context_manager
        )  # get() returns CM, not a coroutine

        # Mock the session context manager
        mock_session_context_manager = unittest.mock.MagicMock()  # Not AsyncMock for the CM object itself
        mock_session_context_manager.__aenter__ = unittest.mock.AsyncMock(return_value=mock_session)
        mock_session_context_manager.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session_context_manager

        # Act
        result = await get_website_content.ainvoke({"url": "https://example.com"})

        # Assert
        mock_session.get.assert_called_once_with("https://example.com")
        self.assertIn("Test content", result)

    @patch("ai.tools.aiohttp.ClientSession")
    @patch("ai.tools.logging")
    async def test_get_website_content__connection_error_handling(self, mock_logging, mock_session_class):
        # Arrange
        mock_session = unittest.mock.MagicMock()
        # Create a context manager that raises aiohttp.ClientError on __aenter__
        mock_response_cm = unittest.mock.MagicMock()
        mock_response_cm.__aenter__ = unittest.mock.AsyncMock(side_effect=aiohttp.ClientError("Connection refused"))
        mock_response_cm.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_session.get = unittest.mock.MagicMock(return_value=mock_response_cm)

        # Mock the session context manager
        mock_session_context_manager = unittest.mock.MagicMock()
        mock_session_context_manager.__aenter__ = unittest.mock.AsyncMock(return_value=mock_session)
        mock_session_context_manager.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session_context_manager

        # Act
        result = await get_website_content.ainvoke({"url": "https://example.com"})

        # Assert
        mock_logging.error.assert_called_with("Connection error connecting to web content")
        mock_logging.exception.assert_called_once()
        self.assertEqual(
            result, "Failed to connect to the website https://example.com. Please check the URL or try again later."
        )

    @patch("ai.tools.aiohttp.ClientSession")
    @patch("ai.tools.logging")
    async def test_get_website_content__timeout_error_handling(self, mock_logging, mock_session_class):
        # Arrange

        mock_session = unittest.mock.MagicMock()
        # Create a context manager that raises TimeoutError on __aenter__
        mock_response_cm = unittest.mock.MagicMock()
        mock_response_cm.__aenter__ = unittest.mock.AsyncMock(side_effect=TimeoutError("Connection timed out"))
        mock_response_cm.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_session.get = unittest.mock.MagicMock(return_value=mock_response_cm)

        # Mock the session context manager
        mock_session_context_manager = unittest.mock.MagicMock()
        mock_session_context_manager.__aenter__ = unittest.mock.AsyncMock(return_value=mock_session)
        mock_session_context_manager.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session_context_manager

        # Act
        result = await get_website_content.ainvoke({"url": "https://example.com"})

        # Assert
        mock_logging.error.assert_called_with("Timeout error connecting to web content")
        mock_logging.exception.assert_called_once()
        self.assertEqual(
            result, "The website https://example.com took too long to respond. It might be unavailable or too large."
        )

    def test_get_tool__returns_correct_tool_for_valid_name(self):
        # Arrange
        expected_tool = multiply

        # Act
        result = get_tool("multiply")

        # Assert
        self.assertEqual(result, expected_tool)
        self.assertEqual(result.name, "multiply")

    def test_get_tool__returns_none_for_empty_string(self):
        # Arrange
        empty_name = ""

        # Act
        result = get_tool(empty_name)

        # Assert
        self.assertIsNone(result)

    def test_get_tool__handles_case_sensitivity(self):
        # Arrange
        uppercase_name = "MULTIPLY"
        mixed_case_name = "MuLtIpLy"

        # Act
        uppercase_result = get_tool(uppercase_name)
        mixed_case_result = get_tool(mixed_case_name)
        # Assert
        self.assertIsNone(uppercase_result)
        self.assertIsNone(mixed_case_result)


class TestGetCurrentTimeTool(unittest.IsolatedAsyncioTestCase):
    async def test_get_current_time_valid_timezone(self):
        # Test with a known timezone
        timezone = "America/New_York"
        result = await get_current_time.ainvoke({"timezone_name": timezone})

        # Verify the result is a TimeResult object
        self.assertIsInstance(result, TimeResult)

        # Verify the timezone matches the input
        self.assertEqual(result.timezone, timezone)

        # Verify the datetime string can be parsed and is in the correct timezone
        dt = datetime.fromisoformat(result.datetime)
        # Get the timezone offset for comparison
        expected_tz = ZoneInfo(timezone)
        expected_offset = datetime.now(expected_tz).utcoffset()
        actual_offset = dt.utcoffset()
        self.assertEqual(actual_offset, expected_offset, f"Expected offset {expected_offset} but got {actual_offset}")

        # Verify day_of_week is a valid day name
        self.assertIn(
            result.day_of_week, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )

        # Verify is_dst is a boolean
        self.assertIsInstance(result.is_dst, bool)

    async def test_get_current_time_invalid_timezone(self):
        # Test with an invalid timezone
        with self.assertRaises(Exception) as context:
            await get_current_time.ainvoke({"timezone_name": "Invalid/Timezone"})

        self.assertIn("Invalid timezone", str(context.exception))

    async def test_get_current_time_with_etc_greenwich(self):
        # Test with the default timezone mentioned in the docstring
        timezone = "Etc/Greenwich"
        result = await get_current_time.ainvoke({"timezone_name": timezone})

        self.assertEqual(result.timezone, timezone)
        dt = datetime.fromisoformat(result.datetime)
        # Get the timezone offset for comparison
        expected_tz = ZoneInfo(timezone)
        expected_offset = datetime.now(expected_tz).utcoffset()
        actual_offset = dt.utcoffset()
        self.assertEqual(actual_offset, expected_offset, f"Expected offset {expected_offset} but got {actual_offset}")


class TestDDGSSearchTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ddgs_patcher = patch("ai.tools.DDGS")
        self.mock_ddgs = self.ddgs_patcher.start()
        self.mock_ddgs_instance = MagicMock()

        # Mock the context manager behavior
        self.mock_ddgs.return_value.__enter__ = MagicMock(return_value=self.mock_ddgs_instance)
        self.mock_ddgs.return_value.__exit__ = MagicMock(return_value=None)

        # Set up default return value for text()
        self.mock_ddgs_instance.text.return_value = []

    def tearDown(self):
        self.ddgs_patcher.stop()

    async def test_ddgs_search_successful(self):
        # Arrange
        expected_results = [
            {"title": "Test Result 1", "link": "https://example.com/1", "snippet": "Test snippet 1"},
            {"title": "Test Result 2", "link": "https://example.com/2", "snippet": "Test snippet 2"},
        ]
        self.mock_ddgs_instance.text.return_value = expected_results
        query = "test query"

        # Act
        result = await ddgs_search.ainvoke({"query": query})

        # Assert
        self.mock_ddgs.assert_called_once()
        self.mock_ddgs_instance.text.assert_called_once_with(query, max_results=5)
        self.assertEqual(result, expected_results)

    async def test_ddgs_search_empty_query(self):
        # Act
        result = await ddgs_search.ainvoke({"query": ""})

        # Assert
        self.mock_ddgs.assert_called_once()
        self.mock_ddgs_instance.text.assert_called_once_with("", max_results=5)
        self.assertEqual(result, [])

    async def test_ddgs_search_handles_api_error(self):
        # Arrange
        self.mock_ddgs_instance.text.side_effect = Exception("API error")

        # Act
        with self.assertRaises(Exception) as context:
            await ddgs_search.ainvoke({"query": "test query"})

        # Assert
        self.assertEqual(str(context.exception), "API error")


class TestGetAllTools(unittest.IsolatedAsyncioTestCase):
    """Tests for get_all_tools function."""

    async def test_get_all_tools_with_conflict_resolution(self):
        """Test that MCP tools override conflicting custom tools."""
        from unittest.mock import AsyncMock, MagicMock

        from ai.tools import get_all_tools, get_tools

        # Get custom tools (should include multiply, etc.)
        custom_tools = get_tools()
        custom_tool_names = {t.name for t in custom_tools}

        # Create mock MCP manager with a conflicting tool
        mock_mcp_manager = MagicMock()
        mock_mcp_manager.is_connected = True

        # Create a mock MCP tool with same name as a custom tool
        conflicting_tool_name = list(custom_tool_names)[0]  # Use first custom tool name
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = conflicting_tool_name
        mock_mcp_tool.description = "MCP version"

        # Create another non-conflicting MCP tool
        mock_unique_tool = MagicMock()
        mock_unique_tool.name = "unique_mcp_tool"
        mock_unique_tool.description = "Unique MCP tool"

        mock_mcp_manager.get_tools = AsyncMock(return_value=[mock_mcp_tool, mock_unique_tool])

        # Act
        all_tools = await get_all_tools(mock_mcp_manager)

        # Assert
        tool_names = [t.name for t in all_tools]

        # Should have all custom tools except the conflicting one
        expected_count = len(custom_tools) - 1 + 2  # -1 conflict +2 MCP tools
        self.assertEqual(len(all_tools), expected_count)

        # The conflicting tool should appear only once (the MCP version)
        self.assertEqual(tool_names.count(conflicting_tool_name), 1)

        # The unique MCP tool should be present
        self.assertIn("unique_mcp_tool", tool_names)

        # Find the conflicting tool in the result
        conflicting_tool = next(t for t in all_tools if t.name == conflicting_tool_name)
        self.assertEqual(conflicting_tool.description, "MCP version")

    async def test_get_all_tools_without_mcp(self):
        """Test that get_all_tools works without MCP manager."""
        from ai.tools import get_all_tools, get_tools

        # Act
        all_tools = await get_all_tools(None)

        # Assert
        custom_tools = get_tools()
        self.assertEqual(len(all_tools), len(custom_tools))
        self.assertEqual([t.name for t in all_tools], [t.name for t in custom_tools])


class TestYouTubeTranscriptTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Mock the Config class for all tests
        self.config_patcher = patch("ai.tools.Config")
        self.mock_config_class = self.config_patcher.start()
        self.mock_config = self.mock_config_class.return_value
        # Set default token limit
        self.mock_config.context_max_tokens = 4000

    def tearDown(self):
        self.config_patcher.stop()

    def test_extract_youtube_video_id__standard_url(self):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        expected_id = "dQw4w9WgXcQ"

        # Act
        result = extract_youtube_video_id(url)

        # Assert
        self.assertEqual(result, expected_id)

    def test_extract_youtube_video_id__shortened_url(self):
        # Arrange
        url = "https://youtu.be/dQw4w9WgXcQ"
        expected_id = "dQw4w9WgXcQ"

        # Act
        result = extract_youtube_video_id(url)

        # Assert
        self.assertEqual(result, expected_id)

    def test_extract_youtube_video_id__shorts_url(self):
        # Arrange
        url = "https://youtube.com/shorts/dQw4w9WgXcQ"
        expected_id = "dQw4w9WgXcQ"

        # Act
        result = extract_youtube_video_id(url)

        # Assert
        self.assertEqual(result, expected_id)

    def test_extract_youtube_video_id__invalid_url(self):
        # Arrange
        url = "https://example.com/not-a-youtube-url"

        # Act
        result = extract_youtube_video_id(url)

        # Assert
        self.assertIsNone(result)

    @patch("ai.tools.YouTubeTranscriptApi")
    async def test_get_youtube_transcript__successful_transcript_retrieval(self, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_transcript = unittest.mock.Mock()
        mock_transcript.fetch.return_value = [
            type("obj", (object,), {"text": "This is the first part of the transcript."}),
            type("obj", (object,), {"text": "This is the second part of the transcript."}),
        ]
        mock_transcript_api.return_value.list.return_value = [mock_transcript]
        expected_result = "This is the first part of the transcript. This is the second part of the transcript. "

        # Act
        result = await get_youtube_transcript.ainvoke({"url": url})

        # Assert
        mock_transcript_api.return_value.list.assert_called_once_with("dQw4w9WgXcQ")
        self.assertEqual(result, expected_result.strip())

    @patch("ai.tools.YouTubeTranscriptApi")
    @patch("ai.tools.logging")
    async def test_get_youtube_transcript__no_transcript_found(self, mock_logging, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_transcript_api.return_value.list.return_value = []

        # Act
        result = await get_youtube_transcript.ainvoke({"url": url})

        # Assert
        mock_logging.error.assert_called_with(f"No transcript found for YouTube video: {url}")
        self.assertIn("No transcript is available for this YouTube video", result)

    @patch("ai.tools.YouTubeTranscriptApi")
    @patch("ai.tools.logging")
    async def test_get_youtube_transcript__transcripts_disabled(self, mock_logging, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        # The implementation raises IndexError when no transcript is found
        mock_transcript_api.return_value.list.return_value = []

        # Act
        result = await get_youtube_transcript.ainvoke({"url": url})

        # Assert
        mock_logging.error.assert_called_with(f"No transcript found for YouTube video: {url}")
        self.assertIn("No transcript is available for this YouTube video", result)

    @patch("ai.tools.extract_youtube_video_id")
    async def test_get_youtube_transcript__invalid_youtube_url(self, mock_extract_id):
        # Arrange
        url = "https://example.com/not-a-youtube-url"
        mock_extract_id.return_value = None

        # Act
        result = await get_youtube_transcript.ainvoke({"url": url})

        # Assert
        self.assertIn("Could not extract a valid YouTube video ID", result)

    @patch("ai.tools.YouTubeTranscriptApi")
    @patch("ai.tools.Config")
    async def test_get_youtube_transcript__truncates_long_transcripts(self, mock_config, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        # Create a very long transcript
        long_text = "This is a word. " * 5000  # Should exceed our token limit
        mock_transcript = unittest.mock.Mock()
        mock_transcript.fetch.return_value = [type("obj", (object,), {"text": long_text})]
        mock_transcript_api.return_value.list.return_value = [mock_transcript]

        # Set max tokens to a small number to force truncation
        mock_config.return_value.context_max_tokens = 100

        # Act
        result = await get_youtube_transcript.ainvoke({"url": url})

        # Assert
        self.assertLess(len(result), len(long_text))
        self.assertIn("[Transcript truncated due to length]", result)
