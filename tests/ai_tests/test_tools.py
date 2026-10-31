import unittest
import unittest.mock
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

from manolo_bot.ai.tools import (
    ReadDocumentTool,
    TimeResult,
    WebContentTool,
    YouTubeTranscriptTool,
    _tavily_extract,
    ddgs_search,
    extract_youtube_video_id,
    get_current_time,
    get_search_instructions,
    get_tool,
    get_tools,
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
        self.config_patcher = patch("manolo_bot.ai.tools.BotConfig")
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

    @patch("manolo_bot.ai.tools.WebBaseLoader")
    async def test_get_website_content__successful_content_retrieval(self, mock_loader_class):
        # Arrange
        mock_loader = unittest.mock.MagicMock()
        mock_doc = unittest.mock.MagicMock()
        mock_doc.page_content = "Test content"

        # Mock the async iterator
        mock_docs = [mock_doc]
        mock_loader.alazy_load.return_value = self._async_iterator(mock_docs)
        mock_loader_class.return_value = mock_loader

        # Act
        result = await WebContentTool().ainvoke({"url": "https://example.com"})

        # Assert
        mock_loader_class.assert_called_once_with(web_path="https://example.com")
        self.assertIn("Test content", result)

    def _async_iterator(self, items):
        """Helper method to create an async iterator from a list."""

        async def iterator():
            for item in items:
                yield item

        return iterator()

    @patch("manolo_bot.ai.tools.WebBaseLoader")
    @patch("manolo_bot.ai.tools.logging")
    async def test_get_website_content__connection_error_handling(self, mock_logging, mock_loader_class):
        # Arrange
        mock_loader = unittest.mock.MagicMock()
        mock_loader.alazy_load.side_effect = Exception("Connection failed")
        mock_loader_class.return_value = mock_loader

        # Act
        result = await WebContentTool().ainvoke({"url": "https://example.com"})

        # Assert
        mock_logging.error.assert_called_with("Error connecting to web content for https://example.com")
        mock_logging.exception.assert_called_once()
        self.assertEqual(
            result,
            "Failed to get content of the website https://example.com. Please try again later or try a different URL.",
        )

    @patch("manolo_bot.ai.tools.WebBaseLoader")
    @patch("manolo_bot.ai.tools.logging")
    async def test_get_website_content__timeout_error_handling(self, mock_logging, mock_loader_class):
        # Arrange
        mock_loader = unittest.mock.MagicMock()
        mock_loader.alazy_load.side_effect = TimeoutError("Connection timed out")
        mock_loader_class.return_value = mock_loader

        # Act
        result = await WebContentTool().ainvoke({"url": "https://example.com"})

        # Assert
        mock_logging.error.assert_called_with("Error connecting to web content for https://example.com")
        mock_logging.exception.assert_called_once()
        self.assertEqual(
            result,
            "Failed to get content of the website https://example.com. Please try again later or try a different URL.",
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
        self.ddgs_patcher = patch("manolo_bot.ai.tools.DDGS")
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

        from manolo_bot.ai.tools import get_all_tools, get_tools

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
        from manolo_bot.ai.tools import get_all_tools, get_tools

        # Act
        all_tools = await get_all_tools(None)

        # Assert
        custom_tools = get_tools()
        self.assertEqual(len(all_tools), len(custom_tools))
        self.assertEqual([t.name for t in all_tools], [t.name for t in custom_tools])

    async def test_get_all_tools_custom_tools_replace_defaults(self):
        """Passing custom_tools REPLACES the default set (documented semantics)."""
        from langchain_core.tools import tool as tool_decorator

        from manolo_bot.ai.tools import get_all_tools, get_tools

        @tool_decorator
        def my_custom_tool(query: str) -> str:
            """Custom tool."""
            return "custom"

        # Act
        all_tools = await get_all_tools(None, custom_tools=[my_custom_tool])

        # Assert: only the custom tool is present, defaults are gone.
        self.assertEqual(len(all_tools), 1)
        self.assertIs(all_tools[0], my_custom_tool)
        default_names = {t.name for t in get_tools()}
        self.assertNotIn("my_custom_tool", default_names)
        self.assertGreater(len(default_names), 1)


class TestReadDocumentTool(unittest.IsolatedAsyncioTestCase):
    async def test_read_document_tool_initialization(self):
        # Arrange
        from manolo_bot.storage.documents.base import BaseDocumentStorage

        mock_storage = MagicMock(spec=BaseDocumentStorage)
        tool = ReadDocumentTool(document_storage=mock_storage, context_max_tokens=1000)

        # Assert
        self.assertEqual(tool.name, "read_document")
        self.assertEqual(tool.document_storage, mock_storage)

    async def test_read_document_tool_run(self):
        # Arrange
        from manolo_bot.storage.documents.base import BaseDocumentStorage

        mock_storage = MagicMock(spec=BaseDocumentStorage)
        mock_storage.retrieve = AsyncMock(return_value="Document content")
        tool = ReadDocumentTool(document_storage=mock_storage)

        mock_config = MagicMock()
        mock_config.get.return_value.get.return_value = 12345  # chat_id

        # Act
        result = await tool._arun("test.pdf", config=mock_config)

        # Assert
        self.assertEqual(result, "Document content")
        mock_storage.retrieve.assert_called_once_with(12345, "test.pdf")

    async def test_read_document_tool_not_found(self):
        # Arrange
        from manolo_bot.storage.documents.base import BaseDocumentStorage

        mock_storage = MagicMock(spec=BaseDocumentStorage)
        mock_storage.retrieve = AsyncMock(return_value=None)
        mock_storage.list_documents = AsyncMock(return_value=["other.pdf"])
        tool = ReadDocumentTool(document_storage=mock_storage)

        mock_config = MagicMock()
        mock_config.get.return_value.get.return_value = 12345

        # Act
        result = await tool._arun("missing.pdf", config=mock_config)

        # Assert
        self.assertIn("missing.pdf' not found", result)
        self.assertIn("other.pdf", result)


class TestYouTubeTranscriptTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Mock the Config class for all tests
        self.config_patcher = patch("manolo_bot.ai.tools.BotConfig")
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

    @patch("manolo_bot.ai.tools.YouTubeTranscriptApi")
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
        result = await YouTubeTranscriptTool().ainvoke({"url": url})

        # Assert
        mock_transcript_api.return_value.list.assert_called_once_with("dQw4w9WgXcQ")
        self.assertEqual(result, expected_result.strip())

    @patch("manolo_bot.ai.tools.YouTubeTranscriptApi")
    @patch("manolo_bot.ai.tools.logging")
    async def test_get_youtube_transcript__no_transcript_found(self, mock_logging, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_transcript_api.return_value.list.return_value = []

        # Act
        result = await YouTubeTranscriptTool().ainvoke({"url": url})

        # Assert
        mock_logging.error.assert_called_with(f"No transcript found for YouTube video: {url}")
        self.assertIn("No transcript is available for this YouTube video", result)

    @patch("manolo_bot.ai.tools.YouTubeTranscriptApi")
    @patch("manolo_bot.ai.tools.logging")
    async def test_get_youtube_transcript__transcripts_disabled(self, mock_logging, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        # The implementation raises IndexError when no transcript is found
        mock_transcript_api.return_value.list.return_value = []

        # Act
        result = await YouTubeTranscriptTool().ainvoke({"url": url})

        # Assert
        mock_logging.error.assert_called_with(f"No transcript found for YouTube video: {url}")
        self.assertIn("No transcript is available for this YouTube video", result)

    @patch("manolo_bot.ai.tools.extract_youtube_video_id")
    async def test_get_youtube_transcript__invalid_youtube_url(self, mock_extract_id):
        # Arrange
        url = "https://example.com/not-a-youtube-url"
        mock_extract_id.return_value = None

        # Act
        result = await YouTubeTranscriptTool().ainvoke({"url": url})

        # Assert
        self.assertIn("Could not extract a valid YouTube video ID", result)

    @patch("manolo_bot.ai.tools.YouTubeTranscriptApi")
    async def test_get_youtube_transcript__truncates_long_transcripts(self, mock_transcript_api):
        from manolo_bot.ai.tools import YouTubeTranscriptTool

        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        # Create a very long transcript
        long_text = "This is a word. " * 5000  # Should exceed our token limit
        mock_transcript = unittest.mock.Mock()
        mock_transcript.fetch.return_value = [type("obj", (object,), {"text": long_text})]
        mock_transcript_api.return_value.list.return_value = [mock_transcript]

        # Use a small limit to force truncation via the configured tool instance
        configured_tool = YouTubeTranscriptTool(context_max_tokens=100)

        # Act
        result = await configured_tool.ainvoke({"url": url})

        # Assert
        self.assertLess(len(result), len(long_text))
        self.assertIn("[Transcript truncated due to length]", result)


class TestTavilyExtractHelper(unittest.IsolatedAsyncioTestCase):
    """Tests for the shared Tavily Extract helper."""

    @patch("manolo_bot.ai.tools.TavilyExtract")
    async def test_returns_content_from_first_result(self, mock_extract_cls):
        mock_extract_cls.return_value.ainvoke = AsyncMock(
            return_value={
                "results": [{"url": "https://example.com/doc.pdf", "raw_content": "  extracted text  "}],
                "failed_results": [],
            }
        )

        content, reason = await _tavily_extract("https://example.com/doc.pdf", "test-key")

        self.assertEqual(content, "extracted text")
        self.assertEqual(reason, "")
        mock_extract_cls.assert_called_once_with(tavily_api_key="test-key", format="text", extract_depth="basic")
        mock_extract_cls.return_value.ainvoke.assert_awaited_once_with({"urls": ["https://example.com/doc.pdf"]})

    @patch("manolo_bot.ai.tools.TavilyExtract")
    async def test_returns_none_when_url_fails(self, mock_extract_cls):
        mock_extract_cls.return_value.ainvoke = AsyncMock(
            return_value={
                "results": [],
                "failed_results": [{"url": "https://example.com/gone", "error": "Failed to fetch url"}],
            }
        )

        content, reason = await _tavily_extract("https://example.com/gone", "test-key")

        self.assertIsNone(content)
        self.assertIn("no results", reason)
        self.assertNotIn("test-key", reason)

    @patch("manolo_bot.ai.tools.TavilyExtract")
    async def test_returns_none_on_error_payload(self, mock_extract_cls):
        mock_extract_cls.return_value.ainvoke = AsyncMock(return_value={"error": TimeoutError("slow")})

        content, reason = await _tavily_extract("https://example.com", "test-key")

        self.assertIsNone(content)
        self.assertIn("returned an error", reason)

    @patch("manolo_bot.ai.tools.TavilyExtract")
    async def test_returns_none_when_request_raises(self, mock_extract_cls):
        mock_extract_cls.return_value.ainvoke = AsyncMock(side_effect=TimeoutError("timed out"))

        content, reason = await _tavily_extract("https://example.com", "test-key")

        self.assertIsNone(content)
        self.assertIn("request failed", reason)
        self.assertIn("TimeoutError", reason)


class TestWebContentToolTavily(unittest.IsolatedAsyncioTestCase):
    """Tavily-first behaviour of get_website_content."""

    @staticmethod
    def _async_iterator(items):
        async def iterator():
            for item in items:
                yield item

        return iterator()

    def _direct_loader(self, content="Direct content"):
        loader = MagicMock()
        doc = MagicMock()
        doc.page_content = content
        loader.alazy_load.return_value = self._async_iterator([doc])
        return loader

    @patch("manolo_bot.ai.tools.WebBaseLoader")
    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_flag_off_never_calls_tavily(self, mock_extract, mock_loader_class):
        mock_loader_class.return_value = self._direct_loader()

        tool = WebContentTool(use_tavily=False, tavily_api_key="test-key")
        result = await tool._arun("https://example.com")

        self.assertIn("Direct content", result)
        mock_extract.assert_not_awaited()

    @patch("manolo_bot.ai.tools.WebBaseLoader")
    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_flag_on_uses_tavily_and_skips_loader(self, mock_extract, mock_loader_class):
        mock_extract.return_value = ("Tavily content", "")

        tool = WebContentTool(use_tavily=True, tavily_api_key="test-key")
        result = await tool._arun("https://example.com/document.pdf")

        self.assertEqual(result, "Tavily content")
        mock_extract.assert_awaited_once_with("https://example.com/document.pdf", "test-key")
        mock_loader_class.assert_not_called()

    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_flag_on_truncates_returned_content(self, mock_extract):
        body = "x" * 500
        mock_extract.return_value = (body, "")

        tool = WebContentTool(use_tavily=True, tavily_api_key="test-key", context_max_tokens=10)
        result = await tool._arun("https://example.com")

        notice = "... [Content truncated due to length]"
        self.assertEqual(len(result), 40 + len(notice))
        self.assertTrue(result.startswith("x" * 40))
        self.assertIn(notice, result)

    @patch("manolo_bot.ai.tools.WebBaseLoader")
    async def test_direct_path_is_truncated_too(self, mock_loader_class):
        mock_loader_class.return_value = self._direct_loader("y" * 500)

        tool = WebContentTool(use_tavily=False, context_max_tokens=10)
        result = await tool._arun("https://example.com")

        notice = "... [Content truncated due to length]"
        self.assertEqual(len(result), 40 + len(notice))
        self.assertIn(notice, result)

    def test_description_mentions_documents_only_when_tavily_can_serve_them(self):
        tavily_tool = WebContentTool(use_tavily=True, tavily_api_key="test-key")
        no_key_tool = WebContentTool(use_tavily=True, tavily_api_key="")
        flag_off_tool = WebContentTool(use_tavily=False, tavily_api_key="test-key")

        self.assertIn("PDFs", tavily_tool.description)
        self.assertNotIn("PDFs", no_key_tool.description)
        self.assertNotIn("PDFs", flag_off_tool.description)

    @patch("manolo_bot.ai.tools.WebBaseLoader")
    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_flag_on_falls_back_to_direct_path_on_failure(self, mock_extract, mock_loader_class):
        mock_extract.return_value = (None, "tavily extract returned no results")
        mock_loader_class.return_value = self._direct_loader("Fallback content")

        tool = WebContentTool(use_tavily=True, tavily_api_key="test-key")
        result = await tool.ainvoke({"url": "https://example.com"})

        self.assertIn("Fallback content", result)
        mock_extract.assert_awaited_once()
        mock_loader_class.assert_called_once_with(web_path="https://example.com")

    @patch("manolo_bot.ai.tools.WebBaseLoader")
    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_flag_on_without_key_uses_direct_path(self, mock_extract, mock_loader_class):
        mock_loader_class.return_value = self._direct_loader()

        tool = WebContentTool(use_tavily=True, tavily_api_key="")
        result = await tool._arun("https://example.com")

        self.assertIn("Direct content", result)
        mock_extract.assert_not_awaited()


class TestYouTubeTranscriptToolTavily(unittest.IsolatedAsyncioTestCase):
    """Tavily-first behaviour of get_youtube_transcript."""

    URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    @patch("manolo_bot.ai.tools.YouTubeTranscriptApi")
    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_flag_off_never_calls_tavily(self, mock_extract, mock_transcript_api):
        transcript = unittest.mock.Mock()
        transcript.fetch.return_value = [type("obj", (object,), {"text": "direct transcript"})]
        mock_transcript_api.return_value.list.return_value = [transcript]

        tool = YouTubeTranscriptTool(use_tavily=False, tavily_api_key="test-key")
        result = await tool._arun(self.URL)

        self.assertIn("direct transcript", result)
        mock_extract.assert_not_awaited()

    @patch("manolo_bot.ai.tools.YouTubeTranscriptApi")
    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_flag_on_uses_timestamped_extract(self, mock_extract, mock_transcript_api):
        mock_extract.return_value = ("Title\n[00:00] intro\n[01:05] body", "")

        tool = YouTubeTranscriptTool(use_tavily=True, tavily_api_key="test-key")
        result = await tool._arun(self.URL)

        self.assertIn("[01:05] body", result)
        mock_extract.assert_awaited_once_with(self.URL, "test-key")
        mock_transcript_api.assert_not_called()

    @patch("manolo_bot.ai.tools.YouTubeTranscriptApi")
    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_extract_without_timestamps_is_not_a_transcript(self, mock_extract, mock_transcript_api):
        mock_extract.return_value = ("Title\nDescription only, the video has no captions", "")
        mock_transcript_api.return_value.list.return_value = []

        tool = YouTubeTranscriptTool(use_tavily=True, tavily_api_key="test-key")
        result = await tool._arun(self.URL)

        self.assertNotIn("Description only", result)
        self.assertIn("No transcript is available", result)
        mock_transcript_api.return_value.list.assert_called_once_with("dQw4w9WgXcQ")

    @patch("manolo_bot.ai.tools.YouTubeTranscriptApi")
    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_flag_on_falls_back_when_tavily_fails(self, mock_extract, mock_transcript_api):
        mock_extract.return_value = (None, "tavily extract request failed")
        transcript = unittest.mock.Mock()
        transcript.fetch.return_value = [type("obj", (object,), {"text": "direct transcript"})]
        mock_transcript_api.return_value.list.return_value = [transcript]

        tool = YouTubeTranscriptTool(use_tavily=True, tavily_api_key="test-key")
        result = await tool._arun(self.URL)

        self.assertIn("direct transcript", result)
        mock_transcript_api.return_value.list.assert_called_once_with("dQw4w9WgXcQ")

    @patch("manolo_bot.ai.tools._tavily_extract", new_callable=AsyncMock)
    async def test_flag_on_truncates_extract(self, mock_extract):
        mock_extract.return_value = ("[00:00] " + "x" * 500, "")

        tool = YouTubeTranscriptTool(use_tavily=True, tavily_api_key="test-key", context_max_tokens=10)
        result = await tool._arun(self.URL)

        self.assertIn("[Transcript truncated due to length]", result)
        self.assertEqual(len(result), 40 + len("... [Transcript truncated due to length]"))


class TestGetToolsTavilyWiring(unittest.TestCase):
    """get_tools wires the resolved key and both opt-in flags onto the tools."""

    @staticmethod
    def _config(**overrides):
        values = {
            "can_use_tavily_search": False,
            "can_use_tavily_extract": False,
            "can_use_tavily_extract_for_youtube": False,
            "tavily_api_key": "",
            "context_max_tokens": 100,
            "is_document_multimodal": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_wires_flags_and_shared_key(self):
        config = self._config(
            can_use_tavily_extract=True,
            can_use_tavily_extract_for_youtube=True,
            tavily_api_key="test-key",
        )

        tools = get_tools(config)

        web_tool = next(t for t in tools if t.name == "get_website_content")
        youtube_tool = next(t for t in tools if t.name == "get_youtube_transcript")
        self.assertIsInstance(web_tool, WebContentTool)
        self.assertIsInstance(youtube_tool, YouTubeTranscriptTool)
        self.assertTrue(web_tool.use_tavily)
        self.assertTrue(youtube_tool.use_tavily)
        self.assertEqual(web_tool.tavily_api_key, "test-key")
        self.assertEqual(youtube_tool.tavily_api_key, "test-key")

    def test_warns_and_keeps_direct_tools_when_key_missing(self):
        config = self._config(can_use_tavily_extract=True, can_use_tavily_extract_for_youtube=True)

        with self.assertLogs(level="WARNING") as captured:
            tools = get_tools(config)

        self.assertTrue(any("Tavily Extract is enabled but no Tavily API key" in line for line in captured.output))
        web_tool = next(t for t in tools if t.name == "get_website_content")
        self.assertEqual(web_tool.tavily_api_key, "")

    def test_unconfigured_get_tools_returns_direct_tools(self):
        tools = get_tools()

        web_tool = next(t for t in tools if t.name == "get_website_content")
        youtube_tool = next(t for t in tools if t.name == "get_youtube_transcript")
        self.assertIsInstance(web_tool, WebContentTool)
        self.assertFalse(web_tool.use_tavily)
        self.assertIsInstance(youtube_tool, YouTubeTranscriptTool)
        self.assertFalse(youtube_tool.use_tavily)

    @patch("manolo_bot.ai.tools.TavilySearch")
    def test_tavily_search_uses_resolved_bot_config_key(self, mock_tavily_search):
        config = self._config(can_use_tavily_search=True, tavily_api_key="test-key")

        get_tools(config)

        mock_tavily_search.assert_called_once_with(tavily_api_key="test-key", max_results=5)
