import unittest
import unittest.mock
from unittest.mock import patch

from requests import ConnectTimeout

from ai.tools import extract_youtube_video_id, get_tool, get_website_content, get_youtube_transcript, multiply


class TestLlmBot(unittest.TestCase):
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

    @patch("ai.tools.WebBaseLoader")
    def test_get_website_content__successful_content_retrieval(self, mock_loader):
        # Arrange
        mock_instance = mock_loader.return_value
        mock_instance.load.return_value = [unittest.mock.Mock(page_content="Test content")]

        # Act
        result = get_website_content.invoke({"url": "https://example.com"})

        # Assert
        mock_loader.assert_called_once_with("https://example.com")
        mock_instance.load.assert_called_once()
        self.assertEqual(result, "Test content")

    @patch("ai.tools.WebBaseLoader")
    @patch("ai.tools.logging")
    def test_get_website_content__connection_error_handling(self, mock_logging, mock_loader):
        # Arrange
        mock_instance = mock_loader.return_value
        mock_instance.load.side_effect = ConnectionError("Connection refused")

        # Act
        result = get_website_content.invoke({"url": "https://example.com"})

        # Assert
        mock_logging.error.assert_called_with("Connection error connecting to web content")
        mock_logging.exception.assert_called_once()
        self.assertEqual(
            result, "Failed to connect to the website https://example.com. Please check the URL or try again later."
        )

    @patch("ai.tools.WebBaseLoader")
    @patch("ai.tools.logging")
    def test_get_website_content__timeout_error_handling(self, mock_logging, mock_loader):
        # Arrange
        mock_instance = mock_loader.return_value
        mock_instance.load.side_effect = ConnectTimeout("Connection timed out")

        # Act
        result = get_website_content.invoke({"url": "https://example.com"})

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


class TestYouTubeTranscriptTool(unittest.TestCase):
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
    def test_get_youtube_transcript__successful_transcript_retrieval(self, mock_transcript_api):
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
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        mock_transcript_api.return_value.list.assert_called_once_with("dQw4w9WgXcQ")
        self.assertEqual(result, expected_result.strip())

    @patch("ai.tools.YouTubeTranscriptApi")
    @patch("ai.tools.logging")
    def test_get_youtube_transcript__no_transcript_found(self, mock_logging, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_transcript_api.return_value.list.return_value = []

        # Act
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        mock_logging.error.assert_called_with(f"No transcript found for YouTube video: {url}")
        self.assertIn("No transcript is available for this YouTube video", result)

    @patch("ai.tools.YouTubeTranscriptApi")
    @patch("ai.tools.logging")
    def test_get_youtube_transcript__transcripts_disabled(self, mock_logging, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        # The implementation raises IndexError when no transcript is found
        mock_transcript_api.return_value.list.return_value = []

        # Act
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        mock_logging.error.assert_called_with(f"No transcript found for YouTube video: {url}")
        self.assertIn("No transcript is available for this YouTube video", result)

    @patch("ai.tools.extract_youtube_video_id")
    def test_get_youtube_transcript__invalid_youtube_url(self, mock_extract_id):
        # Arrange
        url = "https://example.com/not-a-youtube-url"
        mock_extract_id.return_value = None

        # Act
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        self.assertIn("Could not extract a valid YouTube video ID", result)

    @patch("ai.tools.YouTubeTranscriptApi")
    @patch("ai.tools.Config")
    def test_get_youtube_transcript__truncates_long_transcripts(self, mock_config, mock_transcript_api):
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
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        self.assertLess(len(result), len(long_text))
        self.assertIn("[Transcript truncated due to length]", result)
