import unittest
from unittest.mock import patch, MagicMock

from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled

from ai.tools import get_youtube_transcript, extract_youtube_video_id


class TestYouTubeTranscriptTool(unittest.TestCase):
    def setUp(self):
        # Mock the Config class for all tests
        self.config_patcher = patch('ai.tools.Config')
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

    @patch('ai.tools.YouTubeTranscriptApi')
    def test_get_youtube_transcript__successful_transcript_retrieval(self, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_transcript = [
            {'text': 'This is the first part of the transcript.', 'start': 0.0, 'duration': 5.0},
            {'text': 'This is the second part of the transcript.', 'start': 5.0, 'duration': 5.0}
        ]
        mock_transcript_api.get_transcript.return_value = mock_transcript
        expected_result = "This is the first part of the transcript. This is the second part of the transcript."

        # Act
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        mock_transcript_api.get_transcript.assert_called_once_with("dQw4w9WgXcQ")
        self.assertEqual(result, expected_result)

    @patch('ai.tools.YouTubeTranscriptApi')
    @patch('ai.tools.logging')
    def test_get_youtube_transcript__no_transcript_found(self, mock_logging, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        # Create a proper NoTranscriptFound exception with required arguments
        no_transcript_exception = NoTranscriptFound(
            video_id="dQw4w9WgXcQ",
            requested_language_codes=["en"],
            transcript_data={}
        )
        mock_transcript_api.get_transcript.side_effect = no_transcript_exception

        # Act
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        mock_logging.error.assert_called_with(f"No transcript found for YouTube video: {url}")
        self.assertIn("No transcript is available for this YouTube video", result)

    @patch('ai.tools.YouTubeTranscriptApi')
    @patch('ai.tools.logging')
    def test_get_youtube_transcript__transcripts_disabled(self, mock_logging, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        # Create a TranscriptsDisabled exception
        transcripts_disabled_exception = TranscriptsDisabled("dQw4w9WgXcQ")
        mock_transcript_api.get_transcript.side_effect = transcripts_disabled_exception

        # Act
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        mock_logging.error.assert_called_with(f"Transcripts are disabled for YouTube video: {url}")
        self.assertIn("Transcripts are disabled for this YouTube video", result)

    @patch('ai.tools.extract_youtube_video_id')
    def test_get_youtube_transcript__invalid_youtube_url(self, mock_extract_id):
        # Arrange
        url = "https://example.com/not-a-youtube-url"
        mock_extract_id.return_value = None

        # Act
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        self.assertIn("Could not extract a valid YouTube video ID", result)

    @patch('ai.tools.YouTubeTranscriptApi')
    def test_get_youtube_transcript__truncates_long_transcripts(self, mock_transcript_api):
        # Arrange
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        # Create a very long transcript
        long_text = "This is a word. " * 5000  # Should exceed our token limit
        mock_transcript = [{'text': long_text, 'start': 0.0, 'duration': 5.0}]
        mock_transcript_api.get_transcript.return_value = mock_transcript
        
        # Set max tokens to a small number to force truncation
        self.mock_config.context_max_tokens = 100

        # Act
        result = get_youtube_transcript.invoke({"url": url})

        # Assert
        self.assertLess(len(result), len(long_text))
        self.assertIn("[Transcript truncated due to length]", result)

