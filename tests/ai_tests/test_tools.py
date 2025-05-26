import unittest
import unittest.mock
from unittest.mock import patch

from requests import ConnectTimeout

from ai.tools import get_tool, get_website_content, multiply


class TestLlmBot(unittest.TestCase):
    def setUp(self):
        # Mock the Config class for all tests
        self.config_patcher = patch('ai.tools.Config')
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
        self.assertEqual(result, "Failed to connect to the website https://example.com. Please check the URL or try again later.")

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
        self.assertEqual(result, "The website https://example.com took too long to respond. It might be unavailable or too large.")

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
