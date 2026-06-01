import io
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from manolo_bot.ai.document_loaders import DocumentLoader, UnsupportedFileError, clean_text


class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DocumentLoader()

    def test_clean_text(self):
        # Arrange
        raw_text = "This  is   a test.\n\n\nNew  line."
        expected = "This is a test.\n\nNew line."

        # Act
        result = clean_text(raw_text)

        # Assert
        self.assertEqual(result, expected)

    def test_validate_filename_supported(self):
        # Should not raise
        DocumentLoader.validate_filename("test.pdf")
        DocumentLoader.validate_filename("test.docx")
        DocumentLoader.validate_filename("test.txt")

    def test_validate_filename_unsupported(self):
        with self.assertRaises(UnsupportedFileError):
            DocumentLoader.validate_filename("test.exe")

    def test_validate_filename_no_extension(self):
        with self.assertRaises(UnsupportedFileError):
            DocumentLoader.validate_filename("test")

    def test_extract_text_from_txt(self):
        # Arrange
        content = b"Hello, this is a text file."
        file_like = io.BytesIO(content)

        # Act
        result = self.loader.extract_text_from_txt(file_like)

        # Assert
        self.assertEqual(result, "Hello, this is a text file.")

    @patch("manolo_bot.ai.document_loaders.PyPDFParser.lazy_parse")
    def test_extract_text_from_pdf(self, mock_lazy_parse):
        # Arrange
        mock_lazy_parse.return_value = iter([Document(page_content="PDF Content")])
        file_like = io.BytesIO(b"dummy pdf content")

        # Act
        result = self.loader.extract_text_from_pdf(file_like)

        # Assert
        self.assertEqual(result, "PDF Content")
        mock_lazy_parse.assert_called_once()

    @patch("manolo_bot.ai.document_loaders.DocxParser.lazy_parse")
    def test_extract_text_from_docx(self, mock_lazy_parse):
        # Arrange
        mock_lazy_parse.return_value = iter([Document(page_content="DOCX Content")])
        file_like = io.BytesIO(b"dummy docx content")

        # Act
        result = self.loader.extract_text_from_docx(file_like)

        # Assert
        self.assertEqual(result, "DOCX Content")
        mock_lazy_parse.assert_called_once()

    def test_extract_text_dispatcher_txt(self):
        # Arrange
        content = b"Dispatcher test"

        # Act
        result = self.loader.extract_text(content, "test.txt")

        # Assert
        self.assertEqual(result, "Dispatcher test")

    @patch("manolo_bot.ai.document_loaders.PyPDFParser.lazy_parse")
    def test_extract_text_dispatcher_pdf(self, mock_lazy_parse):
        # Arrange
        mock_lazy_parse.return_value = iter([Document(page_content="PDF Dispatcher Content")])

        # Act
        result = self.loader.extract_text(b"pdf content", "test.pdf")

        # Assert
        self.assertEqual(result, "PDF Dispatcher Content")

    @patch("manolo_bot.ai.document_loaders.DocxParser.lazy_parse")
    def test_extract_text_dispatcher_docx(self, mock_lazy_parse):
        # Arrange
        mock_lazy_parse.return_value = iter([Document(page_content="DOCX Dispatcher Content")])

        # Act
        result = self.loader.extract_text(b"docx content", "test.docx")

        # Assert
        self.assertEqual(result, "DOCX Dispatcher Content")

    def test_extract_text_unsupported_extension(self):
        with self.assertRaises(UnsupportedFileError):
            self.loader.extract_text(b"content", "test.exe")


if __name__ == "__main__":
    unittest.main()
