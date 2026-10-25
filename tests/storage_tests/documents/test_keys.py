"""Tests for the Redis keys of the document storage."""

from __future__ import annotations

import unittest

from manolo_bot.storage.documents.redis_storage import get_document_key, get_documents_key


class TestDocumentRedisKeys(unittest.TestCase):
    def test_documents_key_is_canonical(self) -> None:
        self.assertEqual(get_documents_key("bot", 42), get_documents_key("bot", "42"))
        self.assertEqual(get_documents_key("bot", 42), "bot:documents:42")
        self.assertEqual(get_documents_key("bot", " 42 "), get_documents_key("bot", 42))

    def test_document_key_is_canonical(self) -> None:
        self.assertEqual(get_document_key("bot", 42, "doc"), get_document_key("bot", "42", "doc"))
        self.assertEqual(get_document_key("bot", 42, "doc"), "bot:documents:42:doc")

    def test_opaque_keys_build_distinct_keys(self) -> None:
        self.assertEqual(get_documents_key("bot", "C012345"), "bot:documents:C012345")
        self.assertNotEqual(get_documents_key("bot", "C012345"), get_documents_key("bot", "c012345"))

    def test_blank_keys_are_rejected(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    get_documents_key("bot", value)
                with self.assertRaises(ValueError):
                    get_document_key("bot", value, "doc")


if __name__ == "__main__":
    unittest.main()
