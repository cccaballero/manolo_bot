"""Tests for the Redis keys of the document storage."""

from __future__ import annotations

import unittest

from manolo_bot.storage.documents.redis_storage import get_document_key, get_documents_key
from manolo_bot.storage.messages.base import get_messages_key


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

    def test_colon_in_key_cannot_impersonate_a_document_set(self) -> None:
        # Set and item keys share a namespace; a colon in the key must not let
        # one scope read or delete another scope's document.
        self.assertNotEqual(
            get_documents_key("bot", "dashboard:a:b"),
            get_document_key("bot", "dashboard:a", "b"),
        )

    def test_colon_in_filename_cannot_shift_components(self) -> None:
        self.assertNotEqual(
            get_document_key("bot", "a:b", "c"),
            get_document_key("bot", "a", "b:c"),
        )

    def test_message_key_cannot_impersonate_a_document_set(self) -> None:
        self.assertNotEqual(get_messages_key("bot", "documents:42"), get_documents_key("bot", 42))

    def test_numeric_keys_stay_byte_identical(self) -> None:
        self.assertEqual(get_documents_key("bot", 42), "bot:documents:42")
        self.assertEqual(get_document_key("bot", 42, "doc"), "bot:documents:42:doc")
        self.assertEqual(get_messages_key("bot", 42), "bot:42")

    def test_blank_keys_are_rejected(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    get_documents_key("bot", value)
                with self.assertRaises(ValueError):
                    get_document_key("bot", value, "doc")


if __name__ == "__main__":
    unittest.main()
