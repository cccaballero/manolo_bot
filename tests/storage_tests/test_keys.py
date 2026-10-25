"""Tests for canonical storage keys and path segments."""

from __future__ import annotations

import unittest

from manolo_bot.storage.keys import normalize_storage_key, storage_path_segment


class TestNormalizeStorageKey(unittest.TestCase):
    def test_numeric_keys_become_strings(self) -> None:
        self.assertEqual(normalize_storage_key(42), "42")
        self.assertEqual(normalize_storage_key("42"), "42")
        self.assertEqual(normalize_storage_key(-1001234567890), "-1001234567890")

    def test_opaque_keys_are_preserved(self) -> None:
        self.assertEqual(normalize_storage_key("C012345"), "C012345")
        self.assertEqual(normalize_storage_key("dashboard:user:agent:uuid"), "dashboard:user:agent:uuid")

    def test_padding_is_stripped(self) -> None:
        self.assertEqual(normalize_storage_key(" C012345\t"), "C012345")
        self.assertEqual(normalize_storage_key(" 42 "), normalize_storage_key(42))

    def test_blank_values_raise(self) -> None:
        for value in (None, "", "   ", "\n\t "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_storage_key(value)

    def test_normalization_is_idempotent(self) -> None:
        once = normalize_storage_key(" dashboard:user ")
        self.assertEqual(normalize_storage_key(once), once)


class TestStoragePathSegment(unittest.TestCase):
    def test_numeric_keys_keep_their_existing_directory(self) -> None:
        self.assertEqual(storage_path_segment(42), "42")
        self.assertEqual(storage_path_segment("-1001234567890"), "-1001234567890")

    def test_safe_keys_are_used_verbatim(self) -> None:
        self.assertEqual(storage_path_segment("team-123"), "team-123")
        self.assertEqual(storage_path_segment("a" * 64), "a" * 64)

    def test_unsafe_keys_are_hashed(self) -> None:
        for key in (
            "C012345",
            "dashboard:user:agent:uuid",
            "../../etc/passwd",
            "..",
            "with space",
            "a" * 65,
            "ñandú",
        ):
            with self.subTest(key=key):
                segment = storage_path_segment(key)
                self.assertTrue(segment.startswith("h."), segment)
                self.assertEqual(len(segment), 42)
                self.assertNotIn("/", segment)
                self.assertNotIn("\\", segment)
                self.assertEqual(segment, segment.lower())

    def test_hashing_is_deterministic_and_case_sensitive(self) -> None:
        self.assertEqual(storage_path_segment("C012345"), storage_path_segment("C012345"))
        self.assertNotEqual(storage_path_segment("C012345"), storage_path_segment("c012345"))

    def test_a_key_matching_a_hashed_segment_is_not_returned_verbatim(self) -> None:
        # '.' is not allowed in a safe segment, so the "h." prefix can never be
        # produced by a retained key — no collision between the two forms.
        hashed_shape = "h." + "a" * 40
        self.assertNotEqual(storage_path_segment(hashed_shape), hashed_shape)

    def test_padding_is_ignored(self) -> None:
        self.assertEqual(storage_path_segment(" 42 "), "42")
        self.assertEqual(storage_path_segment(" C012345 "), storage_path_segment("C012345"))


if __name__ == "__main__":
    unittest.main()
