"""Ingest resilience tests: per-file isolation + size cap (no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings

from manolo_bot.rag.filesystem_backend import FilesystemRAGBackend
from manolo_bot.rag.inmemory_backend import InMemoryRAGBackend
from manolo_bot.rag.sources import RAGSource


class _FlakyEmbeddings(Embeddings):
    """Deterministic embeddings that raise on texts containing a marker."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "BAD-MARKER" in text:
                raise RuntimeError("embedding boom")
            vectors.append([1.0] * 8)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return [1.0] * 8


def _make_inmemory(store: str, embeddings=None, **kwargs) -> InMemoryRAGBackend:
    return InMemoryRAGBackend(
        embeddings=embeddings or FakeEmbeddings(size=32),
        bot_uuid="test-bot",
        store_path=store,
        chunk_size=200,
        chunk_overlap=20,
        top_k=10,
        **kwargs,
    )


class TestPerFileIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_one_bad_file_does_not_kill_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.md"
            good.write_text("Good content. Unique token: token-good.\n", encoding="utf-8")
            bad = Path(tmp) / "bad.md"
            bad.write_text("Bad content BAD-MARKER. Unique token: token-bad.\n", encoding="utf-8")

            backend = _make_inmemory(str(Path(tmp) / "store"), embeddings=_FlakyEmbeddings())
            await backend.build_or_load()
            with self.assertLogs("manolo_bot.rag.inmemory_backend", level="WARNING"):
                count = await backend.ingest([RAGSource(str(good)), RAGSource(str(bad))])
            self.assertGreater(count, 0)

            results = await backend.query("token-good", top_k=10)
            self.assertIn("token-good", " ".join(r.text for r in results))
            # Good file done; bad file left unfingerprinted for retry.
            self.assertEqual(backend.needs_reindex([RAGSource(str(good))]), [])
            self.assertEqual(backend.needs_reindex([RAGSource(str(bad))]), [RAGSource(str(bad))])

    async def test_size_cap_skips_and_fingerprints_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.md"
            big.write_text("x" * 100 + "\n", encoding="utf-8")

            backend = _make_inmemory(str(Path(tmp) / "store"), max_file_bytes=10)
            await backend.build_or_load()
            with self.assertLogs("manolo_bot.rag.inmemory_backend", level="WARNING") as logs:
                count = await backend.ingest([RAGSource(str(big))])
            self.assertEqual(count, 0)
            self.assertTrue(any("RAG_MAX_FILE_BYTES" in message for message in logs.output))
            # Fingerprinted, so the warning fires once instead of every run.
            self.assertEqual(backend.needs_reindex([RAGSource(str(big))]), [])


class TestLocalFsFailedReingest(unittest.IsolatedAsyncioTestCase):
    async def test_failed_reingest_stays_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "store")
            src = Path(tmp) / "notes.md"
            src.write_text("Version one content here.\n", encoding="utf-8")

            backend = FilesystemRAGBackend(
                embeddings=FakeEmbeddings(size=32),
                bot_uuid="test-bot",
                store_path=store,
                chunk_size=200,
                chunk_overlap=20,
                top_k=10,
            )
            await backend.build_or_load()
            await backend.ingest([RAGSource(str(src))])

            src.write_text("Version two content, considerably longer than version one was.\n", encoding="utf-8")
            with patch.object(backend._store, "aadd_documents", side_effect=RuntimeError("boom")):
                count = await backend.ingest([RAGSource(str(src))])
            self.assertEqual(count, 0)
            # Pruned manifest persisted before ghost-delete: still flagged, not silently lost.
            self.assertEqual(backend.needs_reindex([RAGSource(str(src))]), [RAGSource(str(src))])


if __name__ == "__main__":
    unittest.main()
