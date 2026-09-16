"""Phase 1 tests for the in-memory RAG backend (no network/API keys)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings.fake import FakeEmbeddings

from manolo_bot.rag.factory import build_rag_backend
from manolo_bot.rag.inmemory_backend import InMemoryRAGBackend
from manolo_bot.rag.sources import RAGSource


def _make_backend(store_dir: str, bot_uuid: str = "test-bot") -> InMemoryRAGBackend:
    return InMemoryRAGBackend(
        embeddings=FakeEmbeddings(size=32),
        bot_uuid=bot_uuid,
        store_path=store_dir,
        chunk_size=200,
        chunk_overlap=20,
        top_k=2,
    )


class TestInMemoryRAGBackend(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_and_query_returns_chunks_with_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "docs"
            data_dir.mkdir()
            md_file = data_dir / "guide.md"
            md_file.write_text("# Manolo\nManolo is a helpful Telegram assistant.\n", encoding="utf-8")
            txt_file = data_dir / "notes.txt"
            txt_file.write_text("The assistant supports web search and documents.\n", encoding="utf-8")

            backend = _make_backend(store_dir=str(Path(tmp) / "store"))
            loaded = await backend.build_or_load()
            self.assertFalse(loaded)

            count = await backend.ingest([RAGSource(str(md_file)), RAGSource(str(txt_file))])
            self.assertGreater(count, 0)

            results = await backend.query("What is Manolo?")
            self.assertGreater(len(results), 0)
            self.assertTrue(all(r.text.strip() for r in results))
            self.assertTrue(all(r.source for r in results))
            sources = " ".join(r.source for r in results)
            self.assertTrue("guide.md" in sources or "notes.txt" in sources)

    async def test_needs_reindex_detects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "doc.txt"
            data_file.write_text("original content about manolo bot\n", encoding="utf-8")

            backend = _make_backend(store_dir=str(Path(tmp) / "store"))
            await backend.build_or_load()
            await backend.ingest([RAGSource(str(data_file))])

            self.assertEqual(backend.needs_reindex([RAGSource(str(data_file))]), [])

            data_file.write_text("original content about manolo bot plus much more text here\n", encoding="utf-8")
            stale = backend.needs_reindex([RAGSource(str(data_file))])
            self.assertEqual(stale, [RAGSource(str(data_file))])

    async def test_build_or_load_false_when_only_manifest_persists(self) -> None:
        # Vectors are process-local: a fresh backend must report that ingest
        # is required even when a manifest survived a restart.
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "doc.txt"
            data_file.write_text("Manolo bot knowledge base content.\n", encoding="utf-8")
            store = str(Path(tmp) / "store")

            backend = _make_backend(store_dir=store)
            await backend.build_or_load()
            await backend.ingest([RAGSource(str(data_file))])

            restarted = _make_backend(store_dir=store)
            self.assertFalse(await restarted.build_or_load())

    async def test_ingest_warns_on_unmatched_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = _make_backend(store_dir=str(Path(tmp) / "store"))
            await backend.build_or_load()
            with self.assertLogs("manolo_bot.rag.inmemory_backend", level="WARNING") as logs:
                count = await backend.ingest([RAGSource(str(Path(tmp) / "missing.pdf"))])
            self.assertEqual(count, 0)
            self.assertTrue(any("matched no files" in line for line in logs.output))

    async def test_manifest_is_memory_only(self) -> None:
        # Vectors never survive a restart, so persisting the manifest to disk
        # would be write-only: no manifest file may appear on disk, while
        # in-process change detection keeps working.
        from manolo_bot.rag.base import manifest_path

        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "doc.txt"
            data_file.write_text("original content about manolo bot\n", encoding="utf-8")
            store = str(Path(tmp) / "store")

            backend = _make_backend(store_dir=store)
            await backend.build_or_load()
            await backend.ingest([RAGSource(str(data_file))])
            self.assertFalse(manifest_path(store, "test-bot").is_file())
            self.assertEqual(backend.needs_reindex([RAGSource(str(data_file))]), [])

            restarted = _make_backend(store_dir=store)
            await restarted.build_or_load()
            self.assertEqual(restarted.needs_reindex([RAGSource(str(data_file))]), [RAGSource(str(data_file))])

    async def test_clear_empties_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "doc.md"
            data_file.write_text("Manolo bot knowledge base content.\n", encoding="utf-8")

            backend = _make_backend(store_dir=str(Path(tmp) / "store"))
            await backend.build_or_load()
            await backend.ingest([RAGSource(str(data_file))])
            before = await backend.query("Manolo")
            self.assertGreater(len(before), 0)

            await backend.clear()
            after = await backend.query("Manolo")
            self.assertEqual(after, [])

    async def test_as_tool_and_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = build_rag_backend(
                "in_memory",
                embeddings=FakeEmbeddings(size=32),
                bot_uuid="test-bot",
                store_path=str(Path(tmp) / "store"),
            )
            self.assertIsInstance(backend, InMemoryRAGBackend)
            rag_tool = backend.as_tool("rag_search", "Search the knowledge base.")
            self.assertEqual(rag_tool.name, "rag_search")

            with self.assertRaises(ValueError):
                build_rag_backend("chroma", embeddings=FakeEmbeddings(size=32))


if __name__ == "__main__":
    unittest.main()
