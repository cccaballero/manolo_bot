"""RAG document lifecycle: idempotent ingest, remove, list_sources (no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings.fake import FakeEmbeddings

from manolo_bot.rag.filesystem_backend import FilesystemRAGBackend
from manolo_bot.rag.inmemory_backend import InMemoryRAGBackend
from manolo_bot.rag.sources import RAGSource


def _counting_embeddings(size: int = 32):
    """FakeEmbeddings that records every embed_documents call (call count, batch size)."""
    calls: list[int] = []

    class _Counting(FakeEmbeddings):
        def embed_documents(self, texts):
            calls.append(len(texts))
            return super().embed_documents(texts)

        async def aembed_documents(self, texts):
            calls.append(len(texts))
            return await super().aembed_documents(texts)

    return _Counting(size=size), calls


def _make_backend(kind: str, tmp: str, embeddings):
    kwargs = {
        "embeddings": embeddings,
        "bot_uuid": "test-bot",
        "store_path": str(Path(tmp) / "store"),
        "chunk_size": 200,
        "chunk_overlap": 20,
        "top_k": 10,
    }
    if kind == "local_fs":
        return FilesystemRAGBackend(**kwargs)
    return InMemoryRAGBackend(**kwargs)


def _texts(chunks) -> str:
    return " ".join(chunk.text for chunk in chunks)


class LifecycleCase:
    """Mix-in: same lifecycle contract verified on both backends."""

    KIND = "in_memory"

    def _write(self, tmp: str, name: str, content: str) -> Path:
        path = Path(tmp) / name
        path.write_text(content, encoding="utf-8")
        return path


class TestIdempotentIngest(unittest.IsolatedAsyncioTestCase, LifecycleCase):
    async def _run_case(self, kind: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            embeddings, calls = _counting_embeddings()
            backend = _make_backend(kind, tmp, embeddings)
            await backend.build_or_load()
            doc = self._write(tmp, "doc.md", "Manolo guide. Unique token: token-one.\n" * 10)
            record = RAGSource(path=str(doc))

            first = await backend.ingest([record])
            self.assertGreater(first, 0)
            calls_after_first = list(calls)
            self.assertTrue(calls_after_first)

            # Re-ingesting unchanged content: no new chunks, ZERO embedding calls.
            second = await backend.ingest([record])
            self.assertEqual(second, 0)
            self.assertEqual(calls, calls_after_first)

            # No duplicates: chunk count stable across double ingest.
            before = await backend.query("Manolo", top_k=1000)
            await backend.ingest([record])
            after = await backend.query("Manolo", top_k=1000)
            self.assertEqual(len(before), len(after))
            self.assertGreater(len(after), 0)

    async def test_in_memory(self) -> None:
        await self._run_case("in_memory")

    async def test_local_fs(self) -> None:
        await self._run_case("local_fs")

    async def test_changed_file_replaces_without_ghosts(self) -> None:
        for kind in ("in_memory", "local_fs"):
            with self.subTest(backend=kind), tempfile.TemporaryDirectory() as tmp:
                embeddings, _ = _counting_embeddings()
                backend = _make_backend(kind, tmp, embeddings)
                await backend.build_or_load()
                doc = self._write(tmp, "doc.md", "Original content. Unique token: token-old.\n" * 10)
                record = RAGSource(path=str(doc))
                await backend.ingest([record])

                doc.write_text("Rewritten content. Unique token: token-new.\n" * 10, encoding="utf-8")
                await backend.ingest([record])

                texts = _texts(await backend.query("content", top_k=1000))
                self.assertIn("token-new", texts)
                self.assertNotIn("token-old", texts)


class TestRemove(unittest.IsolatedAsyncioTestCase, LifecycleCase):
    async def _run_case(self, kind: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            embeddings, _ = _counting_embeddings()
            backend = _make_backend(kind, tmp, embeddings)
            await backend.build_or_load()
            alpha = self._write(tmp, "alpha.md", "Alpha content. Unique token: token-alpha.\n")
            beta = self._write(tmp, "beta.md", "Beta content. Unique token: token-beta.\n")
            rec_alpha = RAGSource(path=str(alpha))
            rec_beta = RAGSource(path=str(beta))
            await backend.ingest([rec_alpha, rec_beta])
            self.assertEqual(backend.list_sources(), sorted([str(alpha.resolve()), str(beta.resolve())]))

            removed = await backend.remove([rec_alpha])
            self.assertGreater(removed, 0)
            texts = _texts(await backend.query("content", top_k=1000))
            self.assertNotIn("token-alpha", texts)
            self.assertIn("token-beta", texts)
            self.assertEqual(backend.list_sources(), [str(beta.resolve())])
            # Removed files read as new again (re-addable).
            self.assertEqual(backend.needs_reindex([rec_alpha]), [rec_alpha])

    async def test_in_memory(self) -> None:
        await self._run_case("in_memory")

    async def test_local_fs(self) -> None:
        await self._run_case("local_fs")

    async def test_removal_persists_on_local_fs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            embeddings, _ = _counting_embeddings()
            backend = _make_backend("local_fs", tmp, embeddings)
            await backend.build_or_load()
            alpha = self._write(tmp, "alpha.md", "Alpha content. Unique token: token-alpha.\n")
            beta = self._write(tmp, "beta.md", "Beta content. Unique token: token-beta.\n")
            await backend.ingest([RAGSource(path=str(alpha)), RAGSource(path=str(beta))])
            await backend.remove([RAGSource(path=str(alpha))])

            fresh = _make_backend("local_fs", tmp, embeddings)
            self.assertTrue(await fresh.build_or_load())
            texts = _texts(await fresh.query("content", top_k=1000))
            self.assertNotIn("token-alpha", texts)
            self.assertIn("token-beta", texts)
            self.assertEqual(fresh.list_sources(), [str(beta.resolve())])

    async def test_remove_unknown_warns_and_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            embeddings, _ = _counting_embeddings()
            backend = _make_backend("in_memory", tmp, embeddings)
            await backend.build_or_load()
            ghost = self._write(tmp, "ghost.md", "Never indexed.\n")
            with self.assertLogs("manolo_bot.rag.inmemory_backend", level="WARNING") as logs:
                removed = await backend.remove([RAGSource(path=str(ghost))])
            self.assertEqual(removed, 0)
            self.assertTrue(any("matched nothing indexed" in line for line in logs.output))

    async def test_list_sources_empty_initially(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            embeddings, _ = _counting_embeddings()
            backend = _make_backend("in_memory", tmp, embeddings)
            await backend.build_or_load()
            self.assertEqual(backend.list_sources(), [])


if __name__ == "__main__":
    unittest.main()
