"""Tests for the persistent local_fs RAG backend (FakeEmbeddings, no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings.fake import FakeEmbeddings

from manolo_bot.rag.factory import build_rag_backend
from manolo_bot.rag.filesystem_backend import FilesystemRAGBackend, _embedding_id


class _AltFakeEmbeddings(FakeEmbeddings):
    """FakeEmbeddings variant with a distinct model id for mismatch tests."""

    model: str = "alt-embedding-model"


def _make_backend(store: str, embeddings=None, bot_uuid: str = "test-bot") -> FilesystemRAGBackend:
    return FilesystemRAGBackend(
        embeddings=embeddings or FakeEmbeddings(size=32),
        bot_uuid=bot_uuid,
        store_path=store,
        chunk_size=200,
        chunk_overlap=20,
        top_k=10,
    )


def _namespace(store: str, bot_uuid: str = "test-bot") -> Path:
    return Path(store) / bot_uuid


class TestFilesystemRAGBackend(unittest.IsolatedAsyncioTestCase):
    async def test_roundtrip_persists_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "handbook.md"
            src.write_text("The deployment runbook lives here. Unique token: runbook-alpha.\n", encoding="utf-8")

            backend = _make_backend(str(Path(tmp) / "store"))
            self.assertFalse(await backend.build_or_load())
            count = await backend.ingest([str(src)])
            self.assertGreater(count, 0)
            ns = _namespace(str(Path(tmp) / "store"))
            self.assertTrue((ns / "vectors.json").is_file())
            self.assertTrue((ns / "meta.json").is_file())
            self.assertTrue((ns / "manifest.json").is_file())

            # Fresh instance over the same store: loads without ingesting.
            fresh = _make_backend(str(Path(tmp) / "store"))
            self.assertTrue(await fresh.build_or_load())
            results = await fresh.query("deployment runbook", top_k=10)
            self.assertGreater(len(results), 0)
            texts = " ".join(r.text for r in results)
            self.assertIn("runbook-alpha", texts)
            self.assertTrue(all(r.source for r in results))

    async def test_model_mismatch_clears_and_forces_reingest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "store")
            src = Path(tmp) / "handbook.md"
            src.write_text("The deployment runbook lives here. Unique token: runbook-alpha.\n", encoding="utf-8")

            backend = _make_backend(store)
            await backend.build_or_load()
            await backend.ingest([str(src)])

            other = _make_backend(store, embeddings=_AltFakeEmbeddings(size=32))
            self.assertFalse(await other.build_or_load())
            ns = _namespace(store)
            self.assertFalse((ns / "vectors.json").exists())
            self.assertFalse((ns / "meta.json").exists())
            self.assertFalse((ns / "manifest.json").exists())
            self.assertEqual(await other.query("deployment", top_k=10), [])
            # Manifest cleared, so the source is stale again (re-ingest forced).
            self.assertEqual(other.needs_reindex([str(src)]), [str(src)])

    async def test_incremental_update_replaces_stale_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "store")
            src = Path(tmp) / "notes.md"
            src.write_text("Alpha notes. Unique token: token-alpha.\n", encoding="utf-8")

            backend = _make_backend(store)
            await backend.build_or_load()
            await backend.ingest([str(src)])

            src.write_text("Beta notes, considerably longer than before. Unique token: token-beta.\n", encoding="utf-8")
            count = await backend.ingest([str(src)])
            self.assertGreater(count, 0)

            results = await backend.query("token-beta", top_k=10)
            texts = " ".join(r.text for r in results)
            self.assertIn("token-beta", texts)
            self.assertNotIn("token-alpha", texts)
            self.assertEqual(len(results), 1)

    async def test_clear_removes_dump_meta_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "store")
            src = Path(tmp) / "handbook.md"
            src.write_text("The deployment runbook lives here. Unique token: runbook-alpha.\n", encoding="utf-8")

            backend = _make_backend(store)
            await backend.build_or_load()
            await backend.ingest([str(src)])
            ns = _namespace(store)
            self.assertTrue((ns / "vectors.json").is_file())

            await backend.clear()
            self.assertFalse((ns / "vectors.json").exists())
            self.assertFalse((ns / "meta.json").exists())
            self.assertFalse((ns / "manifest.json").exists())
            self.assertEqual(await backend.query("deployment", top_k=10), [])

    async def test_corrupt_snapshot_never_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "store")
            ns = _namespace(store)
            ns.mkdir(parents=True, exist_ok=True)
            (ns / "vectors.json").write_text("{not valid json", encoding="utf-8")
            (ns / "meta.json").write_text("{not valid json", encoding="utf-8")

            backend = _make_backend(store)
            self.assertFalse(await backend.build_or_load())

    def test_embedding_id_falls_back_to_class_name(self) -> None:
        self.assertEqual(_embedding_id(FakeEmbeddings(size=8)), "FakeEmbeddings")
        self.assertEqual(_embedding_id(_AltFakeEmbeddings(size=8)), "alt-embedding-model")

    def test_factory_routes_local_fs_and_rejects_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = build_rag_backend(
                "local_fs",
                FakeEmbeddings(size=32),
                bot_uuid="test-bot",
                store_path=str(Path(tmp) / "store"),
            )
            self.assertIsInstance(backend, FilesystemRAGBackend)
        with self.assertRaises(ValueError):
            build_rag_backend("chroma", FakeEmbeddings(size=32))


if __name__ == "__main__":
    unittest.main()
