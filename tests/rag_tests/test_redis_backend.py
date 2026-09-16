"""Redis RAG backend tests: stub async client, no live server required."""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from langchain_core.embeddings.fake import FakeEmbeddings

from manolo_bot.rag.factory import build_rag_backend
from manolo_bot.rag.redis_backend import RedisRAGBackend
from manolo_bot.rag.sources import RAGSource

REPO_ROOT = Path(__file__).resolve().parents[2]


class _StubRedis:
    """In-memory async stub of the Redis commands the backend uses."""

    def __init__(self, corrupt_keys: tuple[str, ...] = ()) -> None:
        self.data: dict[str, bytes] = {}
        self.corrupt_keys = set(corrupt_keys)
        self.closed = False

    async def get(self, key: str):
        if key in self.corrupt_keys:
            return b"{corrupt"
        return self.data.get(key)

    async def set(self, key: str, value) -> None:
        self.data[key] = value if isinstance(value, bytes) else str(value).encode()

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                removed += 1
        return removed

    async def aclose(self) -> None:
        self.closed = True


class _OtherEmbeddings(FakeEmbeddings):
    """FakeEmbeddings variant with a distinct model id for mismatch tests."""

    model: str = "other-model"


def _counting_embeddings(size: int = 32):
    """FakeEmbeddings that records every embed_documents call batch size."""
    calls: list[int] = []

    class _Counting(FakeEmbeddings):
        def embed_documents(self, texts):
            calls.append(len(texts))
            return super().embed_documents(texts)

        async def aembed_documents(self, texts):
            calls.append(len(texts))
            return await super().aembed_documents(texts)

    return _Counting(size=size), calls


def _make_backend(stub, embeddings, bot_uuid: str = "test-bot", tmp: str = "") -> RedisRAGBackend:
    return RedisRAGBackend(
        embeddings=embeddings,
        bot_uuid=bot_uuid,
        store_path=str(Path(tmp) / "store") if tmp else None,
        chunk_size=200,
        chunk_overlap=20,
        top_k=10,
        redis_client=stub,
    )


def _write(tmp: str, name: str, content: str) -> Path:
    path = Path(tmp) / name
    path.write_text(content, encoding="utf-8")
    return path


class TestRedisRoundtrip(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_then_fresh_load_with_zero_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubRedis()
            embeddings, calls = _counting_embeddings()
            backend = _make_backend(stub, embeddings, tmp=tmp)
            self.assertFalse(await backend.build_or_load())
            doc = _write(tmp, "doc.md", "Manolo guide. Unique token: token-one.\n" * 10)
            record = RAGSource(path=str(doc))

            first = await backend.ingest([record])
            self.assertGreater(first, 0)
            self.assertIn("test-bot:rag:vectors", stub.data)
            self.assertIn("test-bot:rag:meta", stub.data)
            self.assertIn("test-bot:rag:manifest", stub.data)
            calls_after_first = list(calls)
            self.assertTrue(calls_after_first)

            # Fresh instance over the same stub: loads without ingesting.
            fresh = _make_backend(stub, embeddings, tmp=tmp)
            self.assertTrue(await fresh.build_or_load())
            results = await fresh.query("Manolo guide", top_k=10)
            self.assertTrue(any("token-one" in chunk.text for chunk in results))

            # Re-ingesting unchanged content: no new chunks, ZERO embedding calls.
            self.assertEqual(await fresh.ingest([record]), 0)
            self.assertEqual(calls, calls_after_first)

    async def test_manifest_roundtrip_through_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubRedis()
            backend = _make_backend(stub, FakeEmbeddings(size=32), tmp=tmp)
            await backend.build_or_load()
            doc = _write(tmp, "doc.md", "Manolo guide.\n")
            await backend.ingest([RAGSource(path=str(doc))])

            manifest = json.loads(stub.data["test-bot:rag:manifest"].decode())
            self.assertIn(str(doc.resolve()), manifest)

            fresh = _make_backend(stub, FakeEmbeddings(size=32), tmp=tmp)
            await fresh.build_or_load()
            self.assertEqual(fresh.list_sources(), [str(doc.resolve())])

    async def test_replace_without_ghosts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubRedis()
            backend = _make_backend(stub, FakeEmbeddings(size=32), tmp=tmp)
            await backend.build_or_load()
            doc = _write(tmp, "doc.md", "Original content. Unique token: token-old.\n" * 10)
            record = RAGSource(path=str(doc))
            await backend.ingest([record])

            doc.write_text("Rewritten content. Unique token: token-new.\n" * 10, encoding="utf-8")
            await backend.ingest([record])

            texts = " ".join(chunk.text for chunk in await backend.query("content", top_k=1000))
            self.assertIn("token-new", texts)
            self.assertNotIn("token-old", texts)


class TestRedisRemove(unittest.IsolatedAsyncioTestCase):
    async def test_remove_flow_and_list_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubRedis()
            backend = _make_backend(stub, FakeEmbeddings(size=32), tmp=tmp)
            await backend.build_or_load()
            alpha = _write(tmp, "alpha.md", "Alpha content. Unique token: token-alpha.\n")
            beta = _write(tmp, "beta.md", "Beta content. Unique token: token-beta.\n")
            rec_alpha = RAGSource(path=str(alpha))
            rec_beta = RAGSource(path=str(beta))
            await backend.ingest([rec_alpha, rec_beta])
            self.assertEqual(backend.list_sources(), sorted([str(alpha.resolve()), str(beta.resolve())]))

            removed = await backend.remove([rec_alpha])
            self.assertGreater(removed, 0)
            texts = " ".join(chunk.text for chunk in await backend.query("content", top_k=1000))
            self.assertNotIn("token-alpha", texts)
            self.assertIn("token-beta", texts)
            self.assertEqual(backend.list_sources(), [str(beta.resolve())])

            # Snapshot re-save comes free via inheritance: a fresh instance
            # loads the index without the removed document.
            fresh = _make_backend(stub, FakeEmbeddings(size=32), tmp=tmp)
            self.assertTrue(await fresh.build_or_load())
            texts = " ".join(chunk.text for chunk in await fresh.query("content", top_k=1000))
            self.assertNotIn("token-alpha", texts)
            self.assertIn("token-beta", texts)

    async def test_remove_unknown_warns_and_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubRedis()
            backend = _make_backend(stub, FakeEmbeddings(size=32), tmp=tmp)
            await backend.build_or_load()
            ghost = _write(tmp, "ghost.md", "Never indexed.\n")
            with self.assertLogs("manolo_bot.rag.inmemory_backend", level="WARNING") as logs:
                removed = await backend.remove([RAGSource(path=str(ghost))])
            self.assertEqual(removed, 0)
            self.assertTrue(any("matched nothing indexed" in line for line in logs.output))

    async def test_aclose_releases_client(self) -> None:
        stub = _StubRedis()
        backend = _make_backend(stub, FakeEmbeddings(size=8))
        await backend.aclose()
        self.assertTrue(stub.closed)
        self.assertIsNone(backend._redis)


class TestRedisFailures(unittest.IsolatedAsyncioTestCase):
    async def test_model_mismatch_clears_and_forces_reingest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubRedis()
            backend = _make_backend(stub, FakeEmbeddings(size=32), tmp=tmp)
            await backend.build_or_load()
            doc = _write(tmp, "doc.md", "Manolo guide. Unique token: token-one.\n")
            await backend.ingest([RAGSource(path=str(doc))])
            self.assertTrue(stub.data)

            other = _make_backend(stub, _OtherEmbeddings(size=32), tmp=tmp)
            self.assertFalse(await other.build_or_load())
            self.assertNotIn("test-bot:rag:vectors", stub.data)
            self.assertNotIn("test-bot:rag:meta", stub.data)
            self.assertNotIn("test-bot:rag:manifest", stub.data)
            self.assertEqual(await other.query("Manolo", top_k=10), [])

    async def test_corrupt_vectors_returns_false(self) -> None:
        stub = _StubRedis()
        stub.data["test-bot:rag:meta"] = json.dumps({"embedding_model": "FakeEmbeddings", "format": 1}).encode()
        stub.data["test-bot:rag:vectors"] = b"{corrupt"
        backend = _make_backend(stub, FakeEmbeddings(size=32))
        self.assertFalse(await backend.build_or_load())

    async def test_corrupt_meta_returns_false(self) -> None:
        stub = _StubRedis()
        stub.data["test-bot:rag:vectors"] = b"{}"
        stub.data["test-bot:rag:meta"] = b"{corrupt"
        backend = _make_backend(stub, FakeEmbeddings(size=32))
        self.assertFalse(await backend.build_or_load())

    async def test_unreachable_server_returns_false(self) -> None:
        class _DeadRedis(_StubRedis):
            async def get(self, key: str):
                raise ConnectionError("down")

        backend = _make_backend(_DeadRedis(), FakeEmbeddings(size=32))
        self.assertFalse(await backend.build_or_load())


class TestRedisFactoryAndWiring(unittest.TestCase):
    def test_factory_routes_redis_and_rejects_unknown(self) -> None:
        backend = build_rag_backend(
            "redis",
            FakeEmbeddings(size=32),
            bot_uuid="test-bot",
            redis_url="redis://example:6379/0",
        )
        self.assertIsInstance(backend, RedisRAGBackend)
        self.assertEqual(backend._redis_url, "redis://example:6379/0")
        with self.assertRaises(ValueError):
            build_rag_backend("chroma", FakeEmbeddings(size=32))

    def test_main_passes_redis_url(self) -> None:
        source = (REPO_ROOT / "manolo_bot" / "main.py").read_text(encoding="utf-8")
        self.assertIn("redis_url=config.redis_url", source)


class TestLiveRedisSmoke(unittest.IsolatedAsyncioTestCase):
    async def test_live_roundtrip_if_available(self) -> None:
        try:
            import redis.asyncio as redis_asyncio

            client = redis_asyncio.Redis.from_url(
                "redis://localhost:6379/0", socket_connect_timeout=1, socket_timeout=2
            )
            await client.ping()
        except Exception:
            self.skipTest("no live redis on localhost:6379")
            return
        bot_uuid = f"smoke-{uuid.uuid4().hex[:8]}"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                backend = RedisRAGBackend(
                    embeddings=FakeEmbeddings(size=8),
                    bot_uuid=bot_uuid,
                    store_path=str(Path(tmp) / "store"),
                    chunk_size=200,
                    chunk_overlap=20,
                    top_k=10,
                    redis_client=client,
                )
                self.assertFalse(await backend.build_or_load())
                doc = _write(tmp, "doc.md", "Live smoke content. Unique token: token-live.\n")
                self.assertGreater(await backend.ingest([RAGSource(path=str(doc))]), 0)
                fresh = RedisRAGBackend(
                    embeddings=FakeEmbeddings(size=8),
                    bot_uuid=bot_uuid,
                    store_path=str(Path(tmp) / "store"),
                    redis_client=client,
                )
                self.assertTrue(await fresh.build_or_load())
                texts = " ".join(c.text for c in await fresh.query("smoke", top_k=10))
                self.assertIn("token-live", texts)
                await backend.clear()
                self.assertEqual(await client.exists(f"{bot_uuid}:rag:vectors"), 0)
        finally:
            await client.aclose()


if __name__ == "__main__":
    unittest.main()
