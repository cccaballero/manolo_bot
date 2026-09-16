"""Redis-persisted RAG backend (shared snapshot store)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import redis
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.load.load import load
from langchain_core.vectorstores import InMemoryVectorStore

from manolo_bot.rag.filesystem_backend import _META_FORMAT, FilesystemRAGBackend, _embedding_id

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def _as_str(payload: str | bytes | bytearray | memoryview) -> str:
    """Decode a Redis payload (redis-py returns bytes by default)."""
    if isinstance(payload, str):
        return payload
    return bytes(payload).decode("utf-8")


class RedisRAGBackend(FilesystemRAGBackend):
    """Shared/persistent RAG backend with the snapshot in Redis.

    Vectors live in memory after load — queries behave identically on all
    backends. Redis holds the same snapshot layout as ``local_fs`` (vectors
    dump, meta, manifest) under ``{bot_uuid}:rag:*`` keys, so several bot
    processes can share one index.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        bot_uuid: str = "default",
        store_path: str | Path | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        max_file_bytes: int | None = None,
        redis_url: str = "redis://localhost:6379/0",
        redis_client: Any = None,
    ) -> None:
        """Create the backend; connects lazily on first use.

        :param redis_client: Explicit async client (get/set/delete) for tests
            via DI; otherwise ``redis.asyncio`` connects from ``redis_url``.
            Not ``RedisDBHelper``: that helper is built for the per-chat dance
            (connect per chat + disconnect), while this backend is a process
            singleton whose client lives as long as the process.
        """
        super().__init__(
            embeddings=embeddings,
            bot_uuid=bot_uuid,
            store_path=store_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            max_file_bytes=max_file_bytes,
        )
        self._redis_url = redis_url
        self._redis = redis_client

    @property
    def _vectors_key(self) -> str:
        return f"{self.bot_uuid}:rag:vectors"

    @property
    def _meta_key(self) -> str:
        return f"{self.bot_uuid}:rag:meta"

    @property
    def _manifest_key(self) -> str:
        return f"{self.bot_uuid}:rag:manifest"

    async def _client(self):
        """Lazily connected async client (explicit client wins, for tests/DI)."""
        if self._redis is None:
            self._redis = redis.asyncio.Redis.from_url(self._redis_url)
        return self._redis

    def _dump_vectors(self) -> str:
        """Serialize the vector store with the native dump shape (no files)."""
        return json.dumps(dumpd(self._store.store))

    def _restore_vectors(self, payload: str | bytes | bytearray | memoryview) -> None:
        """Rebuild the in-memory store from a snapshot payload."""
        store = load(json.loads(_as_str(payload)), allowed_objects=[Document])
        if not isinstance(store, dict):
            raise ValueError("RAG snapshot vectors have an unexpected shape")
        fresh = InMemoryVectorStore(embedding=self._embeddings)
        fresh.store = store
        self._store = fresh

    def _read_manifest(self) -> dict:
        # Write-through cache: build_or_load populates it, every mutation
        # persists through _write_manifest, so memory is always fresh.
        return self._manifest

    async def _write_manifest(self, manifest: dict) -> None:
        self._manifest = manifest
        try:
            client = await self._client()
            await client.set(self._manifest_key, json.dumps(manifest))
        except Exception as e:
            # Vectors stay indexed; worst case is rework later.
            logger.warning(f"Could not persist RAG manifest: {e}")

    async def _save_snapshot(self) -> None:
        """Persist vectors dump + meta to Redis (best effort, warn-and-continue)."""
        try:
            client = await self._client()
            meta = {"embedding_model": _embedding_id(self._embeddings), "format": _META_FORMAT}
            await client.set(self._vectors_key, self._dump_vectors())
            await client.set(self._meta_key, json.dumps(meta))
        except Exception as e:
            logger.warning(f"Could not persist RAG vectors: {e}")

    async def build_or_load(self) -> bool:
        """Load the Redis snapshot when present and model-compatible.

        Returns True only when previously indexed vectors are queryable without
        ingesting again. A model mismatch clears the snapshot (embedding spaces
        are incomparable) and returns False to force full re-ingest. Missing or
        corrupt keys never raise — they reset state and return False.
        """
        try:
            client = await self._client()
            raw_vectors = await client.get(self._vectors_key)
            raw_meta = await client.get(self._meta_key)
        except Exception as e:
            logger.warning(f"RAG snapshot unreachable, re-indexing from scratch: {e}")
            await self.clear()
            return False
        if raw_vectors is None or raw_meta is None:
            # No usable snapshot: reset any partial state so the caller re-ingests.
            await self.clear()
            return False
        try:
            meta = json.loads(_as_str(raw_meta))
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.warning(f"RAG snapshot meta unreadable, re-indexing from scratch: {e}")
            await self.clear()
            return False
        if not isinstance(meta, dict) or meta.get("format") != _META_FORMAT:
            logger.warning("RAG snapshot format unsupported, re-indexing from scratch")
            await self.clear()
            return False
        if meta.get("embedding_model") != _embedding_id(self._embeddings):
            logger.warning("RAG embedding model changed; clearing persisted vectors for full re-ingest")
            await self.clear()
            return False
        try:
            self._restore_vectors(raw_vectors)
        except Exception as e:
            logger.warning(f"RAG snapshot vectors unreadable, re-indexing from scratch: {e}")
            await self.clear()
            return False
        try:
            raw_manifest = await client.get(self._manifest_key)
            manifest = json.loads(_as_str(raw_manifest)) if raw_manifest is not None else {}
            self._manifest = manifest if isinstance(manifest, dict) else {}
        except Exception as e:
            logger.warning(f"RAG snapshot manifest unreadable, starting fresh: {e}")
            self._manifest = {}
        return True

    async def clear(self) -> None:
        """Reset memory state and delete the Redis snapshot keys (never touches disk)."""
        self._store = InMemoryVectorStore(embedding=self._embeddings)
        self._manifest = {}
        try:
            client = await self._client()
            await client.delete(self._vectors_key, self._meta_key, self._manifest_key)
        except Exception as e:
            logger.warning(f"Could not clear RAG snapshot: {e}")

    async def aclose(self) -> None:
        """Close the Redis client. Library users should call this; main.py keeps the singleton open."""
        client, self._redis = self._redis, None
        if client is None:
            return
        try:
            await client.aclose()
        except Exception as e:
            logger.warning(f"Could not close RAG Redis client: {e}")
