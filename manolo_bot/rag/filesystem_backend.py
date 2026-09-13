"""Persistent single-node RAG backend (local filesystem snapshot)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from manolo_bot.rag.base import load_manifest, save_manifest
from manolo_bot.rag.inmemory_backend import InMemoryRAGBackend, _expand_paths

logger = logging.getLogger(__name__)

_META_FORMAT = 1


def _embedding_id(embeddings: Embeddings) -> str:
    """Stable id for the embedding space (model name or class name)."""
    return str(getattr(embeddings, "model", None) or type(embeddings).__name__)


def vectors_path(store_path: str | Path, bot_uuid: str) -> Path:
    """Return the persisted-vectors file for a bot namespace."""
    return Path(store_path) / bot_uuid / "vectors.json"


def meta_path(store_path: str | Path, bot_uuid: str) -> Path:
    """Return the snapshot metadata file for a bot namespace."""
    return Path(store_path) / bot_uuid / "meta.json"


def _resolve_source(source: str) -> str:
    """Normalize a stored chunk source for comparison with fingerprints."""
    try:
        return str(Path(source).resolve())
    except OSError:
        return source


def _stored_source(entry: Any) -> str:
    """Extract the source from a serialized (or live) stored entry."""
    metadata = entry.get("metadata", {}) if isinstance(entry, dict) else getattr(entry, "metadata", {})
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("source", ""))


class FilesystemRAGBackend(InMemoryRAGBackend):
    """In-memory vectors with a JSON snapshot per bot namespace.

    Layout under ``<store>/<bot_uuid>/``: ``vectors.json`` (store dump),
    ``meta.json`` (embedding model + format) and the shared ``manifest.json``.
    Simple single-node persistence only — not horizontally scalable.
    """

    # Unlike the ephemeral parent, the manifest is load-bearing here: it backs
    # needs_reindex() for incremental updates alongside the vector snapshot.
    _persist_manifest = True

    def _save_snapshot(self) -> None:
        meta = {"embedding_model": _embedding_id(self._embeddings), "format": _META_FORMAT}
        self._store.dump(str(vectors_path(self.store_path, self.bot_uuid)))
        meta_p = meta_path(self.store_path, self.bot_uuid)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        with meta_p.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

    async def build_or_load(self) -> bool:
        """Load the snapshot when present and model-compatible.

        Returns True only when previously indexed vectors are queryable without
        ingesting again. A model mismatch clears the snapshot (embedding spaces
        are incomparable) and returns False to force full re-ingest. Missing or
        corrupt files never raise — they reset state and return False.
        """
        namespaced = self.store_path / self.bot_uuid
        namespaced.mkdir(parents=True, exist_ok=True)
        vectors_p = vectors_path(self.store_path, self.bot_uuid)
        meta_p = meta_path(self.store_path, self.bot_uuid)
        if not vectors_p.is_file() or not meta_p.is_file():
            # No usable snapshot: reset any partial state so the caller re-ingests.
            await self.clear()
            return False
        try:
            with meta_p.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError) as e:
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
            self._store = InMemoryVectorStore.load(str(vectors_p), embedding=self._embeddings)
        except Exception as e:
            logger.warning(f"RAG snapshot vectors unreadable, re-indexing from scratch: {e}")
            await self.clear()
            return False
        self._manifest = load_manifest(self.store_path, self.bot_uuid)
        return True

    async def ingest(self, paths: list[str]) -> int:
        """Re-index files, dropping their stale vectors first to avoid ghosts.

        Pruned manifest keys are persisted BEFORE ghost-deleting, so a failed
        re-embed leaves the file flagged stale for retry instead of silently lost.
        """
        files = _expand_paths(paths)
        if not files:
            return 0
        manifest = self._manifest or load_manifest(self.store_path, self.bot_uuid)
        targets = {str(f.resolve()) for f in files} & set(manifest.keys())
        if targets:
            for key in targets:
                self._manifest.pop(key, None)
            try:
                pruned = load_manifest(self.store_path, self.bot_uuid)
                for key in targets:
                    pruned.pop(key, None)
                save_manifest(self.store_path, self.bot_uuid, pruned)
                self._manifest = pruned
            except OSError as e:
                logger.warning(f"Could not persist pruned RAG manifest: {e}")
            stale_ids = [
                doc_id
                for doc_id, entry in self._store.store.items()
                if _resolve_source(_stored_source(entry)) in targets
            ]
            if stale_ids:
                await self._store.adelete(stale_ids)
        count = await super().ingest(paths)
        if count:
            try:
                self._save_snapshot()
            except OSError as e:
                logger.warning(f"Could not persist RAG vectors: {e}")
        return count

    async def clear(self) -> None:
        """Reset the vector store and remove dump, meta and manifest files."""
        await super().clear()
        for path in (vectors_path(self.store_path, self.bot_uuid), meta_path(self.store_path, self.bot_uuid)):
            try:
                if path.is_file():
                    path.unlink()
            except OSError as e:
                logger.warning(f"Could not remove {path}: {e}")
