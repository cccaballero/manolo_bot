"""Core RAG abstractions and manifest fingerprinting helpers (Phase 1)."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from manolo_bot.rag.sources import RAGSource

#: Canonical name of the retriever tool a RAG backend exposes to agents.
#: Agents check for this name so an MCP-provided tool can shadow the backend one.
RAG_TOOL_NAME = "rag_search"


@dataclass
class RAGChunk:
    """A single retrieved chunk of document text."""

    text: str
    source: str
    score: float | None = None


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """Compute a fingerprint for a single file.

    The fingerprint captures path, mtime, size and sha256 so content
    changes are detected even when mtime granularity is coarse.

    :param path: File to fingerprint.
    :return: Dict with ``path``, ``mtime``, ``size`` and ``sha256`` keys.
    :raises FileNotFoundError: If the path does not exist or is not a file.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    stat = file_path.stat()
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(block)
    return {
        "path": str(file_path.resolve()),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def fingerprint_paths(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """Fingerprint a list of files, skipping missing entries.

    :param paths: Candidate file paths.
    :return: Mapping of resolved path string to fingerprint dict.
    """
    manifest: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            fingerprint = fingerprint_file(path)
        except OSError:
            continue
        manifest[fingerprint["path"]] = fingerprint
    return manifest


def manifest_path(store_path: str | Path, bot_uuid: str) -> Path:
    """Return the manifest file location for a bot namespace.

    Layout: ``<store_path>/<bot_uuid>/manifest.json``.

    :param store_path: Root RAG store directory (RAG_STORE_PATH).
    :param bot_uuid: Bot namespace.
    """
    return Path(store_path) / bot_uuid / "manifest.json"


def load_manifest(store_path: str | Path, bot_uuid: str) -> dict[str, dict[str, Any]]:
    """Load a bot manifest from disk, returning {} when absent/corrupt.

    :param store_path: Root RAG store directory.
    :param bot_uuid: Bot namespace.
    """
    path = manifest_path(store_path, bot_uuid)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_manifest(store_path: str | Path, bot_uuid: str, manifest: dict[str, dict[str, Any]]) -> Path:
    """Persist a bot manifest to disk, creating parent dirs as needed."""
    path = manifest_path(store_path, bot_uuid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def find_stale_paths(paths: Sequence[str | Path], manifest: dict[str, dict[str, Any]]) -> list[str]:
    """Return the subset of paths that are new or changed vs. the manifest.

    Missing files are skipped (they cannot be ingested). A fingerprint match
    means fresh, with two exceptions: entries recording ``chunks == 0`` never
    indexed anything, so they are always retried; entries without a ``chunks``
    key predate chunk-count tracking and are re-ingested once to heal them.

    :param paths: Concrete file paths to check (already glob-expanded).
    :param manifest: Previously stored fingerprints keyed by resolved path.
    """
    stale: list[str] = []
    for path in paths:
        file_path = Path(path)
        if not file_path.is_file():
            continue
        try:
            current = fingerprint_file(file_path)
        except OSError:
            continue
        previous = manifest.get(current["path"])
        if previous is None:
            stale.append(str(file_path))
            continue
        if (
            previous.get("mtime") != current["mtime"]
            or previous.get("size") != current["size"]
            or previous.get("sha256") != current["sha256"]
        ):
            stale.append(str(file_path))
            continue
        if previous.get("chunks", 0) == 0:
            # Fingerprint matches but nothing was ever indexed (or the entry
            # predates chunk-count tracking): retry instead of staying empty.
            stale.append(str(file_path))
    return stale


class BaseRAGBackend(ABC):
    """Abstract RAG backend contract (Phase 1 seam for Chroma/FAISS later)."""

    @abstractmethod
    async def build_or_load(self) -> bool:
        """Prepare the store, loading persisted state when available.

        :return: True when previously indexed vectors are queryable without
            ingesting again, False when the caller must ingest sources first
            (e.g. ephemeral backends after a restart).
        """
        raise NotImplementedError

    @abstractmethod
    async def ingest(self, paths: Sequence[RAGSource]) -> int:
        """Ingest documents from file paths or glob patterns.

        Idempotent: files whose fingerprint matches the manifest are skipped
        without any embedding call; new files are added, changed files are
        replaced (old vectors dropped first), so re-ingesting never duplicates
        and never pays for already-indexed content.

        :param paths: Files, directories or glob patterns to ingest, as
            ``RAGSource`` records.
        :return: Number of NEW chunks added to the store (0 when everything
            was already fresh).
        """
        raise NotImplementedError

    @abstractmethod
    async def query(self, query: str, top_k: int | None = None) -> list[RAGChunk]:
        """Retrieve the most relevant chunks for a query."""
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        """Remove all indexed data and persisted manifest state."""
        raise NotImplementedError

    @abstractmethod
    async def remove(self, paths: Sequence[RAGSource]) -> int:
        """Drop indexed documents without touching anything else.

        Expands the given records, deletes vectors whose stored source matches,
        and prunes their manifest fingerprints. Entries matching nothing indexed
        warn and are skipped. The inverse of ``ingest`` — the only operation
        besides ``clear`` that deletes.

        :param paths: Source records identifying what to forget.
        :return: Number of chunks removed from the store.
        """
        raise NotImplementedError

    @abstractmethod
    def list_sources(self) -> list[str]:
        """Return sorted indexed file paths (manifest keys)."""
        raise NotImplementedError

    @abstractmethod
    def needs_reindex(self, paths: Sequence[RAGSource]) -> list[RAGSource]:
        """Return the subset of records with new or changed files."""
        raise NotImplementedError

    @abstractmethod
    def find_removed(self, paths: Sequence[RAGSource]) -> list[str]:
        """Return indexed files matching none of the given records.

        Compares manifest keys (resolved paths) against the currently expanded
        records: anything indexed but no longer configured. Used by ``auto``
        reindexing to mirror the configured sources.
        """
        raise NotImplementedError

    @abstractmethod
    def as_tool(self, name: str, description: str) -> BaseTool:
        """Expose this backend as a LangChain retriever tool."""
        raise NotImplementedError

    @abstractmethod
    def as_source_tools(self, entries: Sequence[RAGSource]) -> list[BaseTool]:
        """Expose one retriever tool per structured source entry.

        :param entries: Source entries as ``RAGSource`` records.
        """
        raise NotImplementedError
