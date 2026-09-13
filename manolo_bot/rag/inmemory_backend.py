"""In-memory RAG backend (Phase 1)."""

from __future__ import annotations

import glob
import logging
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from manolo_bot.rag.base import (
    BaseRAGBackend,
    RAGChunk,
    find_stale_paths,
    fingerprint_paths,
    load_manifest,
    manifest_path,
    save_manifest,
)

logger = logging.getLogger(__name__)

_LOADER_BY_SUFFIX = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}


def _expand_paths(paths: list[str]) -> list[Path]:
    """Expand glob patterns/dirs/files into a sorted, deduplicated file list."""
    expanded: list[Path] = []
    for pattern in paths:
        text = str(pattern)
        before = len(expanded)
        if glob.has_magic(text):
            for match in glob.glob(text, recursive=True):
                candidate = Path(match)
                if candidate.is_file():
                    expanded.append(candidate)
                elif candidate.is_dir():
                    expanded.extend(f for f in candidate.rglob("*") if f.is_file())
        else:
            candidate = Path(text)
            if candidate.is_file():
                expanded.append(candidate)
            elif candidate.is_dir():
                expanded.extend(f for f in candidate.rglob("*") if f.is_file())
        if len(expanded) == before:
            logger.warning(f"RAG source matched no files and will be skipped: {pattern}")
    # Deduplicate by resolved path while keeping deterministic order.
    seen: set[str] = set()
    unique: list[Path] = []
    for path in sorted(expanded):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_documents(files: list[Path]) -> list[Document]:
    """Load documents, picking a loader per file suffix."""
    documents: list[Document] = []
    for file in files:
        loader_cls = _LOADER_BY_SUFFIX.get(file.suffix.lower(), TextLoader)
        try:
            loader = loader_cls(str(file))
            loaded = loader.load()
        except Exception as e:
            logger.warning(f"Skipping unreadable file {file}: {e}")
            continue
        for doc in loaded:
            doc.metadata.setdefault("source", str(file))
            if not doc.page_content or not doc.page_content.strip():
                continue
            documents.append(doc)
    return documents


class InMemoryRAGBackend(BaseRAGBackend):
    """Ephemeral vector-store backend with manifest-based reindex detection."""

    # Manifests are memory-only unless a subclass persists them: vectors never
    # survive a restart here, so an on-disk manifest would be write-only.
    _persist_manifest = False

    def __init__(
        self,
        embeddings: Embeddings,
        bot_uuid: str = "default",
        store_path: str | Path | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        max_file_bytes: int | None = None,
    ) -> None:
        import tempfile

        self._embeddings = embeddings
        self.bot_uuid = bot_uuid
        default_root = Path(tempfile.gettempdir()) / "manolo_bot" / "rag"
        self.store_path = Path(store_path) if store_path is not None else default_root
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._max_file_bytes = max_file_bytes
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._store = InMemoryVectorStore(embedding=embeddings)
        self._manifest: dict = {}

    async def build_or_load(self) -> bool:
        """Reset in-process state; load the on-disk manifest only if persisted.

        Always returns False: vectors live only in process memory, so a caller
        must ingest sources on every fresh backend (e.g. each restart) —
        otherwise RAG would stay empty.
        """
        self._manifest = {}
        if self._persist_manifest:
            namespaced = self.store_path / self.bot_uuid
            namespaced.mkdir(parents=True, exist_ok=True)
            path = manifest_path(self.store_path, self.bot_uuid)
            if path.is_file():
                self._manifest = load_manifest(self.store_path, self.bot_uuid)
        return False

    async def ingest(self, paths: list[str]) -> int:
        """Load, split and index files one by one; return total chunk count.

        Per-file isolation: one bad file can't kill the whole index. Files that
        fail are left unfingerprinted so they are retried on the next run.
        """
        files = _expand_paths(paths)
        total = 0
        succeeded: list[Path] = []
        for file in files:
            try:
                if self._max_file_bytes and file.stat().st_size > self._max_file_bytes:
                    logger.warning(f"Skipping RAG file over RAG_MAX_FILE_BYTES ({self._max_file_bytes}): {file}")
                    # Fingerprint it so the warning fires once, not every run. NOTE:
                    # raising the cap later needs RAG_REINDEX=always once to pick the file up.
                    succeeded.append(file)
                    continue
                documents = _load_documents([file])
                chunks = self._splitter.split_documents(documents)
                if chunks:
                    await self._store.aadd_documents(chunks)
                    total += len(chunks)
                # Fingerprint whenever load+split succeeded without exception
                # (even with 0 chunks — deterministic, don't retry forever).
                succeeded.append(file)
            except Exception as e:
                logger.warning(f"Skipping RAG file {file} after ingest error: {e}")
        if succeeded:
            fresh = fingerprint_paths(succeeded)
            if self._persist_manifest:
                try:
                    # Refresh in-memory manifest from disk first to avoid clobbering,
                    # then merge fingerprints for the files just ingested.
                    on_disk = load_manifest(self.store_path, self.bot_uuid)
                    on_disk.update(fresh)
                    save_manifest(self.store_path, self.bot_uuid, on_disk)
                    self._manifest = on_disk
                except OSError as e:
                    # Vectors stay indexed; worst case is rework later.
                    logger.warning(f"Could not persist RAG manifest: {e}")
            self._manifest.update(fresh)
        return total

    async def query(self, query: str, top_k: int | None = None) -> list[RAGChunk]:
        """Similarity-search the store and map hits to RAGChunks."""
        k = top_k if top_k is not None else self.top_k
        try:
            scored = await self._store.asimilarity_search_with_score(query, k=k)
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
            return []
        results: list[RAGChunk] = []
        for doc, score in scored:
            source = str(doc.metadata.get("source", ""))
            results.append(RAGChunk(text=doc.page_content, source=source, score=float(score)))
        return results

    async def clear(self) -> None:
        """Reset the vector store and drop manifest state."""
        self._store = InMemoryVectorStore(embedding=self._embeddings)
        self._manifest = {}
        if not self._persist_manifest:
            return
        path = manifest_path(self.store_path, self.bot_uuid)
        try:
            if path.is_file():
                path.unlink()
        except OSError as e:
            logger.warning(f"Could not remove manifest {path}: {e}")

    def needs_reindex(self, paths: list[str]) -> list[str]:
        """Return expanded files that are new or changed vs. the manifest."""
        files = _expand_paths(paths)
        if self._persist_manifest:
            manifest = self._manifest or load_manifest(self.store_path, self.bot_uuid)
        else:
            manifest = self._manifest
        return find_stale_paths([str(f) for f in files], manifest)

    def as_tool(self, name: str, description: str) -> BaseTool:
        """Expose the backend as a LangChain retriever tool."""
        retriever = self._store.as_retriever(search_kwargs={"k": self.top_k})
        return create_retriever_tool(retriever, name, description)
