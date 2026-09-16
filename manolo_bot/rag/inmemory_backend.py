"""In-memory RAG backend (Phase 1)."""

from __future__ import annotations

import glob
import logging
from collections.abc import Sequence
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
    fingerprint_file,
    load_manifest,
    manifest_path,
    save_manifest,
)
from manolo_bot.rag.prompting import build_rag_source_tool_description
from manolo_bot.rag.sources import RAGSource, describe_rag_tools

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


def _entry_patterns(entries: Sequence[RAGSource]) -> list[str]:
    """Return plain path patterns from records (blank patterns dropped)."""
    return [source.path for source in entries if source.path.strip()]


def _source_filter(allowed: frozenset[str]):
    """Native pre-ranking predicate keeping only chunks from allowed sources."""

    def _keep(doc: Document) -> bool:
        return str((doc.metadata or {}).get("source", "")) in allowed

    return _keep


def _normalize_source(source: str) -> str:
    """Resolve a stored chunk source for comparison with manifest keys."""
    try:
        return str(Path(source).resolve())
    except OSError:
        return source


async def _drop_source_vectors(store: InMemoryVectorStore, allowed_resolved: set[str]) -> int:
    """Delete entries whose normalized stored source is in the set; return count."""
    stale_ids = []
    for doc_id, entry in store.store.items():
        if isinstance(entry, dict):
            metadata = entry.get("metadata", {})
        else:
            metadata = getattr(entry, "metadata", None) or {}
        source = metadata.get("source", "") if isinstance(metadata, dict) else ""
        if _normalize_source(str(source)) in allowed_resolved:
            stale_ids.append(doc_id)
    if stale_ids:
        await store.adelete(stale_ids)
    return len(stale_ids)


def _fingerprinted(file: Path, chunks: int) -> dict[str, dict]:
    """Fingerprint one file, recording how many chunks it produced.

    ``chunks`` is the indexed chunk count, or -1 for intentionally skipped
    files (size cap): -1 reads as fresh so the skip warning fires once, while
    0 (or a missing key on pre-existing entries) always reads as stale so
    empty results are retried instead of silently kept.
    """
    try:
        entry = fingerprint_file(file)
    except OSError:
        return {}
    entry["chunks"] = chunks
    return {entry["path"]: entry}


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

    def _read_manifest(self) -> dict:
        """In-memory manifest, else disk load when the flag persists it."""
        if self._persist_manifest:
            return self._manifest or load_manifest(self.store_path, self.bot_uuid)
        return self._manifest

    async def _write_manifest(self, manifest: dict) -> None:
        """Set the in-memory manifest; save to disk when the flag persists it."""
        self._manifest = manifest
        if self._persist_manifest:
            try:
                save_manifest(self.store_path, self.bot_uuid, manifest)
            except OSError as e:
                # Vectors stay indexed; worst case is rework later.
                logger.warning(f"Could not persist RAG manifest: {e}")

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

    async def ingest(self, paths: Sequence[RAGSource]) -> int:
        """Load, split and index only new or changed files; return new chunk count.

        Idempotent: files whose fingerprint matches the manifest are skipped
        without any embedding call; changed files have their old vectors dropped
        first, so re-ingesting never duplicates and never pays twice.
        Per-file isolation: one bad file can't kill the whole index. Files that
        fail are left unfingerprinted so they are retried on the next run.
        """
        files = _expand_paths(_entry_patterns(paths))
        manifest = self._read_manifest()
        stale = set(find_stale_paths([str(f) for f in files], manifest))
        fresh = [f for f in files if str(f) not in stale]
        if fresh:
            logger.debug(f"Skipping {len(fresh)} fresh RAG file(s), no embedding calls needed")
        previously_indexed = {str(f.resolve()) for f in files if str(f) in stale} & set(manifest.keys())
        if previously_indexed:
            dropped = await _drop_source_vectors(self._store, previously_indexed)
            logger.debug(f"Dropped {dropped} stale vector(s) before re-ingest")
        total = 0
        succeeded: dict[str, dict] = {}
        for file in files:
            if str(file) not in stale:
                continue
            try:
                if self._max_file_bytes and file.stat().st_size > self._max_file_bytes:
                    logger.warning(f"Skipping RAG file over RAG_MAX_FILE_BYTES ({self._max_file_bytes}): {file}")
                    # Fingerprint with a -1 sentinel so the warning fires once, not
                    # every run. NOTE: raising the cap later needs RAG_REINDEX=always
                    # once to pick the file up.
                    succeeded.update(_fingerprinted(file, -1))
                    continue
                documents = _load_documents([file])
                chunks = self._splitter.split_documents(documents)
                if not chunks:
                    # Zero chunks indexed: warn loudly and do NOT fingerprint, so
                    # the file is retried (a silent 0-chunk fingerprint once hid a
                    # missing loader dependency forever).
                    logger.warning(f"RAG file yielded 0 chunks, will retry next run: {file}")
                    continue
                await self._store.aadd_documents(chunks)
                total += len(chunks)
                succeeded.update(_fingerprinted(file, len(chunks)))
            except Exception as e:
                logger.warning(f"Skipping RAG file {file} after ingest error: {e}")
        if succeeded:
            # Refresh from the persisted view first to avoid clobbering,
            # then merge fingerprints for the files just ingested.
            merged = self._read_manifest()
            merged.update(succeeded)
            await self._write_manifest(merged)
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

    async def remove(self, paths: Sequence[RAGSource]) -> int:
        """Drop indexed documents without touching anything else.

        Matches expanded files AND manifest keys directly, so entries pointing
        at files deleted from disk are still forgotten. Unknown or
        never-indexed entries warn and are skipped. Manifest fingerprints are
        pruned (and persisted when applicable) so removed files are treated as
        new if added back later.
        """
        files = _expand_paths(_entry_patterns(paths))
        manifest = self._read_manifest()
        expanded = {str(f.resolve()) for f in files}
        direct = {_normalize_source(record.path) for record in paths if record.path.strip()}
        targets = (expanded | direct) & set(manifest.keys())
        if not targets:
            logger.warning(f"RAG remove matched nothing indexed: {[r.path for r in paths]}")
            return 0
        dropped = await _drop_source_vectors(self._store, targets)
        pruned = {key: value for key, value in manifest.items() if key not in targets}
        await self._write_manifest(pruned)
        return dropped

    def list_sources(self) -> list[str]:
        """Return sorted indexed file paths (manifest keys)."""
        return sorted(self._read_manifest().keys())

    def needs_reindex(self, paths: Sequence[RAGSource]) -> list[RAGSource]:
        """Return the subset of records with new or changed files."""
        records = [record for record in paths if record.path.strip()]
        expanded = [(record, _expand_paths([record.path])) for record in records]
        manifest = self._read_manifest()
        stale = set(find_stale_paths([str(f) for _, files in expanded for f in files], manifest))
        return [record for record, files in expanded if any(str(f) in stale for f in files)]

    def find_removed(self, paths: Sequence[RAGSource]) -> list[str]:
        """Return indexed files matching none of the given records."""
        configured: set[str] = set()
        for record in paths:
            if not record.path.strip():
                continue
            configured.update(str(f.resolve()) for f in _expand_paths([record.path]))
        manifest = self._read_manifest()
        return sorted(set(manifest.keys()) - configured)

    def as_tool(self, name: str, description: str) -> BaseTool:
        """Expose the backend as a LangChain retriever tool."""
        retriever = self._store.as_retriever(search_kwargs={"k": self.top_k})
        return create_retriever_tool(retriever, name, description)

    def as_source_tools(self, entries: Sequence[RAGSource]) -> list[BaseTool]:
        """Expose one retriever tool per structured source entry.

        Each tool searches only chunks whose stored source came from that entry,
        via the store's native pre-ranking filter predicate. Entries matching no
        files are skipped (the unmatched warning already fired on expand).
        """
        tools: list[BaseTool] = []
        described = describe_rag_tools(entries)
        # Same blank-filter as describe_rag_tools, so names stay aligned.
        patterns = [source.path for source in entries if source.path.strip()]
        for (name, label, description), pattern in zip(described, patterns):
            # Identical expression to the stored chunk metadata in _load_documents.
            allowed = frozenset(str(f) for f in _expand_paths([pattern]))
            if not allowed:
                continue
            retriever = self._store.as_retriever(search_kwargs={"k": self.top_k, "filter": _source_filter(allowed)})
            tools.append(create_retriever_tool(retriever, name, build_rag_source_tool_description(label, description)))
        return tools
