"""Backend factory (Phase 1 seam for Chroma/FAISS later)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings

from manolo_bot.rag.base import BaseRAGBackend
from manolo_bot.rag.filesystem_backend import FilesystemRAGBackend
from manolo_bot.rag.inmemory_backend import InMemoryRAGBackend


def build_rag_backend(
    backend_name: str,
    embeddings: Embeddings,
    bot_uuid: str = "default",
    store_path: str | Path | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 4,
    max_file_bytes: int | None = None,
) -> BaseRAGBackend:
    """Build a RAG backend by name.

    :param backend_name: ``"in_memory"`` (ephemeral, debugging/tests) or
        ``"local_fs"`` (single-node JSON snapshot persistence).
    :param embeddings: Embeddings implementation (injectable for tests).
    :param bot_uuid: Bot namespace used for manifest isolation.
    :param store_path: Root store dir; defaults to a tmp manolo_bot/rag dir.
    :param max_file_bytes: Per-file size cap (None = unlimited).
    :raises ValueError: For any unsupported backend name.
    """
    normalized = (backend_name or "").strip().lower()
    if normalized == "in_memory":
        return InMemoryRAGBackend(
            embeddings=embeddings,
            bot_uuid=bot_uuid,
            store_path=store_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            max_file_bytes=max_file_bytes,
        )
    if normalized == "local_fs":
        return FilesystemRAGBackend(
            embeddings=embeddings,
            bot_uuid=bot_uuid,
            store_path=store_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            max_file_bytes=max_file_bytes,
        )
    raise ValueError(f"Unsupported RAG backend: {backend_name!r}")
