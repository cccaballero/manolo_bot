"""Phase 1 RAG package."""

from manolo_bot.rag.base import RAG_TOOL_NAME, BaseRAGBackend, RAGChunk
from manolo_bot.rag.factory import build_rag_backend
from manolo_bot.rag.filesystem_backend import FilesystemRAGBackend
from manolo_bot.rag.prompting import build_rag_instructions, build_rag_tool_description
from manolo_bot.rag.sources import RAGSource

__all__ = [
    "RAG_TOOL_NAME",
    "BaseRAGBackend",
    "RAGChunk",
    "RAGSource",
    "FilesystemRAGBackend",
    "build_rag_backend",
    "build_rag_instructions",
    "build_rag_tool_description",
]
