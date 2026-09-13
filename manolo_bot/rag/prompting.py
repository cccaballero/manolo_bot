"""RAG prompting helpers: tool descriptions and system-instruction snippets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def source_labels(sources: Sequence[str]) -> list[str]:
    """Return short human-readable labels (basenames) for RAG sources."""
    labels: list[str] = []
    for source in sources:
        name = Path(str(source)).name or str(source)
        if name not in labels:
            labels.append(name)
    return labels


def build_rag_tool_description(sources: Sequence[str]) -> str:
    """Build a routing-strong description for the ``rag_search`` tool."""
    labels = source_labels(sources)
    scope = f" It indexes: {', '.join(labels)}." if labels else ""
    return (
        "Search the bot's local document knowledge base and return relevant passages."
        f"{scope} ALWAYS call this tool before answering questions about the indexed"
        " documents, and ground your answer in the returned passages, citing the"
        " source of each fact. If the tool returns nothing relevant, say so instead"
        " of guessing from the documents."
    )


def build_rag_instructions(sources: Sequence[str]) -> str:
    """Build a system-instruction snippet telling the agent when to use RAG."""
    labels = source_labels(sources)
    scope = f" (currently: {', '.join(labels)})" if labels else ""
    return (
        "\n\n## RAG\n\n"
        "You have a `rag_search` tool connected to a local document knowledge base"
        f"{scope}. When the user asks about topics covered in those documents, call"
        " `rag_search` first and base your answer on what it returns, citing sources."
    )
