"""RAG prompting helpers: tool descriptions and system-instruction snippets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from manolo_bot.rag.sources import RAGSource, describe_rag_tools


def source_labels(sources: Sequence[RAGSource]) -> list[str]:
    """Return short human-readable labels (basenames) for RAG sources.

    Descriptions are routing context for tool descriptions only and never
    surface here — system-prompt listings stay short.
    """
    labels: list[str] = []
    for source in sources:
        name = Path(source.path).name or source.path
        if name not in labels:
            labels.append(name)
    return labels


def build_rag_tool_description(sources: Sequence[RAGSource]) -> str:
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


def build_rag_instructions(sources: Sequence[RAGSource]) -> str:
    """Build a system-instruction snippet telling the agent when to use RAG."""
    labels = source_labels(sources)
    scope = f" (currently: {', '.join(labels)})" if labels else ""
    names = [name for name, _, _ in describe_rag_tools(list(sources))]
    if names:
        tools = ", ".join(f"`{name}`" for name in names)
        return (
            "\n\n## RAG\n\n"
            f"You have a local document knowledge base{scope} with scoped search tools:"
            f" {tools}. When the user asks about topics covered in those documents, call"
            " the matching tool first and base your answer on what it returns, citing sources."
        )
    return (
        "\n\n## RAG\n\n"
        "You have a `rag_search` tool connected to a local document knowledge base"
        f"{scope}. When the user asks about topics covered in those documents, call"
        " `rag_search` first and base your answer on what it returns, citing sources."
    )


def build_rag_source_tool_description(label: str, description: str) -> str:
    """Build a routing-strong description for a per-source RAG tool."""
    detail = description.strip() or label
    return f"Search {label}. {detail}. Use this tool (not rag_search) for questions about {detail}."
