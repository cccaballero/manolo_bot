"""Structured RAG sources: single source of truth for per-source tools.

Entries come from RAG_SOURCES (comma-separated), optionally suffixed with
``::DESC=<description>`` to describe a per-source ``rag_search_<slug>`` tool.
Tool slugs always derive from the filename (stable); descriptions carry the
routing semantics the agent actually reads.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from manolo_bot.rag.base import RAG_TOOL_NAME

logger = logging.getLogger(__name__)

#: Separator between a source pattern and its description (split on LAST occurrence).
DESC_SEP = "::DESC="

#: Max per-source tools before falling back to the global tool only.
MAX_SOURCE_TOOLS = 8

_GLOB_CHARS = frozenset("*?[")


@dataclass
class RAGSource:
    """A source path with routing description and tool slug.

    ``path`` is a file, directory, or glob pattern. ``description`` is free
    routing context for tool descriptions (never a name). ``slug`` names the
    per-source tool; when empty it is derived from the path in
    ``__post_init__``, so it is always populated afterwards. Library users
    construct records directly with whatever slug suits what they build.
    """

    path: str
    description: str = ""
    slug: str = ""

    def __post_init__(self) -> None:
        if not self.slug.strip():
            self.slug = default_slug(self.path)


def parse_sources(entries: Sequence[str]) -> list[RAGSource]:
    """Env-boundary parser for RAG_SOURCES strings.

    Programmatic callers construct ``RAGSource`` records directly and never
    call this — ``::DESC=`` strings exist only at the env edge.

    Entries without the separator (or with an empty description) become bare
    sources. Limitation: descriptions must not contain commas — RAG_SOURCES is
    comma-separated (a JSON field later will lift this).
    """
    sources = []
    for entry in entries:
        if not isinstance(entry, str):
            raise TypeError(
                f"parse_sources() takes RAG_SOURCES strings, got {type(entry).__name__}; "
                "pass RAGSource records directly to ingest()/as_source_tools(), not through parse_sources"
            )
        pattern, sep, description = entry.rpartition(DESC_SEP)
        if not sep:
            sources.append(RAGSource(path=entry))
        else:
            sources.append(RAGSource(path=pattern, description=description))
    return sources


def slugify(text: str, max_len: int = 32) -> str:
    """Lowercase slug: non-alnum runs become ``_``, trimmed and truncated."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:max_len].strip("_")
    return slug or "source"


def unique_slugs(slugs: list[str]) -> list[str]:
    """Dedupe slugs, appending ``_2``, ``_3``, ... to repeats."""
    used: set[str] = set()
    counters: dict[str, int] = {}
    unique = []
    for slug in slugs:
        candidate = slug
        if candidate in used:
            n = counters.get(slug, 1)
            while f"{slug}_{n + 1}" in used:
                n += 1
            candidate = f"{slug}_{n + 1}"
            counters[slug] = n + 1
        used.add(candidate)
        unique.append(candidate)
    return unique


def default_slug(path: str) -> str:
    """Derive a tool slug from a path (file stem, raw last part for dirs/globs)."""
    if any(char in path for char in _GLOB_CHARS):
        return slugify(Path(path).name)
    return slugify(Path(path).stem or Path(path).name)


def _display_name(source: RAGSource) -> str:
    """Short human label: always the file stem (raw last part for dirs/globs).

    The description is routing context for tool descriptions, never a label —
    it can be an arbitrarily long sentence and must not leak into names or
    system-prompt listings.
    """
    if any(char in source.path for char in _GLOB_CHARS):
        return Path(source.path).name
    return Path(source.path).stem or Path(source.path).name


def describe_rag_tools(sources: Sequence[RAGSource]) -> list[tuple[str, str, str]]:
    """Describe per-source tools as (tool_name, label, description) triples.

    Pure function shared by backend tool-building and prompt snippets so names
    never drift. Blank paths are skipped; over MAX_SOURCE_TOOLS returns []
    so callers fall back to the global tool. Stored slugs are deduped, so two
    files sharing a stem still get distinct tools.
    """
    parsed = [source for source in sources if source.path.strip()]
    if len(parsed) > MAX_SOURCE_TOOLS:
        logger.info(
            "Too many RAG sources (%d > %d); using the global %s tool only",
            len(parsed),
            MAX_SOURCE_TOOLS,
            RAG_TOOL_NAME,
        )
        return []
    slugs = unique_slugs([source.slug for source in parsed])
    return [
        (f"{RAG_TOOL_NAME}_{slug}", _display_name(source), source.description.strip())
        for source, slug in zip(parsed, slugs)
    ]
