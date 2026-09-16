"""Per-source RAG tools: parsing, naming, scoping (no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings.fake import FakeEmbeddings

from manolo_bot.rag.inmemory_backend import InMemoryRAGBackend
from manolo_bot.rag.prompting import build_rag_instructions
from manolo_bot.rag.sources import (
    MAX_SOURCE_TOOLS,
    RAGSource,
    describe_rag_tools,
    parse_sources,
    slugify,
    unique_slugs,
)


class TestParseSources(unittest.TestCase):
    def test_bare_entry(self) -> None:
        self.assertEqual(parse_sources(["docs/a.md"]), [RAGSource(path="docs/a.md")])

    def test_desc_entry(self) -> None:
        self.assertEqual(parse_sources(["docs/a.md::DESC=Handbook"]), [RAGSource("docs/a.md", "Handbook")])

    def test_empty_desc_is_bare(self) -> None:
        self.assertEqual(parse_sources(["docs/a.md::DESC="]), [RAGSource("docs/a.md", "")])

    def test_split_on_last_separator(self) -> None:
        self.assertEqual(parse_sources(["x::DESC=a::DESC=b"]), [RAGSource(path="x::DESC=a", description="b")])

    def test_slug_auto_derived_when_empty(self) -> None:
        self.assertEqual(RAGSource(path="docs/handbook.md").slug, "handbook")
        self.assertEqual(RAGSource(path="docs/*.md").slug, "md")

    def test_explicit_slug_preserved(self) -> None:
        record = RAGSource(path="docs/handbook.md", description="Handbook", slug="hb")
        self.assertEqual(record.slug, "hb")
        self.assertEqual(
            describe_rag_tools([record]),
            [("rag_search_hb", "handbook", "Handbook")],
        )

    def test_records_pass_through_rejected(self) -> None:
        # Boundary enforcement: records never flow through the env parser.
        record = RAGSource(path="docs/a.md", description="Handbook, with a comma")
        with self.assertRaises(TypeError) as ctx:
            parse_sources([record])  # type: ignore[list-item]
        self.assertIn("RAGSource", str(ctx.exception))

    def test_non_string_types_raise(self) -> None:
        for bad in (None, 123, Path("docs/a.md")):
            with self.assertRaises(TypeError, msg=repr(bad)):
                parse_sources([bad])  # type: ignore[list-item]


class TestSlugs(unittest.TestCase):
    def test_slugify_cases(self) -> None:
        self.assertEqual(slugify("User Handbook!"), "user_handbook")
        self.assertEqual(slugify("docs/*.md"), "docs_md")
        self.assertEqual(slugify(""), "source")
        self.assertEqual(slugify("a" * 40), "a" * 32)

    def test_unique_slugs_dedupe(self) -> None:
        self.assertEqual(unique_slugs(["a", "a", "a"]), ["a", "a_2", "a_3"])
        self.assertEqual(unique_slugs(["a", "a_2", "a"]), ["a", "a_2", "a_3"])


class TestDescribeRagTools(unittest.TestCase):
    def test_desc_is_description_not_label(self) -> None:
        # DESC feeds the routing description however long it is; the label and
        # slug always derive from the filename (stable).
        self.assertEqual(
            describe_rag_tools([RAGSource("/docs/handbook.md", "User Handbook for New Employees Everywhere")]),
            [
                (
                    "rag_search_handbook",
                    "handbook",
                    "User Handbook for New Employees Everywhere",
                )
            ],
        )

    def test_stem_slug(self) -> None:
        self.assertEqual(
            describe_rag_tools([RAGSource("/docs/handbook.md")]), [("rag_search_handbook", "handbook", "")]
        )

    def test_glob_and_dir_use_last_part(self) -> None:
        described = describe_rag_tools([RAGSource("/docs/*.md"), RAGSource("/docs/policies/")])
        self.assertEqual([name for name, _, _ in described], ["rag_search_md", "rag_search_policies"])

    def test_cap_returns_empty(self) -> None:
        entries = [RAGSource(f"/docs/f{i}.md") for i in range(MAX_SOURCE_TOOLS + 1)]
        self.assertEqual(describe_rag_tools(entries), [])
        self.assertEqual(len(describe_rag_tools(entries[:-1])), MAX_SOURCE_TOOLS)

    def test_blank_patterns_skipped(self) -> None:
        self.assertEqual(describe_rag_tools([RAGSource(""), RAGSource("  ")]), [])


class TestSourceToolsScoping(unittest.IsolatedAsyncioTestCase):
    async def _make_backend(self, tmp: str) -> tuple:
        alpha = Path(tmp) / "alpha.md"
        alpha.write_text("Alpha content. Unique token: token-alpha.\n", encoding="utf-8")
        beta = Path(tmp) / "beta.md"
        beta.write_text("Beta content. Unique token: token-beta.\n", encoding="utf-8")
        backend = InMemoryRAGBackend(
            embeddings=FakeEmbeddings(size=32),
            bot_uuid="test-bot",
            store_path=str(Path(tmp) / "store"),
            chunk_size=200,
            chunk_overlap=20,
            top_k=10,
        )
        await backend.build_or_load()
        entries = [RAGSource(str(alpha), "Alpha"), RAGSource(str(beta), "Beta")]
        count = await backend.ingest(entries)
        self.assertGreater(count, 0)
        return backend, entries

    async def test_scoped_tool_returns_only_own_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend, entries = await self._make_backend(tmp)
            tools = {tool.name: tool for tool in backend.as_source_tools(entries)}
            self.assertEqual(set(tools), {"rag_search_alpha", "rag_search_beta"})

            out_alpha = await tools["rag_search_alpha"].ainvoke({"query": "content"})
            self.assertIn("token-alpha", out_alpha)
            self.assertNotIn("token-beta", out_alpha)
            self.assertIn("not rag_search", tools["rag_search_alpha"].description)

    async def test_global_tool_returns_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend, entries = await self._make_backend(tmp)
            out = await backend.as_tool("rag_search", "global").ainvoke({"query": "content"})
            self.assertIn("token-alpha", out)
            self.assertIn("token-beta", out)

    async def test_fileless_entry_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = InMemoryRAGBackend(
                embeddings=FakeEmbeddings(size=32),
                bot_uuid="test-bot",
                store_path=str(Path(tmp) / "store"),
            )
            await backend.build_or_load()
            self.assertEqual(backend.as_source_tools([RAGSource("/nonexistent_xyz/*.qqq")]), [])

    async def test_over_cap_returns_no_scoped_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = InMemoryRAGBackend(
                embeddings=FakeEmbeddings(size=32),
                bot_uuid="test-bot",
                store_path=str(Path(tmp) / "store"),
            )
            await backend.build_or_load()
            entries = [RAGSource(f"/docs/f{i}.md", f"F{i}") for i in range(MAX_SOURCE_TOOLS + 1)]
            self.assertEqual(backend.as_source_tools(entries), [])


class TestSourceInstructions(unittest.TestCase):
    def test_instructions_mention_scoped_names(self) -> None:
        instructions = build_rag_instructions(
            [RAGSource("/docs/alpha.md", "Alpha"), RAGSource("/docs/beta.md", "Beta")]
        )
        self.assertIn("rag_search_alpha", instructions)
        self.assertIn("rag_search_beta", instructions)


class TestRAGSourceRecords(unittest.IsolatedAsyncioTestCase):
    async def test_records_end_to_end_without_desc_strings(self) -> None:
        # Programmatic callers pass RAGSource records — no ::DESC= strings needed,
        # so descriptions may freely contain commas.
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "guide.md"
            doc.write_text("Manolo guide. Unique token: token-guide.\n", encoding="utf-8")
            backend = InMemoryRAGBackend(
                embeddings=FakeEmbeddings(size=32),
                bot_uuid="test-bot",
                store_path=str(Path(tmp) / "store"),
            )
            await backend.build_or_load()
            entries = [RAGSource(path=str(doc), description="Manolo user guide, including setup")]
            self.assertGreater(await backend.ingest(entries), 0)

            tools = backend.as_source_tools(entries)
            self.assertEqual([tool.name for tool in tools], ["rag_search_guide"])
            out = await tools[0].ainvoke({"query": "guide"})
            self.assertIn("token-guide", out)

            instructions = build_rag_instructions(entries)
            self.assertIn("rag_search_guide", instructions)


if __name__ == "__main__":
    unittest.main()
