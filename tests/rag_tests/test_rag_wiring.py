"""Phase 2 tests: RAG config parsing and wiring (no network/API keys)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings.fake import FakeEmbeddings

from manolo_bot.ai.config import BotConfig, LLMConfig
from manolo_bot.config import Config
from manolo_bot.rag.embeddings import (
    GOOGLE_DEFAULT_EMBEDDING_MODEL,
    OLLAMA_DEFAULT_EMBEDDING_MODEL,
    build_embeddings,
)
from manolo_bot.rag.factory import build_rag_backend
from manolo_bot.rag.prompting import build_rag_instructions, build_rag_tool_description

REPO_ROOT = Path(__file__).resolve().parents[2]

RAG_ENV_VARS = [
    "RAG_ENABLED",
    "RAG_BACKEND",
    "RAG_SOURCES",
    "RAG_STORE_PATH",
    "RAG_TOP_K",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_REINDEX",
    "RAG_EMBEDDING_MODEL",
    "RAG_MAX_FILE_BYTES",
]


def _set_telegram_env() -> None:
    os.environ["TELEGRAM_BOT_NAME"] = "Manolo"
    os.environ["TELEGRAM_BOT_USERNAME"] = "ManoloBot"
    os.environ["TELEGRAM_BOT_TOKEN"] = "1234567890"


class TestRAGConfig(unittest.TestCase):
    def setUp(self) -> None:
        _set_telegram_env()
        for var in RAG_ENV_VARS:
            os.environ.pop(var, None)
        self.addCleanup(self._cleanup_rag_env)

    def _cleanup_rag_env(self) -> None:
        for var in RAG_ENV_VARS:
            os.environ.pop(var, None)

    def test_rag_defaults(self) -> None:
        config = Config(lazy=True)
        self.assertFalse(config.rag_enabled)
        self.assertEqual(config.rag_backend, "in_memory")
        self.assertEqual(config.rag_sources, [])
        self.assertEqual(config.rag_store_path, os.path.join(tempfile.gettempdir(), "manolo_bot", "rag"))
        self.assertEqual(config.rag_top_k, 5)
        self.assertEqual(config.rag_chunk_size, 1000)
        self.assertEqual(config.rag_chunk_overlap, 200)
        self.assertEqual(config.rag_reindex, "auto")
        self.assertEqual(config.rag_embedding_model, "")
        self.assertEqual(config.rag_max_file_bytes, 10 * 1024 * 1024)

    def test_rag_env_parsing(self) -> None:
        os.environ["RAG_ENABLED"] = "True"
        os.environ["RAG_BACKEND"] = "in_memory"
        os.environ["RAG_SOURCES"] = "docs/a.md,docs/b/"
        os.environ["RAG_STORE_PATH"] = "/tmp/rag-test"
        os.environ["RAG_TOP_K"] = "3"
        os.environ["RAG_CHUNK_SIZE"] = "500"
        os.environ["RAG_CHUNK_OVERLAP"] = "50"
        os.environ["RAG_REINDEX"] = "always"
        os.environ["RAG_EMBEDDING_MODEL"] = "text-embedding-3-small"
        os.environ["RAG_MAX_FILE_BYTES"] = "1024"
        config = Config(lazy=True)
        self.assertTrue(config.rag_enabled)
        self.assertEqual(config.rag_backend, "in_memory")
        self.assertEqual(config.rag_sources, ["docs/a.md", "docs/b/"])
        self.assertEqual(config.rag_store_path, "/tmp/rag-test")
        self.assertEqual(config.rag_top_k, 3)
        self.assertEqual(config.rag_chunk_size, 500)
        self.assertEqual(config.rag_chunk_overlap, 50)
        self.assertEqual(config.rag_reindex, "always")
        self.assertEqual(config.rag_embedding_model, "text-embedding-3-small")
        self.assertEqual(config.rag_max_file_bytes, 1024)

    def test_bot_config_rag_defaults(self) -> None:
        bot_config = BotConfig(
            bot_uuid="test-bot",
            bot_name="TestBot",
            bot_username="test_bot",
            bot_token="123456:ABC",
            user_id=0,
        )
        self.assertFalse(bot_config.rag_enabled)
        self.assertEqual(bot_config.rag_backend, "in_memory")
        self.assertEqual(bot_config.rag_sources, [])
        self.assertEqual(bot_config.rag_store_path, os.path.join(tempfile.gettempdir(), "manolo_bot", "rag"))
        self.assertEqual(bot_config.rag_top_k, 5)
        self.assertEqual(bot_config.rag_chunk_size, 1000)
        self.assertEqual(bot_config.rag_chunk_overlap, 200)
        self.assertEqual(bot_config.rag_reindex, "auto")
        self.assertEqual(bot_config.rag_embedding_model, "")
        self.assertEqual(bot_config.rag_max_file_bytes, 10 * 1024 * 1024)


class TestRAGWiring(unittest.TestCase):
    def test_factory_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            build_rag_backend("chroma", FakeEmbeddings(size=8))

    def test_llmbot_has_no_rag_wiring(self) -> None:
        source = (REPO_ROOT / "manolo_bot" / "ai" / "llmbot.py").read_text(encoding="utf-8")
        self.assertNotIn("manolo_bot.rag", source)
        self.assertNotIn("rag_search", source)
        self.assertNotIn("build_rag_backend", source)

    def test_main_wires_rag_for_agents_only(self) -> None:
        source = (REPO_ROOT / "manolo_bot" / "main.py").read_text(encoding="utf-8")
        self.assertIn("manolo_bot.rag", source)
        self.assertIn("async def _get_rag_backend", source)
        # Agent branches pass the shared backend as a first-class param...
        self.assertEqual(source.count("rag_backend=rag_backend"), 2)
        # ...and no longer ride the `tools=` custom-tools channel.
        self.assertNotIn("rag_agent_tools", source)
        self.assertNotIn("_get_agent_tools_with_rag", source)
        # LLMBot construction takes no RAG references (branch untouched).
        llmbot_call = source.split("llm_bot = LLMBot(")[1].split("return llm_bot")[0]
        self.assertNotIn("rag_backend", llmbot_call)
        self.assertNotIn("rag_search", llmbot_call)


def _llm_config(**overrides) -> LLMConfig:
    kwargs = {
        "google_api_key": "",
        "google_api_model": "",
        "openai_api_key": "",
        "openai_api_model": "",
        "openai_api_base_url": "",
        "ollama_model": "",
    }
    kwargs.update(overrides)
    return LLMConfig(**kwargs)  # type: ignore[arg-type]


class TestEmbeddingsFactory(unittest.TestCase):
    def test_google_default_model_is_supported(self) -> None:
        # Regression: models/text-embedding-004 was retired by Google (Jan 2026).
        self.assertEqual(GOOGLE_DEFAULT_EMBEDDING_MODEL, "models/gemini-embedding-001")
        embeddings = build_embeddings(_llm_config(google_api_key="key"))
        self.assertEqual(embeddings.model, "models/gemini-embedding-001")

    def test_google_model_override(self) -> None:
        embeddings = build_embeddings(_llm_config(google_api_key="key"), model="custom-model")
        self.assertEqual(embeddings.model, "custom-model")

    def test_ollama_defaults_to_embedding_model(self) -> None:
        embeddings = build_embeddings(_llm_config(ollama_model="gemma3:12b"))
        self.assertEqual(embeddings.model, OLLAMA_DEFAULT_EMBEDDING_MODEL)

    def test_no_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_embeddings(_llm_config())

    def test_openai_model_override_for_compatible_endpoint(self) -> None:
        # LLM on an OpenAI-compatible endpoint without OpenAI's own models.
        cfg = _llm_config(openai_api_base_url="http://localhost:1234/v1")
        embeddings = build_embeddings(cfg, model="text-embedding-3-small")
        self.assertEqual(embeddings.model, "text-embedding-3-small")
        self.assertEqual(str(embeddings.openai_api_base).rstrip("/"), "http://localhost:1234/v1")


class TestRAGPrompting(unittest.TestCase):
    def test_tool_description_names_sources_and_routes(self) -> None:
        description = build_rag_tool_description(["/docs/handbook.md", "/docs/policies/"])
        self.assertIn("handbook.md", description)
        self.assertIn("policies", description)
        self.assertIn("ALWAYS", description)

    def test_instructions_mention_tool_and_sources(self) -> None:
        instructions = build_rag_instructions(["/docs/handbook.md"])
        self.assertIn("rag_search", instructions)
        self.assertIn("handbook.md", instructions)

    def test_empty_sources_still_valid(self) -> None:
        self.assertIn("ALWAYS", build_rag_tool_description([]))
        self.assertIn("rag_search", build_rag_instructions([]))


if __name__ == "__main__":
    unittest.main()
