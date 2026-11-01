import os
import unittest
from unittest.mock import patch

from langchain_google_genai import ChatGoogleGenerativeAI

from manolo_bot.ai.config import LLMConfig
from manolo_bot.ai.llmbot import LLMBuilder


def make_llm_config(**overrides) -> LLMConfig:
    """Build an LLMConfig with every provider disabled except the given overrides."""
    fields = {
        "google_api_key": "",
        "google_api_model": "gemini-2.5-flash",
        "openai_api_key": "",
        "openai_api_model": "",
        "openai_api_base_url": "",
        "ollama_model": "",
    }
    fields.update(overrides)
    return LLMConfig(**fields)


def env_without_google_keys() -> dict:
    """Copy of the process environment with the ambient Google credentials removed."""
    return {k: v for k, v in os.environ.items() if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}


class TestLLMBuilder(unittest.TestCase):
    def test_google_api_key_from_config_is_forwarded_to_client(self):
        """The programmatic key must reach the client without env fallback (issue #108)."""
        with patch.dict(os.environ, env_without_google_keys(), clear=True):
            llm = LLMBuilder(make_llm_config(google_api_key="test-key")).get_llm()

        self.assertIsInstance(llm, ChatGoogleGenerativeAI)
        self.assertEqual(llm.google_api_key.get_secret_value(), "test-key")

    def test_configured_google_api_key_wins_over_environment(self):
        """An ambient GOOGLE_API_KEY must not shadow the key supplied in LLMConfig."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}, clear=False):
            llm = LLMBuilder(make_llm_config(google_api_key="config-key")).get_llm()

        self.assertEqual(llm.google_api_key.get_secret_value(), "config-key")


if __name__ == "__main__":
    unittest.main()
