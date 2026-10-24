import os
import unittest

from manolo_bot.config import Config


class TestConfig(unittest.TestCase):
    def test_config(self):
        os.environ["GOOGLE_API_KEY"] = "1234567890"
        os.environ["GOOGLE_API_MODEL"] = "gemini-2.0-flash"
        os.environ["OPENAI_API_KEY"] = "1234567890"
        os.environ["OPENAI_API_MODEL"] = "gpt-3.5-turbo"
        os.environ["OPENAI_API_BASE_URL"] = "https://api.openai.com/v1"
        os.environ["OLLAMA_MODEL"] = "llama-2-13b"
        os.environ["TELEGRAM_BOT_NAME"] = "Manolo"
        os.environ["TELEGRAM_BOT_USERNAME"] = "ManoloBot"
        os.environ["TELEGRAM_BOT_TOKEN"] = "1234567890"
        os.environ["CONTEXT_MAX_TOKENS"] = "5007"
        os.environ["ENABLE_CONTEXT_SUMMARIZATION"] = "True"
        os.environ["SUMMARY_MAX_TOKENS"] = "256"
        os.environ["SUMMARY_KEEP_MESSAGES"] = "8"
        os.environ["PREFERRED_LANGUAGE"] = "Spanish"
        os.environ["ADD_NO_ANSWER"] = "False"
        os.environ["RATE_LIMITER_REQUESTS_PER_SECOND"] = "0.25"
        os.environ["RATE_LIMITER_CHECK_EVERY_N_SECONDS"] = "0.1"
        os.environ["RATE_LIMITER_MAX_BUCKET_SIZE"] = "10"
        os.environ["IMAGE_MULTIMODAL"] = "True"
        os.environ["AUDIO_MULTIMODAL"] = "True"
        os.environ["ENABLE_GROUP_ASSISTANT"] = "False"
        os.environ["WEBUI_SD_API_URL"] = "http://localhost:7860"
        os.environ["WEBUI_SD_API_PARAMS"] = (
            '{"steps": 1, "cfg_scale": 1, "width": 512, "height": 512, "timestep_spacing": "trailing"}'
        )
        os.environ["TELEGRAM_ALLOWED_CHATS"] = "1234567890,9876543210,-456456456456"

        config = Config()
        self.assertEqual(config.google_api_key, "1234567890")
        self.assertEqual(config.google_api_model, "gemini-2.0-flash")
        self.assertEqual(config.openai_api_key, "1234567890")
        self.assertEqual(config.openai_api_model, "gpt-3.5-turbo")
        self.assertEqual(config.openai_api_base_url, "https://api.openai.com/v1")
        self.assertEqual(config.ollama_model, "llama-2-13b")
        self.assertEqual(config.bot_name, "Manolo")
        self.assertEqual(config.bot_username, "ManoloBot")
        self.assertEqual(config.bot_token, "1234567890")
        self.assertEqual(config.context_max_tokens, 5007)
        self.assertEqual(config.enable_context_summarization, True)
        self.assertEqual(config.summary_max_tokens, 256)
        self.assertEqual(config.summary_keep_messages, 8)
        self.assertEqual(config.preferred_language, "Spanish")
        self.assertEqual(config.add_no_answer, False)
        self.assertEqual(config.rate_limiter_requests_per_second, 0.25)
        self.assertEqual(config.rate_limiter_check_every_n_seconds, 0.1)
        self.assertEqual(config.rate_limiter_max_bucket_size, 10)
        self.assertEqual(config.is_image_multimodal, True)
        self.assertEqual(config.is_audio_multimodal, True)
        self.assertEqual(config.is_group_assistant, False)
        self.assertEqual(config.sdapi_url, "http://localhost:7860")
        self.assertEqual(
            config.sdapi_params,
            {"steps": 1, "cfg_scale": 1, "width": 512, "height": 512, "timestep_spacing": "trailing"},
        )
        self.assertEqual(config.allowed_chat_ids, ["1234567890", "9876543210", "-456456456456"])

        # Cleanup
        del os.environ["GOOGLE_API_KEY"]
        del os.environ["GOOGLE_API_MODEL"]
        del os.environ["OPENAI_API_KEY"]
        del os.environ["OPENAI_API_MODEL"]
        del os.environ["OPENAI_API_BASE_URL"]
        del os.environ["OLLAMA_MODEL"]
        del os.environ["TELEGRAM_BOT_NAME"]
        del os.environ["TELEGRAM_BOT_USERNAME"]
        del os.environ["TELEGRAM_BOT_TOKEN"]
        del os.environ["CONTEXT_MAX_TOKENS"]
        del os.environ["ENABLE_CONTEXT_SUMMARIZATION"]
        del os.environ["SUMMARY_MAX_TOKENS"]
        del os.environ["SUMMARY_KEEP_MESSAGES"]
        del os.environ["PREFERRED_LANGUAGE"]
        del os.environ["ADD_NO_ANSWER"]
        del os.environ["RATE_LIMITER_REQUESTS_PER_SECOND"]
        del os.environ["RATE_LIMITER_CHECK_EVERY_N_SECONDS"]
        del os.environ["RATE_LIMITER_MAX_BUCKET_SIZE"]
        del os.environ["IMAGE_MULTIMODAL"]
        del os.environ["AUDIO_MULTIMODAL"]
        del os.environ["ENABLE_GROUP_ASSISTANT"]
        del os.environ["WEBUI_SD_API_URL"]
        del os.environ["WEBUI_SD_API_PARAMS"]
        del os.environ["TELEGRAM_ALLOWED_CHATS"]

    def test_required_fields_raise_exception_when_missing(self):
        # Clear required environment variables
        for env_var in ["TELEGRAM_BOT_NAME", "TELEGRAM_BOT_USERNAME", "TELEGRAM_BOT_TOKEN"]:
            os.environ.pop(env_var, None)

        # Initialize config
        config = Config(lazy=True)

        # Test each required field raises exception when accessed
        with self.assertRaises(Exception) as context:
            config.bot_name
        self.assertIn("Environment variable TELEGRAM_BOT_NAME is required", str(context.exception))

        with self.assertRaises(Exception) as context:
            config.bot_username
        self.assertIn("Environment variable TELEGRAM_BOT_USERNAME is required", str(context.exception))

        with self.assertRaises(Exception) as context:
            config.bot_token
        self.assertIn("Environment variable TELEGRAM_BOT_TOKEN is required", str(context.exception))

        # Non-required fields should not raise exceptions
        try:
            self.assertEqual(config.google_api_model, "gemini-2.0-flash")
            self.assertEqual(config.preferred_language, "Spanish")
        except Exception as e:
            self.fail(f"Non-required fields raised exception: {e}")

    def set_telegram_env(self):
        os.environ["TELEGRAM_BOT_NAME"] = "Manolo"
        os.environ["TELEGRAM_BOT_USERNAME"] = "ManoloBot"
        os.environ["TELEGRAM_BOT_TOKEN"] = "1234567890"

    def clear_ai_mode_env(self):
        os.environ.pop("AI_MODE", None)
        os.environ.pop("AGENT_MODE", None)

    def test_ai_mode_defaults_to_agent(self):
        self.set_telegram_env()
        self.clear_ai_mode_env()
        config = Config(lazy=True)
        self.assertEqual(config.effective_ai_mode, "agent")

    def test_ai_mode_explicit_value(self):
        self.set_telegram_env()
        self.clear_ai_mode_env()
        os.environ["AI_MODE"] = "deep_agent"
        config = Config(lazy=True)
        self.assertEqual(config.effective_ai_mode, "deep_agent")

    def test_ai_mode_wins_over_agent_mode(self):
        self.set_telegram_env()
        os.environ["AI_MODE"] = "llm"
        os.environ["AGENT_MODE"] = "True"
        config = Config(lazy=True)
        self.assertEqual(config.effective_ai_mode, "llm")
        os.environ.pop("AI_MODE", None)
        os.environ.pop("AGENT_MODE", None)

    def test_agent_mode_backward_compat(self):
        self.set_telegram_env()
        self.clear_ai_mode_env()
        os.environ["AGENT_MODE"] = "True"
        config = Config(lazy=True)
        os.environ.pop("AGENT_MODE", None)
        self.assertEqual(config.effective_ai_mode, "agent")

    def test_deep_agent_workspace_path_default(self):
        import os.path
        import tempfile

        self.set_telegram_env()
        config = Config(lazy=True)
        self.assertEqual(
            config.deep_agent_workspace_path, os.path.join(tempfile.gettempdir(), "manolo_bot", "workspace")
        )


class TestSchedulerConfig(unittest.TestCase):
    SCHEDULER_ENV = [
        "SCHEDULER_ENABLED",
        "SCHEDULER_STORE",
        "SCHEDULER_ENGINE",
        "SCHEDULER_STORE_PATH",
        "SCHEDULER_TIMEZONE",
        "SCHEDULER_POLL_INTERVAL_SECONDS",
        "SCHEDULER_MAX_ACTIONS_PER_CHAT",
        "SCHEDULER_MAX_CONSECUTIVE_FAILURES",
        "SCHEDULER_CATCHUP_GRACE_SECONDS",
    ]

    def setUp(self):
        os.environ["TELEGRAM_BOT_NAME"] = "Manolo"
        os.environ["TELEGRAM_BOT_USERNAME"] = "ManoloBot"
        os.environ["TELEGRAM_BOT_TOKEN"] = "1234567890"
        for var in self.SCHEDULER_ENV:
            os.environ.pop(var, None)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for var in self.SCHEDULER_ENV:
            os.environ.pop(var, None)

    def test_scheduler_defaults(self):
        import os.path
        import tempfile

        config = Config(lazy=True)
        self.assertFalse(config.scheduler_enabled)
        self.assertEqual(config.scheduler_store, "in_memory")
        self.assertEqual(config.scheduler_engine, "asyncio")
        self.assertEqual(
            config.scheduler_store_path,
            os.path.join(tempfile.gettempdir(), "manolo_bot", "scheduler"),
        )
        self.assertEqual(config.scheduler_timezone, "UTC")
        self.assertEqual(config.scheduler_poll_interval_seconds, 5.0)
        self.assertEqual(config.scheduler_max_actions_per_chat, 20)
        self.assertEqual(config.scheduler_max_consecutive_failures, 5)
        self.assertEqual(config.scheduler_catchup_grace_seconds, 0)

    def test_scheduler_env_parsing(self):
        os.environ["SCHEDULER_ENABLED"] = "True"
        os.environ["SCHEDULER_STORE"] = "local_fs"
        os.environ["SCHEDULER_ENGINE"] = "manual"
        os.environ["SCHEDULER_STORE_PATH"] = "/tmp/scheduler-test"
        os.environ["SCHEDULER_TIMEZONE"] = "Europe/Madrid"
        os.environ["SCHEDULER_POLL_INTERVAL_SECONDS"] = "1.5"
        os.environ["SCHEDULER_MAX_ACTIONS_PER_CHAT"] = "7"
        os.environ["SCHEDULER_MAX_CONSECUTIVE_FAILURES"] = "2"
        os.environ["SCHEDULER_CATCHUP_GRACE_SECONDS"] = "3600"

        config = Config(lazy=True)
        self.assertTrue(config.scheduler_enabled)
        self.assertEqual(config.scheduler_store, "local_fs")
        self.assertEqual(config.scheduler_engine, "manual")
        self.assertEqual(config.scheduler_store_path, "/tmp/scheduler-test")
        self.assertEqual(config.scheduler_timezone, "Europe/Madrid")
        self.assertEqual(config.scheduler_poll_interval_seconds, 1.5)
        self.assertEqual(config.scheduler_max_actions_per_chat, 7)
        self.assertEqual(config.scheduler_max_consecutive_failures, 2)
        self.assertEqual(config.scheduler_catchup_grace_seconds, 3600)


if __name__ == "__main__":
    unittest.main()
