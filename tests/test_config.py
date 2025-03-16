import os
import unittest

from config import Config


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
        os.environ["PREFERRED_LANGUAGE"] = "Spanish"
        os.environ["ADD_NO_ANSWER"] = "False"
        os.environ["RATE_LIMITER_REQUESTS_PER_SECOND"] = "0.25"
        os.environ["RATE_LIMITER_CHECK_EVERY_N_SECONDS"] = "0.1"
        os.environ["RATE_LIMITER_MAX_BUCKET_SIZE"] = "10"
        os.environ["ENABLE_MULTIMODAL"] = "True"
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
        self.assertEqual(config.preferred_language, "Spanish")
        self.assertEqual(config.add_no_answer, False)
        self.assertEqual(config.rate_limiter_requests_per_second, 0.25)
        self.assertEqual(config.rate_limiter_check_every_n_seconds, 0.1)
        self.assertEqual(config.rate_limiter_max_bucket_size, 10)
        self.assertEqual(config.is_image_multimodal, True)
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
        del os.environ["PREFERRED_LANGUAGE"]
        del os.environ["ADD_NO_ANSWER"]
        del os.environ["RATE_LIMITER_REQUESTS_PER_SECOND"]
        del os.environ["RATE_LIMITER_CHECK_EVERY_N_SECONDS"]
        del os.environ["RATE_LIMITER_MAX_BUCKET_SIZE"]
        del os.environ["ENABLE_MULTIMODAL"]
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


if __name__ == "__main__":
    unittest.main()
