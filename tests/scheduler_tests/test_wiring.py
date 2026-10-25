"""Wiring checks for the scheduler feature (no network/API keys)."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def source(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class TestSchedulerWiring(unittest.TestCase):
    def test_llmbot_has_no_scheduler_wiring(self):
        llmbot = source("manolo_bot/ai/llmbot.py")
        self.assertNotIn("scheduler", llmbot.lower())

    def test_main_wires_the_scheduler_for_agents(self):
        main = source("manolo_bot/main.py")
        self.assertIn("async def init_scheduler", main)
        self.assertIn("build_scheduler_store", main)
        self.assertIn("build_scheduler_engine", main)
        self.assertIn("SchedulerService", main)
        self.assertIn("answer_scheduled_action", main)
        self.assertIn("send_final_response", main)
        self.assertIn("system_instructions", main)
        self.assertNotIn("build_scheduler_instructions", main)
        self.assertNotIn("build_active_schedules_block", main)
        self.assertIn('Command("flushschedules")', main)
        self.assertIn("clear_actions", main)
        self.assertIn("parse_telegram_chat_id(action.chat_id)", main)
        self.assertEqual(main.count("scheduler=_scheduler_service"), 2)

    def test_llm_branch_does_not_receive_a_scheduler(self):
        main = source("manolo_bot/main.py")
        llmbot_call = main.split("llm_bot = LLMBot(")[1].split("return llm_bot")[0]
        self.assertNotIn("scheduler", llmbot_call)

    def test_agents_resolve_scheduler_tools(self):
        agent = source("manolo_bot/ai/llmagent.py")
        deep_agent = source("manolo_bot/ai/llmdeepagent.py")
        self.assertIn("_resolve_scheduler_tools", agent)
        self.assertIn("_resolve_scheduler_tools", deep_agent)
        self.assertIn("as_instructions", agent)
        self.assertIn("active_instructions", agent)
        self.assertIn("SCHEDULED_TRIGGER_PREFIX", agent)
        self.assertIn("schedule_scope", agent)
        self.assertIn("schedule_scope", deep_agent)

    def test_config_exposes_scheduler_fields(self):
        config = source("manolo_bot/config.py")
        for field in [
            "SCHEDULER_ENABLED",
            "SCHEDULER_STORE",
            "SCHEDULER_ENGINE",
            "SCHEDULER_STORE_PATH",
            "SCHEDULER_TIMEZONE",
            "SCHEDULER_POLL_INTERVAL_SECONDS",
            "SCHEDULER_MAX_ACTIONS_PER_CHAT",
            "SCHEDULER_MAX_CONSECUTIVE_FAILURES",
            "SCHEDULER_CATCHUP_GRACE_SECONDS",
        ]:
            self.assertIn(field, config)

    def test_bot_config_has_no_scheduler_fields(self):
        from manolo_bot.ai.config import BotConfig

        bot_config = BotConfig(
            bot_uuid="test-bot",
            bot_name="TestBot",
            bot_username="test_bot",
            bot_token="123456:ABC",
            user_id=0,
        )
        for attr in [
            "scheduler_enabled",
            "scheduler_store",
            "scheduler_engine",
            "scheduler_store_path",
            "scheduler_timezone",
        ]:
            self.assertFalse(hasattr(bot_config, attr), f"BotConfig should not have {attr}")

    def test_env_example_documents_the_scheduler(self):
        env_example = source("env.example")
        self.assertIn("SCHEDULER_ENABLED", env_example)
        self.assertIn("SCHEDULER_ENGINE", env_example)

    def test_factories_are_importable_from_their_packages(self):
        from manolo_bot.scheduler.engines import build_scheduler_engine
        from manolo_bot.scheduler.stores import build_scheduler_store

        self.assertTrue(callable(build_scheduler_store))
        self.assertTrue(callable(build_scheduler_engine))


if __name__ == "__main__":
    unittest.main()
