"""Tests for the scheduler agent tools and their RunnableConfig chat scoping."""

from __future__ import annotations

import itertools
import unittest
from datetime import datetime, timedelta

from manolo_bot.scheduler.base import (
    BASE_SCHEDULER_TOOL_NAMES,
    STATUS_FAILED,
    ScheduledAction,
)
from manolo_bot.scheduler.service import SchedulerService
from manolo_bot.scheduler.tools import (
    INTERNAL_ERROR_MESSAGE,
    NO_CHAT_CONTEXT_MESSAGE,
    build_scheduler_tools,
)
from tests.scheduler_tests.test_service import (
    NOW,
    FakeStore,
    RecordingEngine,
    RecordingHandler,
)

CHAT_ID = "7"
CONFIG = {"metadata": {"chat_id": CHAT_ID}}
NUMERIC_CONFIG = {"metadata": {"chat_id": 7}}
OPAQUE_SCOPE = "C012345"
OPAQUE_CONFIG = {"metadata": {"chat_id": OPAQUE_SCOPE}}


class BrokenStore(FakeStore):
    """Store double whose add and list paths always fail."""

    async def add(self, action: ScheduledAction) -> ScheduledAction:
        raise RuntimeError("store exploded")

    async def list(self, chat_id: str) -> list[ScheduledAction]:
        raise RuntimeError("store exploded")


class ToolsFixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._ids = itertools.count(1)
        self.store = FakeStore()
        self.engine = RecordingEngine()
        self.handler = RecordingHandler()
        self.service = SchedulerService(
            self.store,
            self.engine,
            handler=self.handler,
            clock=lambda: NOW,
        )
        self.tools = {tool.name: tool for tool in build_scheduler_tools(self.service)}

    async def seed(
        self,
        *,
        prompt: str = "do the thing",
        run_at: datetime = NOW,
        chat_id: str = CHAT_ID,
        cron: str | None = None,
        max_runs: int | None = None,
        status: str = "active",
        last_error: str | None = None,
    ) -> ScheduledAction:
        action = ScheduledAction(
            id=f"action-{next(self._ids)}",
            chat_id=chat_id,
            prompt=prompt,
            run_at=run_at,
            created_at=NOW - timedelta(hours=1),
            cron=cron,
            max_runs=max_runs,
            status=status,
            last_error=last_error,
        )
        await self.store.add(action)
        return action

    def build_tools_with(self, store: FakeStore) -> dict:
        service = SchedulerService(store, RecordingEngine(), handler=self.handler, clock=lambda: NOW)
        return {tool.name: tool for tool in build_scheduler_tools(service)}


class TestToolNames(ToolsFixture):
    def test_tool_names_match_constants(self) -> None:
        self.assertEqual([tool.name for tool in self.tools.values()], list(BASE_SCHEDULER_TOOL_NAMES))
        self.assertEqual(set(self.tools), set(BASE_SCHEDULER_TOOL_NAMES))

    def test_tools_do_not_accept_chat_id(self) -> None:
        for name, tool in self.tools.items():
            with self.subTest(tool=name):
                self.assertNotIn("chat_id", tool.args)


class TestScheduleTool(ToolsFixture):
    async def test_schedule_uses_chat_id_from_config(self) -> None:
        result = await self.tools["schedule_action"].ainvoke(
            {"prompt": "send the report", "run_at": "2026-10-20T13:00:00+00:00"},
            config=CONFIG,
        )

        self.assertEqual(len(self.store.added), 1)
        action = self.store.added[0]
        self.assertEqual(action.chat_id, CHAT_ID)
        self.assertIn(action.id, result)
        self.assertIn(action.run_at.isoformat(), result)
        self.assertIn("one-shot", result)
        self.assertIn("UTC", result)

    async def test_schedule_accepts_numeric_and_string_chat_ids(self) -> None:
        result = await self.tools["schedule_action"].ainvoke(
            {"prompt": "ping", "run_at": "2026-10-20T13:00:00Z"},
            config=NUMERIC_CONFIG,
        )
        self.assertIn("Scheduled action", result)
        self.assertEqual(self.store.added[0].chat_id, CHAT_ID)

        result = await self.tools["schedule_action"].ainvoke(
            {"prompt": "pong", "run_at": "2026-10-20T13:00:00Z"},
            config=CONFIG,
        )
        self.assertIn("Scheduled action", result)
        self.assertEqual(self.store.added[1].chat_id, CHAT_ID)

    async def test_schedule_accepts_opaque_scope(self) -> None:
        result = await self.tools["schedule_action"].ainvoke(
            {"prompt": "ping", "run_at": "2026-10-20T13:00:00Z"},
            config=OPAQUE_CONFIG,
        )

        self.assertIn("Scheduled action", result)
        self.assertEqual(self.store.added[0].chat_id, OPAQUE_SCOPE)

    async def test_schedule_scope_overrides_chat_id(self) -> None:
        result = await self.tools["schedule_action"].ainvoke(
            {"prompt": "ping", "run_at": "2026-10-20T13:00:00Z"},
            config={"metadata": {"schedule_scope": OPAQUE_SCOPE, "chat_id": CHAT_ID}},
        )

        self.assertIn("Scheduled action", result)
        self.assertEqual(self.store.added[0].chat_id, OPAQUE_SCOPE)

    async def test_schedule_reports_cron_and_max_runs(self) -> None:
        result = await self.tools["schedule_action"].ainvoke(
            {"prompt": "daily digest", "cron": "0 9 * * *", "max_runs": 3},
            config=CONFIG,
        )
        self.assertIn("cron '0 9 * * *'", result)
        self.assertIn("max_runs: 3", result)

    async def test_schedule_without_chat_context_returns_unavailable(self) -> None:
        configs = (
            None,
            {},
            {"metadata": {}},
            {"metadata": {"chat_id": None}},
            {"metadata": {"chat_id": ""}},
            {"metadata": {"chat_id": "   "}},
            {"metadata": {"schedule_scope": "  "}},
        )
        for config in configs:
            with self.subTest(config=config):
                result = await self.tools["schedule_action"].ainvoke(
                    {"prompt": "ping", "run_at": "2026-10-20T13:00:00Z"},
                    config=config,
                )
                self.assertEqual(result, NO_CHAT_CONTEXT_MESSAGE)
        self.assertEqual(self.store.added, [])

    async def test_schedule_validation_error_is_returned_as_text(self) -> None:
        result = await self.tools["schedule_action"].ainvoke({"prompt": "ping"}, config=CONFIG)
        self.assertIn("exactly one", result)
        self.assertEqual(self.store.added, [])

    async def test_schedule_bad_run_at_is_returned_as_text(self) -> None:
        result = await self.tools["schedule_action"].ainvoke(
            {"prompt": "ping", "run_at": "someday"},
            config=CONFIG,
        )
        self.assertIn("ISO-8601", result)

    async def test_schedule_unexpected_error_returns_friendly_message(self) -> None:
        tools = self.build_tools_with(BrokenStore())
        with self.assertLogs("manolo_bot.scheduler.tools", level="ERROR"):
            result = await tools["schedule_action"].ainvoke(
                {"prompt": "ping", "run_at": "2026-10-20T13:00:00Z"},
                config=CONFIG,
            )
        self.assertEqual(result, INTERNAL_ERROR_MESSAGE)


class TestListTool(ToolsFixture):
    async def test_list_empty_chat(self) -> None:
        result = await self.tools["list_scheduled_actions"].ainvoke({}, config=CONFIG)
        self.assertEqual(result, "No scheduled actions in this chat.")

    async def test_list_formats_only_the_current_chat(self) -> None:
        own = await self.seed(prompt="send report", run_at=NOW + timedelta(hours=1), max_runs=2)
        await self.seed(prompt="other chat", chat_id=99)

        result = await self.tools["list_scheduled_actions"].ainvoke({}, config=CONFIG)

        self.assertIn(own.id, result)
        self.assertIn("one-shot", result)
        self.assertIn("runs=0/2", result)
        self.assertIn("status=active", result)
        self.assertIn(repr("send report"), result)
        self.assertNotIn("other chat", result)

    async def test_list_includes_cron_infinity_and_last_error(self) -> None:
        await self.seed(prompt="recurring", cron="0 9 * * *", run_at=NOW + timedelta(hours=1))
        await self.seed(
            prompt="broken",
            status=STATUS_FAILED,
            run_at=NOW - timedelta(minutes=1),
            last_error="RuntimeError: boom",
        )

        result = await self.tools["list_scheduled_actions"].ainvoke({}, config=CONFIG)

        self.assertIn("cron '0 9 * * *'", result)
        self.assertIn("runs=0/inf", result)
        self.assertIn("status=failed", result)
        self.assertIn("last_error='RuntimeError: boom'", result)

    async def test_list_truncates_long_prompts(self) -> None:
        long_prompt = "x" * 1000
        await self.seed(prompt=long_prompt, run_at=NOW + timedelta(hours=1))

        result = await self.tools["list_scheduled_actions"].ainvoke({}, config=CONFIG)

        self.assertNotIn(long_prompt, result)
        self.assertIn("…", result)
        self.assertLess(len(result), 700)

    async def test_list_without_chat_context_returns_unavailable(self) -> None:
        result = await self.tools["list_scheduled_actions"].ainvoke({})
        self.assertEqual(result, NO_CHAT_CONTEXT_MESSAGE)


class TestToolDescriptions(ToolsFixture):
    def test_schedule_tool_explains_silent_no_answer(self) -> None:
        self.assertIn("NO_ANSWER", self.tools["schedule_action"].description)


class TestToolLogging(ToolsFixture):
    async def test_delete_logs_the_outcome(self) -> None:
        action = await self.seed()

        with self.assertLogs("manolo_bot.scheduler.tools", level="INFO") as logs:
            await self.tools["delete_scheduled_action"].ainvoke({"action_id": action.id}, config=CONFIG)

        self.assertTrue(any(action.id in line and "deleted=True" in line for line in logs.output))


class TestDeleteTool(ToolsFixture):
    async def test_delete_reports_both_branches(self) -> None:
        action = await self.seed()

        deleted = await self.tools["delete_scheduled_action"].ainvoke({"action_id": action.id}, config=CONFIG)
        missing = await self.tools["delete_scheduled_action"].ainvoke({"action_id": action.id}, config=CONFIG)

        self.assertEqual(deleted, f"Deleted scheduled action {action.id}.")
        self.assertEqual(missing, f"No scheduled action {action.id} in this chat.")

    async def test_delete_cannot_reach_another_chat(self) -> None:
        foreign = await self.seed(chat_id=99)

        result = await self.tools["delete_scheduled_action"].ainvoke({"action_id": foreign.id}, config=CONFIG)

        self.assertEqual(result, f"No scheduled action {foreign.id} in this chat.")
        self.assertIn(foreign.id, self.store.actions)

    async def test_delete_without_chat_context_returns_unavailable(self) -> None:
        result = await self.tools["delete_scheduled_action"].ainvoke({"action_id": "action-1"})
        self.assertEqual(result, NO_CHAT_CONTEXT_MESSAGE)


class TestOpaqueScopeTools(ToolsFixture):
    async def test_list_and_delete_use_the_opaque_scope(self) -> None:
        own = await self.seed(chat_id=OPAQUE_SCOPE, run_at=NOW + timedelta(hours=1))
        foreign = await self.seed(chat_id=CHAT_ID, run_at=NOW + timedelta(hours=1))

        result = await self.tools["list_scheduled_actions"].ainvoke({}, config=OPAQUE_CONFIG)
        self.assertIn(own.id, result)
        self.assertNotIn(foreign.id, result)

        crossed = await self.tools["delete_scheduled_action"].ainvoke(
            {"action_id": foreign.id},
            config=OPAQUE_CONFIG,
        )
        self.assertEqual(crossed, f"No scheduled action {foreign.id} in this chat.")
        self.assertIn(foreign.id, self.store.actions)

        deleted = await self.tools["delete_scheduled_action"].ainvoke(
            {"action_id": own.id},
            config=OPAQUE_CONFIG,
        )
        self.assertEqual(deleted, f"Deleted scheduled action {own.id}.")
        self.assertNotIn(own.id, self.store.actions)

    async def test_schedule_scope_drives_list_and_delete(self) -> None:
        action = await self.seed(chat_id=OPAQUE_SCOPE, run_at=NOW + timedelta(hours=1))
        config = {"metadata": {"schedule_scope": OPAQUE_SCOPE, "chat_id": CHAT_ID}}

        listed = await self.tools["list_scheduled_actions"].ainvoke({}, config=config)
        self.assertIn(action.id, listed)

        deleted = await self.tools["delete_scheduled_action"].ainvoke({"action_id": action.id}, config=config)
        self.assertEqual(deleted, f"Deleted scheduled action {action.id}.")
