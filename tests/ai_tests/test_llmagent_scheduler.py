"""Agent-level scheduler tests: `scheduler=` param, tool resolution, trigger turns.

Fully offline: the scheduler service module is never imported (the service is
still being implemented in parallel), a fake object stands in for it.
"""

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.config import Config
from manolo_bot.scheduler.base import (
    DELETE_SCHEDULED_ACTION_TOOL_NAME,
    LIST_SCHEDULED_ACTIONS_TOOL_NAME,
    SCHEDULE_ACTION_TOOL_NAME,
    SCHEDULED_TRIGGER_PREFIX,
    STATUS_FAILED,
    ScheduledAction,
)
from manolo_bot.scheduler.prompting import (
    build_active_schedules_block,
    build_scheduler_instructions,
)


class FakeTool(BaseTool):
    """Named no-op tool standing in for scheduler or MCP-provided tools."""

    name: str
    description: str = "no-op test tool"

    def _run(self, *args, **kwargs) -> str:
        return "no-op"


class FakeScheduler:
    """Minimal stand-in for ``SchedulerService`` used by the agent tests."""

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = list(tools)
        self.as_tools_calls = 0
        self.as_instructions_calls = 0
        self.active_instructions_calls = 0
        self.active_scope = None

    def as_tools(self) -> list[BaseTool]:
        self.as_tools_calls += 1
        return list(self._tools)

    def as_instructions(self) -> str:
        self.as_instructions_calls += 1
        return "\n\n## Scheduling\n\nfake scheduler instructions"

    async def active_instructions(self, scope: str) -> str:
        self.active_instructions_calls += 1
        self.active_scope = scope
        return "\n\n## Active scheduled actions in this chat\n- id=fake"


def _make_agent(scheduler=None, system_instructions=None, schedule_scope=None) -> LLMAgent:
    """Build an LLMAgent with mocks (no network, no real config)."""
    mock_llm = MagicMock()
    mock_config = MagicMock(spec=Config)
    mock_config.enable_mcp = False
    mock_config.use_tools = False
    mock_storage = MagicMock()
    mock_storage.chat_id = 4242
    if system_instructions is None:
        system_instructions = [SystemMessage(content="You are a helpful assistant")]
    return LLMAgent(
        mock_llm,
        mock_config,
        system_instructions,
        mock_storage,
        scheduler=scheduler,
        schedule_scope=schedule_scope,
    )


class TestLLMAgentSchedulerTools(unittest.IsolatedAsyncioTestCase):
    def test_no_scheduler_returns_tools_unchanged(self):
        # Arrange
        agent = _make_agent()
        tools = [FakeTool(name="search")]

        # Act
        resolved = agent._resolve_scheduler_tools(tools)

        # Assert
        self.assertIs(resolved, tools)
        self.assertEqual([t.name for t in resolved], ["search"])

    def test_scheduler_tools_are_appended_to_a_new_list(self):
        # Arrange
        list_tool = FakeTool(name=LIST_SCHEDULED_ACTIONS_TOOL_NAME)
        delete_tool = FakeTool(name=DELETE_SCHEDULED_ACTION_TOOL_NAME)
        scheduler = FakeScheduler([list_tool, delete_tool])
        agent = _make_agent(scheduler=scheduler)
        tools = [FakeTool(name="search")]

        # Act
        resolved = agent._resolve_scheduler_tools(tools)

        # Assert: new list, service tools appended in order, input untouched.
        self.assertIsNot(resolved, tools)
        self.assertEqual(
            [t.name for t in resolved],
            ["search", LIST_SCHEDULED_ACTIONS_TOOL_NAME, DELETE_SCHEDULED_ACTION_TOOL_NAME],
        )
        self.assertIs(resolved[1], list_tool)
        self.assertIs(resolved[2], delete_tool)
        self.assertEqual([t.name for t in tools], ["search"])
        self.assertIsNone(agent.tools)

    def test_existing_tool_with_same_name_wins(self):
        # Arrange: an MCP/custom tool already exposes schedule_action.
        existing = FakeTool(name=SCHEDULE_ACTION_TOOL_NAME)
        service_schedule = FakeTool(name=SCHEDULE_ACTION_TOOL_NAME)
        service_list = FakeTool(name=LIST_SCHEDULED_ACTIONS_TOOL_NAME)
        agent = _make_agent(scheduler=FakeScheduler([service_schedule, service_list]))

        # Act
        resolved = agent._resolve_scheduler_tools([existing])

        # Assert: the existing object is kept, the twin is skipped, no duplicates.
        self.assertEqual([t.name for t in resolved], [SCHEDULE_ACTION_TOOL_NAME, LIST_SCHEDULED_ACTIONS_TOOL_NAME])
        self.assertIs(resolved[0], existing)
        self.assertIs(resolved[1], service_list)

    async def test_answer_scheduled_action_prefixes_the_prompt(self):
        # Arrange
        agent = _make_agent()
        agent.answer_message = AsyncMock(return_value=AIMessage(content="Watered the plants"))

        # Act
        response = await agent.answer_scheduled_action(42, "Water the plants")

        # Assert: exact prefixed text, same chat, response passed through.
        agent.answer_message.assert_awaited_once_with(42, f"{SCHEDULED_TRIGGER_PREFIX}Water the plants")
        self.assertEqual(response.content, "Watered the plants")

    async def test_answer_scheduled_action_includes_action_id(self):
        # Arrange
        agent = _make_agent()
        agent.answer_message = AsyncMock(return_value=AIMessage(content="Done"))

        # Act
        await agent.answer_scheduled_action(42, "Water the plants", action_id="abc123")

        # Assert
        agent.answer_message.assert_awaited_once_with(42, f"{SCHEDULED_TRIGGER_PREFIX}id=abc123 Water the plants")

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_initialize_async_resources_includes_scheduler_tools(self, mock_create_agent, mock_get_all_tools):
        # Arrange
        search_tool = FakeTool(name="search")
        mock_get_all_tools.return_value = [search_tool]
        captured: dict = {}

        def _create(model, tools):
            captured["tools"] = tools
            return MagicMock()

        mock_create_agent.side_effect = _create
        scheduler = FakeScheduler(
            [
                FakeTool(name=SCHEDULE_ACTION_TOOL_NAME),
                FakeTool(name=LIST_SCHEDULED_ACTIONS_TOOL_NAME),
                FakeTool(name=DELETE_SCHEDULED_ACTION_TOOL_NAME),
            ]
        )

        # Act
        agent = _make_agent(scheduler=scheduler)
        await agent.initialize_async_resources()

        # Assert
        self.assertEqual(
            [t.name for t in captured["tools"]],
            [
                "search",
                SCHEDULE_ACTION_TOOL_NAME,
                LIST_SCHEDULED_ACTIONS_TOOL_NAME,
                DELETE_SCHEDULED_ACTION_TOOL_NAME,
            ],
        )
        self.assertEqual(scheduler.as_tools_calls, 1)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_initialize_injects_scheduler_instructions(self, mock_create_agent, mock_get_all_tools):
        # Arrange
        mock_get_all_tools.return_value = []
        scheduler = FakeScheduler([])

        # Act
        agent = _make_agent(scheduler=scheduler)
        await agent.initialize_async_resources()

        # Assert
        content = agent.system_instructions[0].content
        self.assertIn("fake scheduler instructions", content)
        self.assertIn("id=fake", content)
        self.assertEqual(scheduler.as_instructions_calls, 1)
        self.assertEqual(scheduler.active_instructions_calls, 1)
        self.assertEqual(scheduler.active_scope, "4242")

    async def test_composition_uses_schedule_scope_when_set(self) -> None:
        # Arrange: opaque scheduler scope distinct from the storage chat id.
        scheduler = FakeScheduler([])
        agent = _make_agent(scheduler=scheduler, schedule_scope="C012345")

        # Act
        composed = await agent._compose_system_instructions()

        # Assert: scope wins over the conversation storage key.
        self.assertEqual(scheduler.active_scope, "C012345")
        self.assertIn("id=fake", composed[0].content)

    async def test_composition_falls_back_to_storage_chat_id_without_scope(self) -> None:
        # Arrange
        scheduler = FakeScheduler([])
        agent = _make_agent(scheduler=scheduler)

        # Act
        await agent._compose_system_instructions()

        # Assert: unchanged pre-scope behavior.
        self.assertEqual(scheduler.active_scope, str(agent.messages_storage.chat_id))

    def test_get_langchain_config_includes_schedule_scope_when_set(self) -> None:
        # Arrange
        agent = _make_agent(schedule_scope="C012345")

        # Act
        config = agent._get_langchain_config(99)

        # Assert: metadata carries both the scope and the raw chat id.
        self.assertEqual(config["metadata"]["schedule_scope"], "C012345")
        self.assertEqual(config["metadata"]["chat_id"], 99)

    def test_get_langchain_config_omits_schedule_scope_when_none(self) -> None:
        # Arrange
        agent = _make_agent()

        # Act
        config = agent._get_langchain_config(99)

        # Assert
        self.assertNotIn("schedule_scope", config["metadata"])
        self.assertEqual(config["metadata"]["chat_id"], 99)

    def test_blank_schedule_scope_behaves_as_none(self) -> None:
        # Arrange
        agent = _make_agent(schedule_scope="   ")

        # Act
        config = agent._get_langchain_config(99)

        # Assert
        self.assertIsNone(agent._schedule_scope)
        self.assertNotIn("schedule_scope", config["metadata"])

    def test_padded_schedule_scope_is_canonicalized(self) -> None:
        # Arrange
        agent = _make_agent(schedule_scope="  C012345\t")

        # Act
        config = agent._get_langchain_config(99)

        # Assert: the padding never becomes part of the scheduler namespace.
        self.assertEqual(agent._schedule_scope, "C012345")
        self.assertEqual(config["metadata"]["schedule_scope"], "C012345")

    async def test_blank_schedule_scope_falls_back_during_composition(self) -> None:
        # Arrange
        scheduler = FakeScheduler([])
        agent = _make_agent(scheduler=scheduler, schedule_scope="\t ")

        # Act
        await agent._compose_system_instructions()

        # Assert
        self.assertEqual(scheduler.active_scope, str(agent.messages_storage.chat_id))

    async def test_composition_preserves_system_message_metadata(self) -> None:
        system = SystemMessage(
            content="base",
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        )
        agent = _make_agent(scheduler=FakeScheduler([]), system_instructions=[system])

        composed = await agent._compose_system_instructions()

        self.assertEqual(composed[0].additional_kwargs, system.additional_kwargs)
        self.assertIn("fake scheduler instructions", composed[0].content)

    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    async def test_initialize_async_resources_without_scheduler_keeps_tools(
        self, mock_create_agent, mock_get_all_tools
    ):
        # Arrange
        search_tool = FakeTool(name="search")
        mock_get_all_tools.return_value = [search_tool]
        captured: dict = {}

        def _create(model, tools):
            captured["tools"] = tools
            return MagicMock()

        mock_create_agent.side_effect = _create

        # Act
        agent = _make_agent()
        await agent.initialize_async_resources()

        # Assert
        self.assertEqual(captured["tools"], [search_tool])


class TestSchedulerPrompting(unittest.TestCase):
    def test_instructions_mention_tools_timezone_limit_and_trigger(self):
        # Act
        text = build_scheduler_instructions(default_timezone="Europe/Madrid", max_actions_per_chat=7)

        # Assert
        for tool_name in (
            SCHEDULE_ACTION_TOOL_NAME,
            LIST_SCHEDULED_ACTIONS_TOOL_NAME,
            DELETE_SCHEDULED_ACTION_TOOL_NAME,
        ):
            self.assertIn(f"`{tool_name}`", text)
        self.assertIn("Europe/Madrid", text)
        self.assertIn("5-field", text)
        self.assertIn("cron", text)
        self.assertIn("ISO-8601", text)
        self.assertIn(SCHEDULED_TRIGGER_PREFIX.strip(), text)
        self.assertIn("NO_ANSWER", text)
        self.assertIn("survive `/flushcontext`", text)
        self.assertIn("Never claim", text)
        self.assertIn("one execution", text)
        self.assertIn("7", text)
        self.assertIn("THIS chat", text)

    def test_defaults(self):
        # Act
        text = build_scheduler_instructions()

        # Assert
        self.assertIn("UTC", text)
        self.assertIn("20", text)


class TestActiveSchedulesBlock(unittest.TestCase):
    NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def _action(self, index: int, *, prompt: str = "do it", cron: str | None = None, status: str = "active"):
        return ScheduledAction(
            id=f"action-{index}",
            chat_id=42,
            prompt=prompt,
            run_at=self.NOW,
            created_at=self.NOW,
            cron=cron,
            status=status,
        )

    def test_empty_chat_returns_empty_string(self):
        self.assertEqual(build_active_schedules_block([]), "")

    def test_renders_ids_schedule_and_tool_hint(self):
        block = build_active_schedules_block([self._action(1, cron="0 9 * * *")])

        self.assertIn("id=action-1", block)
        self.assertIn("cron='0 9 * * *'", block)
        self.assertIn(LIST_SCHEDULED_ACTIONS_TOOL_NAME, block)
        self.assertIn(DELETE_SCHEDULED_ACTION_TOOL_NAME, block)

    def test_truncates_prompts_and_caps_rows(self):
        actions = [self._action(index, prompt="x" * 300) for index in range(12)]

        block = build_active_schedules_block(actions)

        self.assertIn("…", block)
        self.assertIn("and 2 more", block)
        self.assertNotIn("x" * 200, block)

    def test_marks_failed_actions(self):
        block = build_active_schedules_block([self._action(1, status=STATUS_FAILED)])
        self.assertIn("status=failed", block)

    def test_open_actions_win_the_row_budget_over_failed_ones(self):
        failed = [self._action(index, status=STATUS_FAILED) for index in range(10)]
        upcoming = self._action(99, status="active")

        block = build_active_schedules_block([*failed, upcoming])

        self.assertIn("id=action-99", block)


if __name__ == "__main__":
    unittest.main()
