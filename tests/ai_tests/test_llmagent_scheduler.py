"""Agent-level scheduler tests: `scheduler=` param, tool resolution, trigger turns.

Fully offline: the scheduler service module is never imported (the service is
still being implemented in parallel), a fake object stands in for it.
"""

import unittest
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
)
from manolo_bot.scheduler.prompting import build_scheduler_instructions


class FakeTool(BaseTool):
    """Named no-op tool standing in for scheduler or MCP-provided tools."""

    name: str
    description: str = "no-op test tool"

    def _run(self, *args, **kwargs) -> str:
        return "no-op"


class FakeScheduler:
    """Minimal stand-in for ``SchedulerService`` (only ``as_tools`` is used)."""

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = list(tools)
        self.as_tools_calls = 0

    def as_tools(self) -> list[BaseTool]:
        self.as_tools_calls += 1
        return list(self._tools)


def _make_agent(scheduler=None) -> LLMAgent:
    """Build an LLMAgent with mocks (no network, no real config)."""
    mock_llm = MagicMock()
    mock_config = MagicMock(spec=Config)
    mock_config.enable_mcp = False
    mock_config.use_tools = False
    return LLMAgent(
        mock_llm,
        mock_config,
        [SystemMessage(content="You are a helpful assistant")],
        MagicMock(),
        scheduler=scheduler,
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
        self.assertIn("7", text)
        self.assertIn("THIS chat", text)

    def test_defaults(self):
        # Act
        text = build_scheduler_instructions()

        # Assert
        self.assertIn("UTC", text)
        self.assertIn("20", text)


if __name__ == "__main__":
    unittest.main()
