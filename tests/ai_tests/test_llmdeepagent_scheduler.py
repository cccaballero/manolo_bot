"""Deep-agent scheduler wiring tests: `scheduler=` forwarding and tool injection.

Fully offline: the scheduler service module is never imported (a fake stands in).
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from manolo_bot.ai.config import BotConfig
from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.ai.llmdeepagent import LLMDeepAgent
from manolo_bot.scheduler.base import (
    DELETE_SCHEDULED_ACTION_TOOL_NAME,
    LIST_SCHEDULED_ACTIONS_TOOL_NAME,
    SCHEDULE_ACTION_TOOL_NAME,
)


class FakeTool(BaseTool):
    """Named no-op tool standing in for scheduler or MCP-provided tools."""

    name: str
    description: str = "no-op test tool"

    def _run(self, *args, **kwargs) -> str:
        return "no-op"


class FakeScheduler:
    """Minimal stand-in for ``SchedulerService`` used by the deep-agent tests."""

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = list(tools)
        self.as_tools_calls = 0

    def as_tools(self) -> list[BaseTool]:
        self.as_tools_calls += 1
        return list(self._tools)

    def as_instructions(self) -> str:
        return "\n\n## Scheduling\n\nfake scheduler instructions"

    async def active_instructions(self, scope: str) -> str:
        return "\n\n## Active scheduled actions in this chat\n- id=fake"


def _make_config(**overrides) -> BotConfig:
    defaults = dict(
        bot_uuid="test-bot",
        bot_name="TestBot",
        bot_username="test_bot",
        bot_token="123456:ABC",
        user_id=0,
        use_tools=False,
        enable_mcp=False,
        mcp_servers_config={},
        context_max_tokens=4000,
        web_content_request_timeout=30,
        can_use_tavily_search=False,
        max_document_size=10 * 1024 * 1024,
        max_voice_size=10 * 1024 * 1024,
    )
    defaults.update(overrides)
    return BotConfig(**defaults)


def _make_llm() -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.get_num_tokens = MagicMock(return_value=10)
    mock_llm.get_num_tokens_from_messages = MagicMock(return_value=10)
    return mock_llm


class TestLLMDeepAgentScheduler(unittest.IsolatedAsyncioTestCase):
    def test_scheduler_is_forwarded_to_llm_agent(self):
        # Arrange
        scheduler = FakeScheduler([FakeTool(name=SCHEDULE_ACTION_TOOL_NAME)])

        # Act
        agent = LLMDeepAgent(
            _make_llm(),
            _make_config(),
            [SystemMessage(content="You are a helpful assistant")],
            MagicMock(),
            scheduler=scheduler,
        )

        # Assert: the base class stores it, so LLMAgent.init actually ran with it.
        self.assertIsInstance(agent, LLMAgent)
        self.assertIs(agent._scheduler, scheduler)

    def test_scheduler_defaults_to_none(self):
        # Act
        agent = LLMDeepAgent(
            _make_llm(),
            _make_config(),
            [SystemMessage(content="You are a helpful assistant")],
            MagicMock(),
        )

        # Assert
        self.assertIsNone(agent._scheduler)

    def test_schedule_scope_is_forwarded_to_llm_agent(self):
        # Act
        agent = LLMDeepAgent(
            _make_llm(),
            _make_config(),
            [SystemMessage(content="You are a helpful assistant")],
            MagicMock(),
            schedule_scope="C012345",
        )

        # Assert: opaque scope stored by the base class, not the storage key.
        self.assertEqual(agent._schedule_scope, "C012345")

    def test_schedule_scope_defaults_to_none(self):
        # Act
        agent = LLMDeepAgent(
            _make_llm(),
            _make_config(),
            [SystemMessage(content="You are a helpful assistant")],
            MagicMock(),
        )

        # Assert
        self.assertIsNone(agent._schedule_scope)

    @patch("manolo_bot.ai.llmdeepagent.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_initialize_path_injects_scheduler_tools(
        self,
        mock_create_deep_agent,
        mock_create_agent,
        mock_get_all_tools,
        mock_deep_get_all_tools,
    ):
        # Arrange
        search_tool = FakeTool(name="search")
        mock_get_all_tools.return_value = [search_tool]
        mock_deep_get_all_tools.return_value = [search_tool]
        mock_create_agent.return_value = MagicMock()
        mock_create_deep_agent.return_value = MagicMock()

        schedule_tool = FakeTool(name=SCHEDULE_ACTION_TOOL_NAME)
        list_tool = FakeTool(name=LIST_SCHEDULED_ACTIONS_TOOL_NAME)
        delete_tool = FakeTool(name=DELETE_SCHEDULED_ACTION_TOOL_NAME)
        scheduler = FakeScheduler([schedule_tool, list_tool, delete_tool])

        # Act
        agent = LLMDeepAgent(
            _make_llm(),
            _make_config(),
            [SystemMessage(content="You are a helpful assistant")],
            MagicMock(),
            scheduler=scheduler,
        )
        await agent.initialize_async_resources()

        # Assert: create_deep_agent received the custom tool plus scheduler tools.
        tools = mock_create_deep_agent.call_args.kwargs["tools"]
        self.assertEqual(
            [t.name for t in tools],
            [
                "search",
                SCHEDULE_ACTION_TOOL_NAME,
                LIST_SCHEDULED_ACTIONS_TOOL_NAME,
                DELETE_SCHEDULED_ACTION_TOOL_NAME,
            ],
        )
        self.assertIs(tools[1], schedule_tool)
        self.assertIs(tools[2], list_tool)
        self.assertIs(tools[3], delete_tool)
        self.assertEqual(agent.agent, mock_create_deep_agent.return_value)

    @patch("manolo_bot.ai.llmdeepagent.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.tools.get_all_tools", new_callable=AsyncMock)
    @patch("manolo_bot.ai.llmagent.create_agent")
    @patch("manolo_bot.ai.llmdeepagent.create_deep_agent")
    async def test_initialize_path_without_scheduler_keeps_tools(
        self,
        mock_create_deep_agent,
        mock_create_agent,
        mock_get_all_tools,
        mock_deep_get_all_tools,
    ):
        # Arrange
        search_tool = FakeTool(name="search")
        mock_get_all_tools.return_value = [search_tool]
        mock_deep_get_all_tools.return_value = [search_tool]
        mock_create_agent.return_value = MagicMock()
        mock_create_deep_agent.return_value = MagicMock()

        # Act
        agent = LLMDeepAgent(
            _make_llm(),
            _make_config(),
            [SystemMessage(content="You are a helpful assistant")],
            MagicMock(),
        )
        await agent.initialize_async_resources()

        # Assert
        self.assertEqual(mock_create_deep_agent.call_args.kwargs["tools"], [search_tool])


if __name__ == "__main__":
    unittest.main()
