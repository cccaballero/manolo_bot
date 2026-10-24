"""LangChain tools exposing the scheduler to agents.

The chat id is never a tool argument: it comes from the ``RunnableConfig``
metadata populated by the Telegram layer, so an agent cannot schedule or
inspect actions belonging to another chat.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from manolo_bot.scheduler.base import (
    DELETE_SCHEDULED_ACTION_TOOL_NAME,
    LIST_SCHEDULED_ACTIONS_TOOL_NAME,
    SCHEDULE_ACTION_TOOL_NAME,
    SCHEDULED_TRIGGER_PREFIX,
    STATUS_FAILED,
    ScheduledAction,
    SchedulerError,
)
from manolo_bot.scheduler.service import SchedulerService

logger = logging.getLogger(__name__)

NO_CHAT_CONTEXT_MESSAGE = "Scheduling is unavailable: no chat context was provided to the tool."
INTERNAL_ERROR_MESSAGE = "The scheduling service hit an internal error; please try again later."


def _extract_chat_id(config: RunnableConfig | None) -> int | None:
    """Read ``chat_id`` from runnable metadata, returning None when absent/invalid."""
    metadata = (config or {}).get("metadata") or {}
    value = metadata.get("chat_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_schedule_confirmation(action: ScheduledAction, timezone_name: str) -> str:
    schedule = f"cron {action.cron!r}" if action.cron else "one-shot"
    max_runs = "unlimited" if action.max_runs is None else str(action.max_runs)
    return (
        f"Scheduled action {action.id}.\n"
        f"Next run: {action.run_at.isoformat()} ({schedule}), timezone {timezone_name}.\n"
        f"max_runs: {max_runs}. The action is scoped to this chat; when it fires the prompt is "
        f"sent prefixed with {SCHEDULED_TRIGGER_PREFIX!r}."
    )


#: Maximum prompt length shown by the list tool. Full prompts can reach
#: ``MAX_PROMPT_CHARS``; listing a whole quota would flood the model context.
PROMPT_PREVIEW_CHARS = 300


def _preview(text: str, limit: int = PROMPT_PREVIEW_CHARS) -> str:
    """Shorten a prompt for list output, marking the truncation."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _format_action_line(action: ScheduledAction) -> str:
    schedule = f"cron {action.cron!r}" if action.cron else "one-shot"
    max_runs = "inf" if action.max_runs is None else str(action.max_runs)
    line = " | ".join(
        (
            f"- id={action.id}",
            f"next={action.run_at.isoformat()}",
            schedule,
            f"runs={action.run_count}/{max_runs}",
            f"status={action.status}",
            f"prompt={_preview(action.prompt)!r}",
        )
    )
    if action.status == STATUS_FAILED and action.last_error:
        line += f" | last_error={action.last_error!r}"
    return line


def build_scheduler_tools(service: SchedulerService) -> list[BaseTool]:
    """Build the three scheduler tools bound to one :class:`SchedulerService`."""

    @tool(SCHEDULE_ACTION_TOOL_NAME)
    async def schedule_action(
        prompt: str,
        run_at: str | None = None,
        cron: str | None = None,
        max_runs: int | None = None,
        config: RunnableConfig = None,  # type: ignore[assignment]
    ) -> str:
        """Schedule a chat-scoped instruction for the bot to execute later.

        Provide exactly one of:
        - run_at: an absolute ISO-8601 datetime with a timezone offset,
          e.g. "2026-10-21T09:00:00+02:00". Naive datetimes use the bot timezone.
        - cron: a standard 5-field cron expression
          (minute hour day-of-month month day-of-week), e.g. "0 9 * * 1-5".
          Cron is evaluated in the bot timezone.

        The prompt must be a self-contained instruction: when the action fires
        it runs as a new agent turn in this chat's existing conversation,
        prefixed with SCHEDULED_TRIGGER_PREFIX (the literal marker
        "[SCHEDULED ACTION TRIGGERED] "). Actions belong to the current chat
        only. Use max_runs to cap how many times a recurring action executes.
        If the prompt says to stay silent when a condition is not met, reply
        exactly NO_ANSWER and nothing else; the bot will not post anything.
        """
        chat_id = _extract_chat_id(config)
        if chat_id is None:
            return NO_CHAT_CONTEXT_MESSAGE
        try:
            action = await service.add_action(chat_id, prompt, run_at=run_at, cron=cron, max_runs=max_runs)
        except SchedulerError as exc:
            return str(exc)
        except Exception:
            logger.exception("Failed to schedule an action for chat %s", chat_id)
            return INTERNAL_ERROR_MESSAGE
        return _format_schedule_confirmation(action, service.timezone_name)

    @tool(LIST_SCHEDULED_ACTIONS_TOOL_NAME)
    async def list_scheduled_actions(
        config: RunnableConfig = None,  # type: ignore[assignment]
    ) -> str:
        """List the scheduled actions of the current chat.

        Each line shows the action id (used by the delete tool), the next run
        in ISO-8601, whether it is a cron or one-shot action, the run count
        against max_runs, its status, and a short prompt preview. Failed
        actions also include their last error.
        """
        chat_id = _extract_chat_id(config)
        if chat_id is None:
            return NO_CHAT_CONTEXT_MESSAGE
        try:
            actions = await service.list_actions(chat_id)
        except SchedulerError as exc:
            return str(exc)
        except Exception:
            logger.exception("Failed to list scheduled actions for chat %s", chat_id)
            return INTERNAL_ERROR_MESSAGE
        if not actions:
            return "No scheduled actions in this chat."
        return "\n".join(_format_action_line(action) for action in actions)

    @tool(DELETE_SCHEDULED_ACTION_TOOL_NAME)
    async def delete_scheduled_action(
        action_id: str,
        config: RunnableConfig = None,  # type: ignore[assignment]
    ) -> str:
        """Delete one scheduled action of the current chat by its id.

        Use the ids reported by the list tool. Actions from other chats are
        never visible or deletable here.
        """
        chat_id = _extract_chat_id(config)
        if chat_id is None:
            return NO_CHAT_CONTEXT_MESSAGE
        try:
            deleted = await service.delete_action(chat_id, action_id)
        except SchedulerError as exc:
            return str(exc)
        except Exception:
            logger.exception("Failed to delete scheduled action %s for chat %s", action_id, chat_id)
            return INTERNAL_ERROR_MESSAGE
        if deleted:
            return f"Deleted scheduled action {action_id}."
        return f"No scheduled action {action_id} in this chat."

    return [schedule_action, list_scheduled_actions, delete_scheduled_action]


__all__ = [
    "INTERNAL_ERROR_MESSAGE",
    "NO_CHAT_CONTEXT_MESSAGE",
    "build_scheduler_tools",
]
