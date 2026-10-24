"""Scheduler prompting helpers: system-instruction snippet for scheduled actions.

Kept separate from the agent classes (mirroring :mod:`manolo_bot.rag.prompting`)
so the wording can be reused and tested without constructing an agent.
"""

from __future__ import annotations

from manolo_bot.scheduler.base import (
    DELETE_SCHEDULED_ACTION_TOOL_NAME,
    LIST_SCHEDULED_ACTIONS_TOOL_NAME,
    SCHEDULE_ACTION_TOOL_NAME,
    SCHEDULED_TRIGGER_PREFIX,
)


def build_scheduler_instructions(default_timezone: str = "UTC", max_actions_per_chat: int = 20) -> str:
    """Build a system-instruction snippet explaining the scheduling tools.

    :param default_timezone: Timezone name used to evaluate recurring cron
        expressions; surfaced verbatim so the model knows the reference frame.
    :param max_actions_per_chat: Per-chat quota of open actions, surfaced so
        the model cleans up finished reminders instead of hitting the limit.
    """
    trigger = SCHEDULED_TRIGGER_PREFIX.strip()
    return (
        "\n\n## Scheduling\n\n"
        f"You can schedule future actions in this chat with `{SCHEDULE_ACTION_TOOL_NAME}`,"
        f" inspect them with `{LIST_SCHEDULED_ACTIONS_TOOL_NAME}`, and cancel them with"
        f" `{DELETE_SCHEDULED_ACTION_TOOL_NAME}`.\n"
        "- One-shot actions use an absolute ISO-8601 `run_at`; recurring actions use a 5-field"
        f" cron expression evaluated in {default_timezone}. Compute future times from the"
        " current datetime already in the prompt (or `get_current_time`); never guess.\n"
        "- Scheduled actions belong to THIS chat only and are invisible to other chats.\n"
        f"- When an action fires, the next message starts with `{trigger}`. Treat it as a"
        " commitment you made earlier: carry it out now and answer in the chat. Do not schedule"
        " another action unless the trigger explicitly asks for one.\n"
        "- If a scheduled prompt asks you to stay silent when a condition is not met, reply"
        " exactly `NO_ANSWER` and nothing else when it is not met. The bot then posts no"
        " message, though the action still counts as run.\n"
        f"- At most {max_actions_per_chat} open actions per chat. Delete reminders that are"
        " finished so the quota stays free."
    )
