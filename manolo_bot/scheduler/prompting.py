"""Scheduler prompting helpers: system-instruction snippet for scheduled actions.

Kept separate from the agent classes (mirroring :mod:`manolo_bot.rag.prompting`)
so the wording can be reused and tested without constructing an agent.
"""

from __future__ import annotations

from collections.abc import Sequence

from manolo_bot.scheduler.base import (
    DELETE_SCHEDULED_ACTION_TOOL_NAME,
    LIST_SCHEDULED_ACTIONS_TOOL_NAME,
    SCHEDULE_ACTION_TOOL_NAME,
    SCHEDULED_TRIGGER_PREFIX,
    STATUS_ACTIVE,
    ScheduledAction,
    is_open,
)

#: Keep the per-chat snapshot compact: at most this many rows and this many
#: prompt characters per row. The list tool remains available for full detail.
ACTIVE_SCHEDULES_LIMIT = 10
ACTIVE_PROMPT_PREVIEW_CHARS = 120


def build_active_schedules_block(actions: Sequence[ScheduledAction]) -> str:
    """Render a compact snapshot of a chat's actions for the system prompt.

    Returns an empty string when the chat has no actions, so chats without
    schedules pay neither tokens nor prompt noise. The snapshot is what keeps
    the agent aware of schedules after ``/flushcontext`` wipes the messages
    that originally created them.
    """
    if not actions:
        return ""
    lines = ["\n\n## Active scheduled actions in this chat"]
    # Failed actions keep their past run_at; sort open actions first so they
    # cannot crowd upcoming schedules out of the compact row budget.
    ordered = sorted(actions, key=lambda action: (not is_open(action), action.run_at))
    for action in ordered[:ACTIVE_SCHEDULES_LIMIT]:
        schedule = f"cron={action.cron!r}" if action.cron else "one-shot"
        prompt = action.prompt
        if len(prompt) > ACTIVE_PROMPT_PREVIEW_CHARS:
            prompt = prompt[: ACTIVE_PROMPT_PREVIEW_CHARS - 1] + "…"
        status = "" if action.status == STATUS_ACTIVE else f" status={action.status}"
        lines.append(f"- id={action.id} next={action.run_at.isoformat()} {schedule}{status} prompt={prompt!r}")
    remaining = len(actions) - ACTIVE_SCHEDULES_LIMIT
    if remaining > 0:
        lines.append(f"- …and {remaining} more; use `{LIST_SCHEDULED_ACTIONS_TOOL_NAME}`.")
    lines.append(
        f"Use `{LIST_SCHEDULED_ACTIONS_TOOL_NAME}` for full prompts and "
        f"`{DELETE_SCHEDULED_ACTION_TOOL_NAME}` to cancel one."
    )
    return "\n".join(lines)


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
        f"- When an action fires, the next message starts with `{trigger}` (followed by its id)."
        " Treat it as a commitment you made earlier: carry it out now and answer in the chat. Do"
        " not schedule another action unless the trigger explicitly asks for one. A cron prompt"
        " describes one execution; the recurrence is already handled by the cron expression.\n"
        "- Scheduled actions are stored outside the conversation and survive `/flushcontext`."
        f" If the user asks to stop, review, or cancel recurring messages, call"
        f" `{LIST_SCHEDULED_ACTIONS_TOOL_NAME}` and then `{DELETE_SCHEDULED_ACTION_TOOL_NAME}`"
        " with a real id. A fired action's trigger includes its id, so you can cancel that exact"
        " action directly. Never claim an action was scheduled or deleted unless the tool"
        " returned success.\n"
        "- If a scheduled prompt asks you to stay silent when a condition is not met, reply"
        " exactly `NO_ANSWER` and nothing else when it is not met. The bot then posts no"
        " message, though the action still counts as run.\n"
        f"- At most {max_actions_per_chat} open actions per chat. Delete reminders that are"
        " finished so the quota stays free."
    )
