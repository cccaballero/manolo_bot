"""Chat-scoped scheduling for agent modes.

Two independent seams compose here:

- stores decide *what* is due and persist it;
- engines decide *when* to check.

See :mod:`manolo_bot.scheduler.base` for the contracts and
:class:`manolo_bot.scheduler.service.SchedulerService` for the composition.
"""

from manolo_bot.scheduler.base import (
    BASE_SCHEDULER_TOOL_NAMES,
    DELETE_SCHEDULED_ACTION_TOOL_NAME,
    LIST_SCHEDULED_ACTIONS_TOOL_NAME,
    MAX_PROMPT_CHARS,
    SCHEDULE_ACTION_TOOL_NAME,
    SCHEDULED_TRIGGER_PREFIX,
    STATUS_ACTIVE,
    STATUS_FAILED,
    STATUS_FIRING,
    BaseSchedulerEngine,
    BaseSchedulerStore,
    ScheduledAction,
    SchedulerDispatch,
    SchedulerError,
    SchedulerLimitError,
    SchedulerValidationError,
)

__all__ = [
    "BASE_SCHEDULER_TOOL_NAMES",
    "DELETE_SCHEDULED_ACTION_TOOL_NAME",
    "LIST_SCHEDULED_ACTIONS_TOOL_NAME",
    "MAX_PROMPT_CHARS",
    "SCHEDULE_ACTION_TOOL_NAME",
    "SCHEDULED_TRIGGER_PREFIX",
    "STATUS_ACTIVE",
    "STATUS_FAILED",
    "STATUS_FIRING",
    "BaseSchedulerEngine",
    "BaseSchedulerStore",
    "ScheduledAction",
    "SchedulerDispatch",
    "SchedulerError",
    "SchedulerLimitError",
    "SchedulerValidationError",
]
