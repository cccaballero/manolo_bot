"""Cron expression helpers.

``croniter`` is used only as a parser/evaluator. Scheduling, persistence and
triggering stay in the scheduler store/engine seams.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import CroniterBadCronError, CroniterBadDateError, croniter

CRON_FIELD_COUNT = 5


def validate_timezone(name: str) -> ZoneInfo:
    """Return a ``ZoneInfo`` or raise ``ValueError`` with a useful message."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown timezone {name!r}") from exc


def validate_cron(expression: str) -> str:
    """Validate a 5-field cron expression and return its normalized form.

    :raises ValueError: When the expression is empty, has the wrong number of
        fields or cannot be parsed.
    """
    normalized = " ".join((expression or "").split())
    if len(normalized.split(" ")) != CRON_FIELD_COUNT:
        raise ValueError("Cron expressions must have exactly 5 fields (minute hour day-of-month month day-of-week)")
    try:
        croniter(normalized)
    except (CroniterBadCronError, ValueError) as exc:
        raise ValueError(f"Invalid cron expression {expression!r}: {exc}") from exc
    return normalized


def next_cron_run(expression: str, after: datetime, timezone_name: str = "UTC") -> datetime:
    """Return the first occurrence strictly after ``after`` as aware UTC.

    Cron expressions are evaluated in ``timezone_name`` (default UTC), which
    keeps DST transitions correct. ``after`` may be naive (treated as UTC).
    """
    timezone = validate_timezone(timezone_name)
    if after.tzinfo is None:
        after = after.replace(tzinfo=UTC)
    local_after = after.astimezone(timezone)
    try:
        iterator = croniter(expression, local_after)
        next_local = iterator.get_next(datetime)
    except (CroniterBadCronError, CroniterBadDateError, ValueError) as exc:
        raise ValueError(f"Cron expression {expression!r} has no future occurrence: {exc}") from exc
    if next_local.tzinfo is None:
        next_local = next_local.replace(tzinfo=timezone)
    return next_local.astimezone(UTC)


__all__ = ["CRON_FIELD_COUNT", "next_cron_run", "validate_cron", "validate_timezone"]
