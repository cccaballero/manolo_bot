"""Core scheduler abstractions.

The scheduler is split into two independent seams that are composed by
:class:`manolo_bot.scheduler.service.SchedulerService`:

- :class:`BaseSchedulerStore` decides **what** is due (persistence and state
  transitions). Every operation is scoped to an opaque chat token.
- :class:`BaseSchedulerEngine` decides **when** to check (triggering only). It
  never reads or mutates actions.

This module also owns the :class:`ScheduledAction` model and the state-machine
helpers shared by every store implementation, so backends cannot drift apart
on claim/deliver/failure semantics.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from manolo_bot.scheduler.cron import next_cron_run

#: Canonical tool names exposed to agents.
SCHEDULE_ACTION_TOOL_NAME = "schedule_action"
LIST_SCHEDULED_ACTIONS_TOOL_NAME = "list_scheduled_actions"
DELETE_SCHEDULED_ACTION_TOOL_NAME = "delete_scheduled_action"

#: Prefix persisted before the prompt of an agent turn started by the
#: scheduler. The system prompt explains its meaning to the model.
SCHEDULED_TRIGGER_PREFIX = "[SCHEDULED ACTION TRIGGERED] "

#: Action lifecycle. ``firing`` is transient: it is written before the agent
#: turn runs, so a crash skips that run instead of duplicating it.
STATUS_ACTIVE = "active"
STATUS_FIRING = "firing"
STATUS_FAILED = "failed"

#: Hard cap for a scheduled prompt, kept small because it is stored and later
#: replayed as a message in the conversation.
MAX_PROMPT_CHARS = 4000

_OPEN_STATUSES = frozenset({STATUS_ACTIVE, STATUS_FIRING})


class SchedulerError(Exception):
    """Base class for scheduler errors surfaced to tools as text."""


class SchedulerValidationError(SchedulerError):
    """The requested action is malformed (bad cron, past date, empty prompt)."""


class SchedulerLimitError(SchedulerError):
    """The chat already holds the maximum number of open actions."""


def ensure_utc(value: datetime) -> datetime:
    """Return an aware UTC datetime, treating naive input as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def normalize_scope(value: str | int) -> str:
    """Return the scheduler scope token for ``value``.

    Scopes are opaque strings: a numeric chat id normalizes to its decimal
    representation so ``42`` and ``"42"`` address the same records. ``None``
    and blank values are rejected because they cannot identify a scope.
    """
    if value is None:
        raise ValueError("Scheduler scope must not be empty.")
    scope = str(value)
    if not scope.strip():
        raise ValueError("Scheduler scope must not be empty.")
    return scope


@dataclass
class ScheduledAction:
    """An action scoped to an opaque chat token, carried out at ``run_at``.

    ``chat_id`` is an opaque scope token (a Telegram chat id today, a string
    like ``"C012345"`` for other consumers). ``run_at`` is always stored as an
    aware UTC datetime. For recurring actions it is the next occurrence and is
    advanced at claim time; for one-shot actions it is the only occurrence.
    """

    id: str
    chat_id: str
    prompt: str
    run_at: datetime
    created_at: datetime
    cron: str | None = None
    max_runs: int | None = None
    run_count: int = 0
    attempts: int = 0
    status: str = STATUS_ACTIVE
    last_error: str | None = None

    def __post_init__(self) -> None:
        self.chat_id = normalize_scope(self.chat_id)
        self.run_at = ensure_utc(self.run_at)
        self.created_at = ensure_utc(self.created_at)
        if self.max_runs is not None:
            self.max_runs = int(self.max_runs)

    @property
    def is_recurring(self) -> bool:
        return self.cron is not None

    def clone(self) -> ScheduledAction:
        """Return an independent copy (stores never hand out internal objects)."""
        return replace(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "prompt": self.prompt,
            "run_at": self.run_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "cron": self.cron,
            "max_runs": self.max_runs,
            "run_count": self.run_count,
            "attempts": self.attempts,
            "status": self.status,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledAction:
        return cls(
            id=str(data["id"]),
            chat_id=data["chat_id"],
            prompt=str(data["prompt"]),
            run_at=ensure_utc(datetime.fromisoformat(str(data["run_at"]))),
            created_at=ensure_utc(datetime.fromisoformat(str(data["created_at"]))),
            cron=data.get("cron"),
            max_runs=data.get("max_runs"),
            run_count=int(data.get("run_count", 0)),
            attempts=int(data.get("attempts", 0)),
            status=str(data.get("status", STATUS_ACTIVE)),
            last_error=data.get("last_error"),
        )


def new_action_id() -> str:
    """Return a fresh action id (uuid4 hex)."""
    return uuid.uuid4().hex


def is_open(action: ScheduledAction) -> bool:
    """True while the action still holds a slot in its chat's quota."""
    return action.status in _OPEN_STATUSES


def ensure_chat_limit(actions: Sequence[ScheduledAction], max_actions_per_chat: int) -> None:
    """Raise :class:`SchedulerLimitError` when the chat quota is exhausted.

    ``actions`` are the chat's current records; completed one-shot actions are
    removed by the store, and failed records do not hold a slot.
    """
    open_count = sum(1 for action in actions if is_open(action))
    if open_count >= max_actions_per_chat:
        raise SchedulerLimitError(
            f"This chat already has {open_count} scheduled actions "
            f"(limit {max_actions_per_chat}). Delete one before scheduling another."
        )


def apply_claim(action: ScheduledAction, now: datetime, timezone_name: str) -> ScheduledAction:
    """Mark an action as being delivered.

    Recurring actions advance to their next occurrence strictly after ``now``
    before the agent runs, so a crash or a slow run cannot replay the same
    occurrence. ``run_count`` is incremented at claim time (failed runs count
    against ``max_runs``).
    """
    claimed = replace(action, status=STATUS_FIRING, run_count=action.run_count + 1)
    if action.cron:
        claimed.run_at = next_cron_run(action.cron, now, timezone_name)
    return claimed


def apply_catchup_grace(
    action: ScheduledAction,
    now: datetime,
    timezone_name: str,
    grace_seconds: int,
) -> ScheduledAction | None:
    """Skip an action that is too old to fire, returning its replacement.

    Returns ``None`` when the action is within the grace window (or grace is
    disabled) and should be claimed normally. Recurring actions move to their
    next occurrence and stay active; one-shot actions are marked failed.
    """
    if grace_seconds <= 0:
        return None
    overdue = (now - action.run_at).total_seconds()
    if overdue <= grace_seconds:
        return None
    reason = f"Missed run skipped (overdue by {int(overdue)}s)"
    if action.cron:
        return replace(
            action,
            run_at=next_cron_run(action.cron, now, timezone_name),
            status=STATUS_ACTIVE,
            last_error=reason,
        )
    return replace(action, status=STATUS_FAILED, last_error=reason)


def apply_delivered(action: ScheduledAction) -> ScheduledAction | None:
    """Return the record after a successful delivery, or ``None`` to remove it.

    One-shot actions are removed; recurring actions return to active with
    their already-advanced ``run_at``. A recurring action that reached
    ``max_runs`` is removed.
    """
    if action.cron is None:
        return None
    if action.max_runs is not None and action.run_count >= action.max_runs:
        return None
    return replace(action, status=STATUS_ACTIVE, attempts=0, last_error=None)


def apply_failure(
    action: ScheduledAction,
    error: str,
    max_consecutive_failures: int,
) -> ScheduledAction:
    """Return the record after a failed delivery.

    One-shot actions fail permanently. Recurring actions stay active until
    they reach ``max_consecutive_failures`` (or ``max_runs``), then fail.
    """
    attempts = action.attempts + 1
    if action.cron is None:
        return replace(action, status=STATUS_FAILED, attempts=attempts, last_error=error)
    exhausted = attempts >= max_consecutive_failures
    over_budget = action.max_runs is not None and action.run_count >= action.max_runs
    if exhausted or over_budget:
        return replace(action, status=STATUS_FAILED, attempts=attempts, last_error=error)
    return replace(action, status=STATUS_ACTIVE, attempts=attempts, last_error=error)


def apply_interrupted(action: ScheduledAction) -> ScheduledAction:
    """Recover a record left in ``firing`` by a process crash.

    Recurring actions resume at their next occurrence; one-shot actions are
    marked failed so the interruption is visible instead of silently lost.
    """
    if action.status != STATUS_FIRING:
        return action
    reason = "Delivery interrupted by a restart; this run was skipped"
    if action.cron:
        return replace(action, status=STATUS_ACTIVE, last_error=reason)
    return replace(action, status=STATUS_FAILED, last_error=reason)


class BaseSchedulerStore(ABC):
    """Persistence contract for scheduled actions.

    Every method that reads or mutates user data takes an opaque ``chat_id``
    scope token; cross-scope access is impossible through the public API.
    Implementations normalize incoming values with :func:`normalize_scope` at
    the boundary, so ``42`` and ``"42"`` address the same records.
    Implementations must:

    - persist the claim before returning from ``claim_due``/``claim_one``;
    - make claims atomic across concurrent callers (process-local lock for
      in-memory; an OS file lock for the filesystem store);
    - use the state-machine helpers in this module instead of reimplementing
      claim/deliver/failure semantics.
    """

    def __init__(
        self,
        bot_uuid: str = "default",
        *,
        max_actions_per_chat: int = 20,
        max_consecutive_failures: int = 5,
        catchup_grace_seconds: int = 0,
        default_timezone: str = "UTC",
    ) -> None:
        self.bot_uuid = bot_uuid
        self.max_actions_per_chat = int(max_actions_per_chat)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.catchup_grace_seconds = int(catchup_grace_seconds)
        self.default_timezone = default_timezone

    @abstractmethod
    async def build_or_load(self) -> None:
        """Prepare the store and recover records left in ``firing``."""

    @abstractmethod
    async def add(self, action: ScheduledAction) -> ScheduledAction:
        """Persist a new action, enforcing the per-chat quota."""

    @abstractmethod
    async def list(self, chat_id: str) -> list[ScheduledAction]:
        """Return the scope's active, firing and failed actions sorted by run time."""

    @abstractmethod
    async def delete(self, chat_id: str, action_id: str) -> bool:
        """Delete an action from one scope, returning whether it existed."""

    @abstractmethod
    async def claim_due(self, now: datetime, limit: int = 10) -> list[ScheduledAction]:
        """Atomically claim up to ``limit`` due actions across all chats."""

    @abstractmethod
    async def claim_one(self, chat_id: str, action_id: str, now: datetime) -> ScheduledAction | None:
        """Atomically claim one specific action, or return None if not due."""

    @abstractmethod
    async def mark_delivered(self, action: ScheduledAction, now: datetime) -> None:
        """Finalize a successful delivery (remove one-shots, reactivate cron)."""

    @abstractmethod
    async def record_failure(self, action: ScheduledAction, error: str, now: datetime) -> None:
        """Record a failed delivery and apply the failure policy."""

    @abstractmethod
    async def clear(self, chat_id: str) -> None:
        """Remove every scheduled action belonging to one scope."""

    @abstractmethod
    async def list_all(self) -> list[ScheduledAction]:
        """Return every action across chats (engine-facing, never a tool)."""


class SchedulerDispatch(Protocol):
    """The surface an engine uses to ask the service to do work."""

    async def sweep(self) -> int:
        """Claim and dispatch one batch of due actions; return how many ran."""

    async def fire(self, chat_id: str, action_id: str) -> bool:
        """Claim and dispatch one specific action; False when it is not due."""


class BaseSchedulerEngine(ABC):
    """Trigger contract.

    Engines only decide *when* work should be checked. They never read or
    mutate actions directly. In-process engines call ``dispatch.sweep()``;
    out-of-process engines (Celery, system cron) invoke a stable worker
    entrypoint that calls ``dispatch.sweep()`` or ``dispatch.fire()``.
    """

    #: True when ``on_actions_changed`` consumes the full action snapshot
    #: (job-registration engines). Sweep-based engines leave it False so the
    #: service does not pay a store read on every sweep/fire/add/delete.
    wants_action_snapshots: bool = False

    @abstractmethod
    async def start(self, dispatch: SchedulerDispatch) -> None:
        """Start triggering. Must be idempotent."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop triggering and release resources. Must be idempotent."""

    async def on_actions_changed(self, actions: Sequence[ScheduledAction]) -> None:
        """Optional hook for engines that register per-action jobs.

        Called after actions are added, deleted or finalized. Sweep-based
        engines can ignore it.
        """


__all__ = [
    "BASE_SCHEDULER_TOOL_NAMES",
    "BaseSchedulerEngine",
    "BaseSchedulerStore",
    "DELETE_SCHEDULED_ACTION_TOOL_NAME",
    "LIST_SCHEDULED_ACTIONS_TOOL_NAME",
    "MAX_PROMPT_CHARS",
    "SCHEDULE_ACTION_TOOL_NAME",
    "SCHEDULED_TRIGGER_PREFIX",
    "STATUS_ACTIVE",
    "STATUS_FAILED",
    "STATUS_FIRING",
    "ScheduledAction",
    "SchedulerDispatch",
    "SchedulerError",
    "SchedulerLimitError",
    "SchedulerValidationError",
    "apply_catchup_grace",
    "apply_claim",
    "apply_delivered",
    "apply_failure",
    "apply_interrupted",
    "ensure_chat_limit",
    "ensure_utc",
    "is_open",
    "new_action_id",
    "normalize_scope",
]

#: Convenience tuple for clash detection / wiring tests.
BASE_SCHEDULER_TOOL_NAMES = (
    SCHEDULE_ACTION_TOOL_NAME,
    LIST_SCHEDULED_ACTIONS_TOOL_NAME,
    DELETE_SCHEDULED_ACTION_TOOL_NAME,
)
