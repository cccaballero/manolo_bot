"""Composition layer for the scheduler subsystem.

:class:`SchedulerService` wires a store (what is due) to an engine (when to
check) and owns the operational semantics neither seam should duplicate:

- validating user requests (prompt, one-shot vs cron, limits);
- turning claimed actions into handler calls;
- reporting success/failure back to the store without letting handler
  exceptions escape a sweep or a manual fire.

The handler is injected so this module stays independent of Telegram and of
the agent implementations. Wiring code (``main.py``) supplies a handler that
starts an agent turn with :data:`~manolo_bot.scheduler.base.SCHEDULED_TRIGGER_PREFIX`
plus the action prompt.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from langchain_core.tools import BaseTool

from manolo_bot.scheduler.base import (
    MAX_PROMPT_CHARS,
    BaseSchedulerEngine,
    BaseSchedulerStore,
    ScheduledAction,
    SchedulerDispatch,
    SchedulerError,
    SchedulerValidationError,
    new_action_id,
    normalize_scope,
)
from manolo_bot.scheduler.cron import next_cron_run, validate_cron, validate_timezone

logger = logging.getLogger(__name__)

#: Callable invoked for every claimed action. It runs the agent turn and must
#: raise on failure so the service can record it.
ActionHandler = Callable[[ScheduledAction], Awaitable[None]]

#: Injectable clock used for "now" (tests replace it with a fixed datetime).
Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SchedulerService(SchedulerDispatch):
    """Validate, persist and dispatch chat-scoped scheduled actions."""

    def __init__(
        self,
        store: BaseSchedulerStore,
        engine: BaseSchedulerEngine,
        *,
        handler: ActionHandler | None = None,
        batch_size: int = 10,
        clock: Clock | None = None,
    ) -> None:
        """Compose a store and an engine.

        :param store: Persistence backend. Its ``default_timezone`` is the zone
            used for naive ``run_at`` values and cron evaluation.
        :param engine: Trigger backend (asyncio loop, manual/worker, ...).
        :param handler: Coroutine called with each claimed action. When omitted
            the service can still be used for validation/listing, but sweeps and
            fires raise :class:`~manolo_bot.scheduler.base.SchedulerError`.
        :param batch_size: Maximum actions claimed by one sweep.
        :param clock: Returns the current aware datetime; defaults to UTC now.
        """
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        self.store = store
        self.engine = engine
        self.handler = handler
        self.batch_size = int(batch_size)
        self.clock: Clock = clock or _utcnow
        self._started = False

    @property
    def timezone_name(self) -> str:
        """Timezone used to interpret naive datetimes and cron expressions."""
        return self.store.default_timezone

    async def start(self) -> None:
        """Validate configuration, load the store and start the engine.

        Idempotent: calling it again after a successful start is a no-op.
        """
        if self._started:
            return
        try:
            validate_timezone(self.store.default_timezone)
        except ValueError as exc:
            raise SchedulerValidationError(str(exc)) from exc
        await self.store.build_or_load()
        await self.engine.start(self)
        await self._notify_engine()
        self._started = True

    async def stop(self) -> None:
        """Stop the engine. Idempotent."""
        await self.engine.stop()
        self._started = False

    async def sweep(self) -> int:
        """Claim one batch of due actions, dispatch them and return how many ran."""
        handler = self._require_handler()
        now = self.clock()
        due = await self.store.claim_due(now, self.batch_size)
        for action in due:
            await self._dispatch(action, handler)
        if due:
            await self._notify_engine()
        return len(due)

    async def fire(self, chat_id: str, action_id: str) -> bool:
        """Claim and dispatch one specific action; ``False`` when it is not due."""
        handler = self._require_handler()
        scope = normalize_scope(chat_id)
        now = self.clock()
        action = await self.store.claim_one(scope, action_id, now)
        if action is None:
            return False
        await self._dispatch(action, handler)
        await self._notify_engine()
        return True

    async def add_action(
        self,
        chat_id: str,
        prompt: str,
        *,
        run_at: str | datetime | None = None,
        cron: str | None = None,
        max_runs: int | None = None,
    ) -> ScheduledAction:
        """Validate and persist a new action, returning an independent copy.

        Exactly one of ``run_at`` (absolute ISO-8601, or a datetime) and
        ``cron`` (5-field expression) must be provided. Naive datetimes are
        interpreted in the store timezone, then normalized to aware UTC.
        """
        scope = normalize_scope(chat_id)
        now = self.clock()
        cleaned_prompt = (prompt or "").strip()
        if not cleaned_prompt:
            raise SchedulerValidationError("The scheduled prompt must not be empty.")
        if len(cleaned_prompt) > MAX_PROMPT_CHARS:
            raise SchedulerValidationError(
                f"The scheduled prompt is too long ({len(cleaned_prompt)} characters); "
                f"the maximum is {MAX_PROMPT_CHARS} characters."
            )
        if (run_at is None) == (cron is None):
            raise SchedulerValidationError(
                "Provide exactly one of run_at (an ISO-8601 datetime, "
                "e.g. 2026-10-21T09:00:00+02:00) or cron "
                "(a 5-field cron expression, e.g. 0 9 * * 1-5)."
            )
        max_runs = self._validate_max_runs(max_runs)

        if run_at is not None:
            expression: str | None = None
            fire_at = self._parse_run_at(run_at)
            if fire_at <= now:
                raise SchedulerValidationError(
                    f"run_at must be strictly in the future; the current time is {now.isoformat()}."
                )
        else:
            try:
                expression = validate_cron(cron)
                fire_at = next_cron_run(expression, now, self.store.default_timezone)
            except ValueError as exc:
                raise SchedulerValidationError(str(exc)) from exc

        action = ScheduledAction(
            id=new_action_id(),
            chat_id=scope,
            prompt=cleaned_prompt,
            run_at=fire_at,
            created_at=now,
            cron=expression,
            max_runs=max_runs,
        )
        created = await self.store.add(action)
        await self._notify_engine()
        return created.clone()

    async def list_actions(self, chat_id: str) -> list[ScheduledAction]:
        """Return the scope's actions (copies, sorted by next run)."""
        return [action.clone() for action in await self.store.list(normalize_scope(chat_id))]

    async def delete_action(self, chat_id: str, action_id: str) -> bool:
        """Delete one action scoped to ``chat_id``; notify the engine on success."""
        deleted = await self.store.delete(normalize_scope(chat_id), action_id)
        if deleted:
            await self._notify_engine()
        return deleted

    async def clear_actions(self, chat_id: str) -> int:
        """Delete every action in one scope, returning how many existed.

        Used by the ``/flushschedules`` command. Counts before clearing because
        the store contract's ``clear()`` is fire-and-forget.
        """
        scope = normalize_scope(chat_id)
        actions = await self.store.list(scope)
        if not actions:
            return 0
        await self.store.clear(scope)
        await self._notify_engine()
        return len(actions)

    def as_tools(self) -> list[BaseTool]:
        """Build the agent tools bound to this service (import kept lazy)."""
        from manolo_bot.scheduler.tools import build_scheduler_tools

        return build_scheduler_tools(self)

    def as_instructions(self) -> str:
        """Return the generic scheduling prompt block.

        Override in a subclass to customize the wording; return an empty
        string to suppress the block.
        """
        from manolo_bot.scheduler.prompting import build_scheduler_instructions

        return build_scheduler_instructions(self.timezone_name, self.store.max_actions_per_chat)

    async def active_instructions(self, chat_id: str) -> str:
        """Return the live per-scope schedule snapshot for the system prompt.

        Override in a subclass to customize or suppress it.
        """
        from manolo_bot.scheduler.prompting import build_active_schedules_block

        return build_active_schedules_block(await self.list_actions(normalize_scope(chat_id)))

    def _require_handler(self) -> ActionHandler:
        if self.handler is None:
            raise SchedulerError("SchedulerService has no handler configured")
        return self.handler

    async def _notify_engine(self) -> None:
        """Hand the engine a fresh snapshot only when it consumes one."""
        if not self.engine.wants_action_snapshots:
            return
        await self.engine.on_actions_changed(await self.store.list_all())

    async def _dispatch(self, action: ScheduledAction, handler: ActionHandler) -> None:
        """Run the handler once, recording delivery or failure. Never raises.

        Bookkeeping failures are logged instead of propagating: aborting here
        would strand the remaining already-claimed actions in ``firing``.
        """
        try:
            await handler(action)
        except Exception as exc:
            logger.exception("Scheduled action %s for chat %s failed", action.id, action.chat_id)
            try:
                await self.store.record_failure(action, f"{type(exc).__name__}: {exc}", self.clock())
            except Exception:
                logger.exception("Failed to record failure of scheduled action %s", action.id)
        else:
            try:
                await self.store.mark_delivered(action, self.clock())
            except Exception:
                logger.exception("Failed to record delivery of scheduled action %s", action.id)

    def _parse_run_at(self, value: str | datetime) -> datetime:
        """Parse and normalize a one-shot datetime to aware UTC."""
        if isinstance(value, datetime):
            moment = value
        elif isinstance(value, str):
            try:
                moment = datetime.fromisoformat(value.strip())
            except ValueError as exc:
                raise SchedulerValidationError(
                    "run_at must be an ISO-8601 datetime, e.g. 2026-10-21T09:00:00+02:00."
                ) from exc
        else:
            raise SchedulerValidationError("run_at must be an ISO-8601 datetime string or a datetime object.")
        if moment.tzinfo is None:
            try:
                zone = validate_timezone(self.store.default_timezone)
            except ValueError as exc:
                raise SchedulerValidationError(str(exc)) from exc
            moment = moment.replace(tzinfo=zone)
        return moment.astimezone(UTC)

    @staticmethod
    def _validate_max_runs(max_runs: int | None) -> int | None:
        if max_runs is None:
            return None
        if isinstance(max_runs, bool) or not isinstance(max_runs, int):
            raise SchedulerValidationError("max_runs must be a positive integer (>= 1).")
        if max_runs < 1:
            raise SchedulerValidationError("max_runs must be a positive integer (>= 1).")
        return max_runs


__all__ = ["ActionHandler", "Clock", "SchedulerService"]
