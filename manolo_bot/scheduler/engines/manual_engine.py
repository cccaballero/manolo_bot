"""Manual engine: no background work, callers drive sweeps explicitly.

Useful for tests, CLI/worker entrypoints, or deployments where an external
scheduler (system cron, Celery beat) invokes ``dispatch.sweep()``/``fire()``.
"""

from __future__ import annotations

from collections.abc import Sequence

from manolo_bot.scheduler.base import BaseSchedulerEngine, ScheduledAction, SchedulerDispatch


class ManualSchedulerEngine(BaseSchedulerEngine):
    """Record the dispatch target and never trigger anything by itself."""

    def __init__(self) -> None:
        self._dispatch: SchedulerDispatch | None = None

    @property
    def dispatch(self) -> SchedulerDispatch | None:
        """The dispatch currently registered by ``start``, if any."""
        return self._dispatch

    async def start(self, dispatch: SchedulerDispatch) -> None:
        """Remember the dispatch target. Idempotent (last one wins)."""
        self._dispatch = dispatch

    async def stop(self) -> None:
        """Forget the dispatch target. Idempotent."""
        self._dispatch = None

    async def on_actions_changed(self, actions: Sequence[ScheduledAction]) -> None:
        """No-op: nothing is registered per action."""


__all__ = ["ManualSchedulerEngine"]
