"""Periodic sweep engine driven by :func:`asyncio.sleep`.

The loop sleeps until a fixed deadline rather than for a fixed duration, so
time spent sweeping does not accumulate drift. The sleep and monotonic clock
are injected to keep tests deterministic and free of real timers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence

from manolo_bot.scheduler.base import BaseSchedulerEngine, ScheduledAction, SchedulerDispatch

logger = logging.getLogger(__name__)

SleepFn = Callable[[float], Awaitable[None]]
MonotonicFn = Callable[[], float]


class AsyncioSchedulerEngine(BaseSchedulerEngine):
    """Run ``dispatch.sweep()`` every ``interval_seconds`` in a background task."""

    def __init__(
        self,
        interval_seconds: float = 5.0,
        *,
        sleep: SleepFn = asyncio.sleep,
        monotonic: MonotonicFn = time.monotonic,
    ) -> None:
        """Create the engine.

        :param interval_seconds: Target delay between sweeps; must be > 0.
        :param sleep: Awaitable sleep (injectable for tests).
        :param monotonic: Monotonic clock used to compute the next deadline.
        """
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        self.interval_seconds = float(interval_seconds)
        self._sleep = sleep
        self._monotonic = monotonic
        self._dispatch: SchedulerDispatch | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        """Whether the background sweep task is alive."""
        return self._task is not None and not self._task.done()

    async def start(self, dispatch: SchedulerDispatch) -> None:
        """Start the sweep loop. Idempotent while the task is alive."""
        if self.is_running:
            return
        self._dispatch = dispatch
        self._task = asyncio.create_task(self._run(), name="scheduler-engine")

    async def stop(self) -> None:
        """Cancel the sweep loop and wait for it to finish. Idempotent."""
        task, self._task = self._task, None
        self._dispatch = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def on_actions_changed(self, actions: Sequence[ScheduledAction]) -> None:
        """No-op: sweep-based engines do not register per-action jobs."""

    def _next_delay(self, deadline: float, now: float) -> tuple[float, float]:
        """Advance the cadence, resetting it when a sweep overran the interval.

        A sweep runs handlers inline, so an agent turn can take much longer than
        ``interval_seconds``. Without the clamp, the deadline falls permanently
        behind real time and the loop degenerates into a 0-second busy spin.
        """
        deadline += self.interval_seconds
        if deadline < now:
            deadline = now
        return deadline, deadline - now

    async def _run(self) -> None:
        """Sweep forever, surviving handler/store errors until cancelled."""
        dispatch = self._dispatch
        if dispatch is None:  # pragma: no cover - guarded by start()
            return
        deadline = self._monotonic()
        while True:
            try:
                await dispatch.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler sweep failed; retrying in %ss", self.interval_seconds)
            deadline, delay = self._next_delay(deadline, self._monotonic())
            await self._sleep(delay)


__all__ = ["AsyncioSchedulerEngine"]
