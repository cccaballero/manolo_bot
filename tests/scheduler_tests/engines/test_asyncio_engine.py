"""Tests for the asyncio sweep engine.

Real timers are never used: ``sleep`` and ``monotonic`` are injected, and the
fake sleep blocks on an :class:`asyncio.Event` instead of sleeping.
"""

from __future__ import annotations

import asyncio
import unittest

from manolo_bot.scheduler.engines import AsyncioSchedulerEngine


class FakeDispatch:
    """Dispatch double counting sweeps and optionally raising."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    async def sweep(self) -> int:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return 0

    async def fire(self, chat_id: int, action_id: str) -> bool:
        return False


class SleepProbe:
    """Fake sleep: records delays, signals entry and blocks until released."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)
        self.entered.set()
        await self.release.wait()


class TestAsyncioSchedulerEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.dispatch = FakeDispatch()
        self.probe = SleepProbe()
        self.engine = AsyncioSchedulerEngine(interval_seconds=2.0, sleep=self.probe, monotonic=lambda: 0.0)

    async def asyncTearDown(self) -> None:
        await self.engine.stop()

    async def test_start_runs_first_sweep_and_uses_injected_sleep(self) -> None:
        await self.engine.start(self.dispatch)
        await asyncio.wait_for(self.probe.entered.wait(), timeout=1.0)

        self.assertGreaterEqual(self.dispatch.calls, 1)
        self.assertEqual(self.probe.calls[0], 2.0)
        self.assertTrue(self.engine.is_running)

    async def test_task_is_named(self) -> None:
        await self.engine.start(self.dispatch)
        await asyncio.wait_for(self.probe.entered.wait(), timeout=1.0)
        self.assertEqual(self.engine._task.get_name(), "scheduler-engine")

    async def test_start_twice_is_idempotent(self) -> None:
        await self.engine.start(self.dispatch)
        await asyncio.wait_for(self.probe.entered.wait(), timeout=1.0)

        await self.engine.start(self.dispatch)
        for _ in range(5):
            await asyncio.sleep(0)

        self.assertEqual(self.dispatch.calls, 1)

    async def test_stop_cancels_the_loop_and_is_idempotent(self) -> None:
        await self.engine.start(self.dispatch)
        await asyncio.wait_for(self.probe.entered.wait(), timeout=1.0)

        await self.engine.stop()
        self.assertFalse(self.engine.is_running)

        calls_after_stop = self.dispatch.calls
        await self.engine.stop()
        await asyncio.sleep(0)
        self.assertEqual(self.dispatch.calls, calls_after_stop)

    async def test_stop_before_start_is_a_noop(self) -> None:
        await self.engine.stop()
        self.assertFalse(self.engine.is_running)

    async def test_dispatch_error_is_logged_and_loop_survives(self) -> None:
        self.dispatch.error = RuntimeError("boom")

        with self.assertLogs("manolo_bot.scheduler.engines.asyncio_engine", level="ERROR"):
            await self.engine.start(self.dispatch)
            await asyncio.wait_for(self.probe.entered.wait(), timeout=1.0)

        self.assertEqual(self.dispatch.calls, 1)
        self.assertTrue(self.engine.is_running)

    async def test_deadline_uses_injected_monotonic_clock(self) -> None:
        times = iter([100.0, 100.5])
        probe = SleepProbe()
        engine = AsyncioSchedulerEngine(interval_seconds=5.0, sleep=probe, monotonic=lambda: next(times))
        self.addAsyncCleanup(engine.stop)

        await engine.start(FakeDispatch())
        await asyncio.wait_for(probe.entered.wait(), timeout=1.0)

        # deadline = 100.0 + 5.0, clock says 100.5 -> sleep 4.5 (drift-free).
        self.assertEqual(probe.calls[0], 4.5)

    async def test_start_after_stop_creates_a_new_loop(self) -> None:
        await self.engine.start(self.dispatch)
        await asyncio.wait_for(self.probe.entered.wait(), timeout=1.0)
        await self.engine.stop()
        self.assertEqual(self.dispatch.calls, 1)

        await self.engine.start(self.dispatch)
        for _ in range(100):
            if self.dispatch.calls >= 2:
                break
            await asyncio.sleep(0)

        self.assertEqual(self.dispatch.calls, 2)

    async def test_on_actions_changed_is_a_noop(self) -> None:
        self.assertIsNone(await self.engine.on_actions_changed([]))

    def test_non_positive_interval_is_rejected(self) -> None:
        for interval in (0, -1, -0.5):
            with self.subTest(interval=interval):
                with self.assertRaises(ValueError):
                    AsyncioSchedulerEngine(interval_seconds=interval)

    def test_next_delay_keeps_cadence_without_overrun(self) -> None:
        engine = AsyncioSchedulerEngine(interval_seconds=5.0)
        deadline, delay = engine._next_delay(100.0, 101.0)
        self.assertEqual(deadline, 105.0)
        self.assertEqual(delay, 4.0)

    def test_next_delay_resets_after_an_overrun(self) -> None:
        engine = AsyncioSchedulerEngine(interval_seconds=5.0)

        deadline, delay = engine._next_delay(0.0, 100.0)
        self.assertEqual(deadline, 100.0)
        self.assertEqual(delay, 0.0)

        # The next tick starts a fresh cadence instead of staying behind.
        deadline, delay = engine._next_delay(deadline, 100.0)
        self.assertEqual(deadline, 105.0)
        self.assertEqual(delay, 5.0)
