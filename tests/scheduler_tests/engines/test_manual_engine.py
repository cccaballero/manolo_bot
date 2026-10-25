"""Tests for the manual scheduler engine."""

from __future__ import annotations

import asyncio
import unittest

from manolo_bot.scheduler.engines import ManualSchedulerEngine


class FakeDispatch:
    def __init__(self) -> None:
        self.calls = 0

    async def sweep(self) -> int:
        self.calls += 1
        return 0

    async def fire(self, chat_id: str, action_id: str) -> bool:
        return False


class TestManualSchedulerEngine(unittest.IsolatedAsyncioTestCase):
    async def test_start_records_dispatch_without_triggering(self) -> None:
        engine = ManualSchedulerEngine()
        dispatch = FakeDispatch()

        await engine.start(dispatch)
        await asyncio.sleep(0)

        self.assertIs(engine.dispatch, dispatch)
        self.assertEqual(dispatch.calls, 0)

    async def test_stop_clears_dispatch_and_is_idempotent(self) -> None:
        engine = ManualSchedulerEngine()
        await engine.start(FakeDispatch())

        await engine.stop()
        self.assertIsNone(engine.dispatch)

        await engine.stop()
        self.assertIsNone(engine.dispatch)

    async def test_start_replaces_the_previous_dispatch(self) -> None:
        engine = ManualSchedulerEngine()
        first = FakeDispatch()
        second = FakeDispatch()

        await engine.start(first)
        await engine.start(second)

        self.assertIs(engine.dispatch, second)

    async def test_on_actions_changed_is_a_noop(self) -> None:
        engine = ManualSchedulerEngine()
        self.assertIsNone(await engine.on_actions_changed([]))
