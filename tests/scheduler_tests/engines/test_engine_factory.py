"""Tests for scheduler engine selection."""

from __future__ import annotations

import unittest

from manolo_bot.scheduler.engines import (
    AsyncioSchedulerEngine,
    ManualSchedulerEngine,
    build_scheduler_engine,
)


class TestEngineFactory(unittest.TestCase):
    def test_builds_asyncio_engine_with_interval(self) -> None:
        engine = build_scheduler_engine("asyncio", interval_seconds=3.5)
        self.assertIsInstance(engine, AsyncioSchedulerEngine)
        self.assertEqual(engine.interval_seconds, 3.5)

    def test_builds_manual_engine(self) -> None:
        engine = build_scheduler_engine("manual")
        self.assertIsInstance(engine, ManualSchedulerEngine)

    def test_name_is_normalized(self) -> None:
        self.assertIsInstance(build_scheduler_engine("  ASYNCIO  "), AsyncioSchedulerEngine)
        self.assertIsInstance(build_scheduler_engine("Manual"), ManualSchedulerEngine)

    def test_unknown_name_raises_with_the_name(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_scheduler_engine("celery")
        self.assertIn("celery", str(ctx.exception))

    def test_invalid_interval_propagates(self) -> None:
        with self.assertRaises(ValueError):
            build_scheduler_engine("asyncio", interval_seconds=0)
