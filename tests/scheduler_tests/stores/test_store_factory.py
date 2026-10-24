"""Tests for the scheduler store factory."""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from manolo_bot.scheduler.base import STATUS_ACTIVE, ScheduledAction
from manolo_bot.scheduler.stores.factory import build_scheduler_store
from manolo_bot.scheduler.stores.filesystem_store import FilesystemSchedulerStore
from manolo_bot.scheduler.stores.inmemory_store import InMemorySchedulerStore


class TestSchedulerStoreFactory(unittest.IsolatedAsyncioTestCase):
    """Name routing and option forwarding."""

    def test_routes_in_memory(self) -> None:
        store = build_scheduler_store("in_memory")
        self.assertIsInstance(store, InMemorySchedulerStore)

    def test_routes_local_fs_with_namespaced_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = build_scheduler_store("local_fs", bot_uuid="my-bot", store_path=tmp)
            self.assertIsInstance(store, FilesystemSchedulerStore)
            assert isinstance(store, FilesystemSchedulerStore)
            self.assertEqual(store.path, Path(tmp) / "my-bot" / "schedules.json")
            self.assertEqual(store.store_path, Path(tmp))

    def test_normalizes_name_whitespace_and_case(self) -> None:
        self.assertIsInstance(build_scheduler_store("  In_Memory  "), InMemorySchedulerStore)
        with tempfile.TemporaryDirectory() as tmp:
            store = build_scheduler_store(" LOCAL_FS ", store_path=tmp)
            self.assertIsInstance(store, FilesystemSchedulerStore)

    def test_forwards_options(self) -> None:
        store = build_scheduler_store(
            "in_memory",
            bot_uuid="bot-7",
            max_actions_per_chat=3,
            max_consecutive_failures=2,
            catchup_grace_seconds=90,
            default_timezone="Europe/Madrid",
        )
        self.assertEqual(store.bot_uuid, "bot-7")
        self.assertEqual(store.max_actions_per_chat, 3)
        self.assertEqual(store.max_consecutive_failures, 2)
        self.assertEqual(store.catchup_grace_seconds, 90)
        self.assertEqual(store.default_timezone, "Europe/Madrid")

    def test_rejects_unknown_store(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Unsupported scheduler store: 'redis'"):
            build_scheduler_store("redis")
        with self.assertRaises(ValueError):
            build_scheduler_store("")

    async def test_built_store_is_usable(self) -> None:
        store = build_scheduler_store("in_memory")
        await store.build_or_load()
        action = ScheduledAction(
            id="a",
            chat_id=1,
            prompt="Ping",
            run_at=datetime(2026, 1, 1, tzinfo=UTC),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        stored = await store.add(action)
        self.assertEqual(stored.status, STATUS_ACTIVE)
        self.assertEqual([item.id for item in await store.list(1)], ["a"])


if __name__ == "__main__":
    unittest.main()
