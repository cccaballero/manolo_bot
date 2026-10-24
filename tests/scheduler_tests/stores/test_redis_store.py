"""Tests for the Redis-backed scheduler store (fakeredis, no server needed)."""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta

import fakeredis.aioredis

from manolo_bot.scheduler.base import (
    STATUS_ACTIVE,
    STATUS_FAILED,
    STATUS_FIRING,
    ScheduledAction,
    SchedulerLimitError,
)
from manolo_bot.scheduler.stores import build_scheduler_store
from manolo_bot.scheduler.stores.redis_store import RedisSchedulerStore

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def make_action(
    action_id: str,
    *,
    chat_id: int = 42,
    run_at: datetime = NOW,
    prompt: str = "do the thing",
    cron: str | None = None,
    max_runs: int | None = None,
) -> ScheduledAction:
    return ScheduledAction(
        id=action_id,
        chat_id=chat_id,
        prompt=prompt,
        run_at=run_at,
        created_at=NOW - timedelta(hours=1),
        cron=cron,
        max_runs=max_runs,
    )


class RedisStoreFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.redis = fakeredis.aioredis.FakeRedis()
        self.store = RedisSchedulerStore(bot_uuid="test-bot", redis_client=self.redis)
        await self.store.build_or_load()

    async def asyncTearDown(self) -> None:
        await self.redis.flushall()
        await self.redis.aclose()

    async def seed(self, action: ScheduledAction) -> ScheduledAction:
        await self.store.add(action)
        return action


class TestRedisStoreCRUD(RedisStoreFixture):
    async def test_add_list_and_sort(self) -> None:
        later = make_action("later", run_at=NOW + timedelta(hours=1))
        sooner = make_action("sooner", run_at=NOW + timedelta(minutes=1))
        await self.seed(later)
        await self.seed(sooner)

        self.assertEqual([a.id for a in await self.store.list(42)], ["sooner", "later"])

    async def test_list_returns_clones(self) -> None:
        await self.seed(make_action("a"))

        first = (await self.store.list(42))[0]
        first.prompt = "mutated"

        self.assertEqual((await self.store.list(42))[0].prompt, "do the thing")

    async def test_chat_isolation_and_delete(self) -> None:
        own = await self.seed(make_action("own", chat_id=42))
        foreign = await self.seed(make_action("foreign", chat_id=7))

        self.assertFalse(await self.store.delete(42, foreign.id))
        self.assertEqual([a.id for a in await self.store.list(7)], ["foreign"])
        self.assertTrue(await self.store.delete(42, own.id))
        self.assertEqual(await self.store.list(42), [])

    async def test_quota_is_enforced(self) -> None:
        store = RedisSchedulerStore(
            bot_uuid="quota-bot",
            redis_client=self.redis,
            max_actions_per_chat=2,
        )
        await store.build_or_load()
        await store.add(make_action("a", chat_id=7))
        await store.add(make_action("b", chat_id=7))
        with self.assertRaises(SchedulerLimitError):
            await store.add(make_action("c", chat_id=7))

    async def test_clear_removes_only_one_chat(self) -> None:
        await self.seed(make_action("own", chat_id=42))
        await self.seed(make_action("foreign", chat_id=7))

        await self.store.clear(42)

        self.assertEqual(await self.store.list(42), [])
        self.assertEqual([a.id for a in await self.store.list(7)], ["foreign"])

    async def test_list_all(self) -> None:
        await self.seed(make_action("b", chat_id=7))
        await self.seed(make_action("a", chat_id=42))

        # Sorted by chat_id first: chat 7 before chat 42.
        self.assertEqual([a.id for a in await self.store.list_all()], ["b", "a"])

    async def test_lock_is_released(self) -> None:
        await self.seed(make_action("a"))
        self.assertIsNone(await self.redis.get(self.store.lock_key))


class TestRedisStoreClaims(RedisStoreFixture):
    async def test_one_shot_claim_deliver_and_remove(self) -> None:
        await self.seed(make_action("a", run_at=NOW - timedelta(minutes=1)))

        claimed = await self.store.claim_due(NOW)

        self.assertEqual([a.id for a in claimed], ["a"])
        self.assertEqual(claimed[0].status, STATUS_FIRING)
        self.assertEqual((await self.store.list(42))[0].status, STATUS_FIRING)
        self.assertIsNone(await self.redis.get(self.store.lock_key))

        await self.store.mark_delivered(claimed[0], NOW)
        self.assertEqual(await self.store.list(42), [])

    async def test_cron_claim_advances_and_reactivates(self) -> None:
        await self.seed(make_action("cron", cron="*/15 * * * *", run_at=NOW - timedelta(minutes=1)))

        claimed = await self.store.claim_due(NOW)
        self.assertEqual(claimed[0].run_at, NOW + timedelta(minutes=15))
        self.assertEqual(claimed[0].run_count, 1)

        await self.store.mark_delivered(claimed[0], NOW)
        stored = (await self.store.list(42))[0]
        self.assertEqual(stored.status, STATUS_ACTIVE)
        self.assertEqual(stored.run_count, 1)

    async def test_cron_max_runs_removes_after_final_delivery(self) -> None:
        await self.seed(make_action("cron", cron="*/15 * * * *", run_at=NOW - timedelta(minutes=1), max_runs=1))

        claimed = await self.store.claim_due(NOW)
        await self.store.mark_delivered(claimed[0], NOW)

        self.assertEqual(await self.store.list(42), [])

    async def test_claim_one_rejects_future_unknown_and_foreign(self) -> None:
        future = await self.seed(make_action("future", run_at=NOW + timedelta(hours=1)))
        foreign = await self.seed(make_action("foreign", chat_id=7, run_at=NOW - timedelta(minutes=1)))

        self.assertIsNone(await self.store.claim_one(42, future.id, NOW))
        self.assertIsNone(await self.store.claim_one(42, "missing", NOW))
        self.assertIsNone(await self.store.claim_one(42, foreign.id, NOW))
        self.assertTrue((await self.store.claim_one(7, foreign.id, NOW)) is not None)

    async def test_catchup_grace_fails_stale_one_shot(self) -> None:
        store = RedisSchedulerStore(
            bot_uuid="grace-bot",
            redis_client=self.redis,
            catchup_grace_seconds=60,
        )
        await store.build_or_load()
        await store.add(make_action("stale", run_at=NOW - timedelta(hours=1)))

        self.assertEqual(await store.claim_due(NOW), [])
        self.assertEqual((await store.list(42))[0].status, STATUS_FAILED)

    async def test_failure_marks_one_shot_failed(self) -> None:
        await self.seed(make_action("a", run_at=NOW - timedelta(minutes=1)))
        claimed = (await self.store.claim_due(NOW))[0]

        await self.store.record_failure(claimed, "boom", NOW)

        stored = (await self.store.list(42))[0]
        self.assertEqual(stored.status, STATUS_FAILED)
        self.assertEqual(stored.attempts, 1)
        self.assertEqual(stored.last_error, "boom")

    async def test_crash_recovery_recovers_firing_records(self) -> None:
        await self.seed(make_action("a", run_at=NOW - timedelta(minutes=1)))
        await self.store.claim_due(NOW)

        recovered = RedisSchedulerStore(bot_uuid="test-bot", redis_client=self.redis)
        await recovered.build_or_load()

        self.assertEqual((await recovered.list(42))[0].status, STATUS_FAILED)


class TestRedisStorePersistence(RedisStoreFixture):
    async def test_snapshot_is_shared_between_instances(self) -> None:
        await self.seed(make_action("a"))
        other = RedisSchedulerStore(bot_uuid="test-bot", redis_client=self.redis)

        await other.build_or_load()

        self.assertEqual([a.id for a in await other.list(42)], ["a"])

    async def test_corrupt_snapshot_raises_and_is_not_overwritten(self) -> None:
        raw = b"{not json"
        await self.redis.set(self.store.snapshot_key, raw)

        with self.assertRaises(ValueError):
            await self.store.build_or_load()
        with self.assertRaises(ValueError):
            await self.store.list(42)

        self.assertEqual(await self.redis.get(self.store.snapshot_key), raw)

    async def test_unknown_version_raises_and_is_not_overwritten(self) -> None:
        payload = json.dumps({"version": 99, "actions": []}).encode()
        await self.redis.set(self.store.snapshot_key, payload)

        with self.assertRaises(ValueError):
            await self.store.build_or_load()

        self.assertEqual(await self.redis.get(self.store.snapshot_key), payload)

    async def test_two_instances_claim_a_due_action_only_once(self) -> None:
        await self.seed(make_action("a", run_at=NOW - timedelta(minutes=1)))
        first = RedisSchedulerStore(bot_uuid="test-bot", redis_client=self.redis)
        second = RedisSchedulerStore(bot_uuid="test-bot", redis_client=self.redis)

        results = await asyncio.gather(
            first.claim_due(NOW),
            second.claim_due(NOW),
        )

        claimed = [action for batch in results for action in batch]
        self.assertEqual([action.id for action in claimed], ["a"])


class TestRedisStoreFactory(unittest.TestCase):
    def test_factory_routes_redis(self) -> None:
        store = build_scheduler_store("redis", bot_uuid="factory-bot")
        self.assertIsInstance(store, RedisSchedulerStore)
        self.assertEqual(store.bot_uuid, "factory-bot")


if __name__ == "__main__":
    unittest.main()
