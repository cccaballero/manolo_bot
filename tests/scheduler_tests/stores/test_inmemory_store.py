"""Tests for the in-memory scheduler store."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from manolo_bot.scheduler.base import (
    STATUS_ACTIVE,
    STATUS_FAILED,
    STATUS_FIRING,
    ScheduledAction,
    SchedulerLimitError,
)
from manolo_bot.scheduler.stores.inmemory_store import InMemorySchedulerStore

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _action(
    chat_id: str = "1",
    *,
    action_id: str,
    run_at: datetime | None = None,
    cron: str | None = None,
    max_runs: int | None = None,
    status: str = STATUS_ACTIVE,
    prompt: str = "Ping",
) -> ScheduledAction:
    """Build a deterministic action for store tests."""
    return ScheduledAction(
        id=action_id,
        chat_id=chat_id,
        prompt=prompt,
        run_at=run_at if run_at is not None else NOW,
        created_at=NOW,
        cron=cron,
        max_runs=max_runs,
        status=status,
    )


class TestInMemorySchedulerStore(unittest.IsolatedAsyncioTestCase):
    """CRUD, quota, claim and recovery behaviour of the in-memory store."""

    async def asyncSetUp(self) -> None:
        self.store = InMemorySchedulerStore()
        await self.store.build_or_load()

    async def test_crud_and_sorting(self) -> None:
        late = _action("1", action_id="late", run_at=NOW + timedelta(hours=2))
        early = _action("1", action_id="early", run_at=NOW)
        middle = _action("1", action_id="middle", run_at=NOW + timedelta(hours=1))
        for action in (late, early, middle):
            await self.store.add(action)

        listed = await self.store.list("1")
        self.assertEqual([action.id for action in listed], ["early", "middle", "late"])
        # An integer scope addresses the same records as its string form.
        self.assertEqual([action.id for action in await self.store.list(1)], ["early", "middle", "late"])
        self.assertTrue(await self.store.delete("1", "middle"))
        self.assertFalse(await self.store.delete("1", "middle"))
        self.assertEqual([action.id for action in await self.store.list("1")], ["early", "late"])

    async def test_add_stores_a_clone(self) -> None:
        action = _action("1", action_id="a")
        returned = await self.store.add(action)
        returned.prompt = "mutated externally"
        action.prompt = "mutated input"
        listed = await self.store.list("1")
        self.assertEqual(listed[0].prompt, "Ping")
        self.assertIsNot(listed[0], action)

    async def test_chat_isolation_and_same_id_across_chats(self) -> None:
        await self.store.add(_action("1", action_id="shared", run_at=NOW + timedelta(hours=1), prompt="chat one"))
        await self.store.add(_action("2", action_id="shared", run_at=NOW, prompt="chat two"))

        first = await self.store.list("1")
        second = await self.store.list("2")
        self.assertEqual([action.prompt for action in first], ["chat one"])
        self.assertEqual([action.prompt for action in second], ["chat two"])

        self.assertTrue(await self.store.delete("1", "shared"))
        self.assertEqual(await self.store.list("1"), [])
        self.assertEqual([action.prompt for action in await self.store.list("2")], ["chat two"])
        self.assertFalse(await self.store.delete("2", "missing"))

    async def test_build_or_load_is_idempotent(self) -> None:
        await self.store.add(_action("1", action_id="a"))
        await self.store.build_or_load()
        await self.store.build_or_load()
        self.assertEqual([action.id for action in await self.store.list("1")], ["a"])

    async def test_quota_counts_only_open_actions(self) -> None:
        self.store = InMemorySchedulerStore(max_actions_per_chat=2)
        await self.store.build_or_load()
        first = await self.store.add(_action("1", action_id="first"))
        await self.store.add(_action("1", action_id="second"))
        with self.assertRaises(SchedulerLimitError):
            await self.store.add(_action("1", action_id="third"))

        await self.store.record_failure(first, "boom", NOW)
        await self.store.add(_action("1", action_id="third"))
        with self.assertRaises(SchedulerLimitError):
            await self.store.add(_action("1", action_id="fourth"))

    async def test_one_shot_delivery_frees_quota_slot(self) -> None:
        self.store = InMemorySchedulerStore(max_actions_per_chat=1)
        await self.store.build_or_load()
        await self.store.add(_action("1", action_id="only", run_at=NOW - timedelta(minutes=1)))

        claimed = await self.store.claim_one("1", "only", NOW)
        self.assertIsNotNone(claimed)
        await self.store.mark_delivered(claimed, NOW)
        self.assertEqual(await self.store.list("1"), [])

        await self.store.add(_action("1", action_id="replacement"))
        self.assertEqual([action.id for action in await self.store.list("1")], ["replacement"])

    async def test_one_shot_claim_then_delivery_removes_record(self) -> None:
        await self.store.add(_action("1", action_id="once", run_at=NOW - timedelta(minutes=1)))

        claimed = await self.store.claim_one("1", "once", NOW)
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed.status, STATUS_FIRING)
        self.assertEqual(claimed.run_count, 1)
        self.assertEqual([action.status for action in await self.store.list("1")], [STATUS_FIRING])

        await self.store.mark_delivered(claimed, NOW)
        self.assertEqual(await self.store.list("1"), [])

    async def test_one_shot_failure_marks_failed(self) -> None:
        await self.store.add(_action("1", action_id="once", run_at=NOW - timedelta(minutes=1)))
        claimed = await self.store.claim_one("1", "once", NOW)
        assert claimed is not None

        await self.store.record_failure(claimed, "provider down", NOW)
        failed = await self.store.list("1")
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].status, STATUS_FAILED)
        self.assertEqual(failed[0].attempts, 1)
        self.assertEqual(failed[0].last_error, "provider down")

    async def test_cron_claim_advances_and_delivery_reactivates(self) -> None:
        await self.store.add(_action("1", action_id="cron", run_at=NOW - timedelta(minutes=1), cron="*/5 * * * *"))

        claimed = await self.store.claim_one("1", "cron", NOW)
        assert claimed is not None
        self.assertEqual(claimed.status, STATUS_FIRING)
        self.assertEqual(claimed.run_count, 1)
        self.assertGreater(claimed.run_at, NOW)

        await self.store.mark_delivered(claimed, NOW)
        active = await self.store.list("1")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].status, STATUS_ACTIVE)
        self.assertEqual(active[0].run_count, 1)
        self.assertEqual(active[0].run_at, claimed.run_at)
        self.assertEqual(active[0].attempts, 0)
        self.assertIsNone(active[0].last_error)

    async def test_cron_max_runs_removes_after_final_delivery(self) -> None:
        await self.store.add(
            _action(
                "1",
                action_id="cron",
                run_at=NOW - timedelta(minutes=1),
                cron="*/5 * * * *",
                max_runs=2,
            )
        )

        first = await self.store.claim_one("1", "cron", NOW)
        assert first is not None
        await self.store.mark_delivered(first, NOW)

        second = await self.store.claim_one("1", "cron", first.run_at)
        assert second is not None
        self.assertEqual(second.run_count, 2)
        await self.store.mark_delivered(second, NOW)
        self.assertEqual(await self.store.list("1"), [])

    async def test_catchup_grace_skips_overdue_actions(self) -> None:
        store = InMemorySchedulerStore(catchup_grace_seconds=60)
        await store.build_or_load()
        await store.add(_action("1", action_id="once", run_at=NOW - timedelta(minutes=2)))
        await store.add(_action("2", action_id="cron", run_at=NOW - timedelta(minutes=2), cron="*/5 * * * *"))

        self.assertEqual(await store.claim_due(NOW), [])
        one_shot = (await store.list("1"))[0]
        self.assertEqual(one_shot.status, STATUS_FAILED)
        self.assertIn("Missed run skipped", one_shot.last_error or "")
        recurring = (await store.list("2"))[0]
        self.assertEqual(recurring.status, STATUS_ACTIVE)
        self.assertGreater(recurring.run_at, NOW)
        self.assertIn("Missed run skipped", recurring.last_error or "")

    async def test_catchup_grace_zero_fires_overdue_action(self) -> None:
        await self.store.add(_action("1", action_id="once", run_at=NOW - timedelta(days=1)))
        claimed = await self.store.claim_due(NOW)
        self.assertEqual([action.id for action in claimed], ["once"])
        self.assertEqual(claimed[0].status, STATUS_FIRING)

    async def test_claim_due_respects_limit_and_sorting(self) -> None:
        await self.store.add(_action("2", action_id="z", run_at=NOW - timedelta(minutes=3)))
        await self.store.add(_action("1", action_id="b", run_at=NOW - timedelta(minutes=2)))
        await self.store.add(_action("1", action_id="a", run_at=NOW - timedelta(minutes=2)))
        await self.store.add(_action("1", action_id="future", run_at=NOW + timedelta(minutes=1)))

        claimed = await self.store.claim_due(NOW, limit=3)
        self.assertEqual([action.id for action in claimed], ["z", "a", "b"])
        self.assertTrue(all(action.status == STATUS_FIRING for action in claimed))
        self.assertEqual(await self.store.claim_due(NOW, limit=0), [])
        self.assertEqual(
            [action.id for action in await self.store.list("1") if action.status == STATUS_ACTIVE],
            ["future"],
        )

    async def test_claim_one_rejects_future_other_chat_and_unknown(self) -> None:
        await self.store.add(_action("1", action_id="future", run_at=NOW + timedelta(minutes=5)))
        await self.store.add(_action("1", action_id="due", run_at=NOW - timedelta(minutes=1)))
        await self.store.add(_action("2", action_id="other", run_at=NOW - timedelta(minutes=1)))

        self.assertIsNone(await self.store.claim_one("1", "future", NOW))
        self.assertIsNone(await self.store.claim_one("1", "other", NOW))
        self.assertIsNone(await self.store.claim_one("1", "missing", NOW))

        claimed = await self.store.claim_one("1", "due", NOW)
        self.assertIsNotNone(claimed)
        self.assertIsNone(await self.store.claim_one("1", "due", NOW))
        # The other chat's identical-in-time action was untouched.
        other = await self.store.list("2")
        self.assertEqual([action.status for action in other], [STATUS_ACTIVE])

    async def test_crash_recovery_rewrites_firing_records(self) -> None:
        await self.store.add(_action("1", action_id="once", status=STATUS_FIRING))
        await self.store.add(_action("2", action_id="cron", status=STATUS_FIRING, cron="*/5 * * * *"))

        await self.store.build_or_load()
        once = (await self.store.list("1"))[0]
        cron = (await self.store.list("2"))[0]
        self.assertEqual(once.status, STATUS_FAILED)
        self.assertEqual(cron.status, STATUS_ACTIVE)
        self.assertIn("interrupted", once.last_error or "")
        self.assertIn("interrupted", cron.last_error or "")

    async def test_clear_and_list_all_uses_lexicographic_scope_order(self) -> None:
        await self.store.add(_action("7", action_id="b", run_at=NOW))
        await self.store.add(_action("42", action_id="c", run_at=NOW + timedelta(hours=1)))
        await self.store.add(_action("42", action_id="a", run_at=NOW))

        all_actions = await self.store.list_all()
        # Scopes sort lexicographically ("42" before "7"), then by run time.
        self.assertEqual([action.id for action in all_actions], ["a", "c", "b"])

        await self.store.clear("42")
        self.assertEqual(await self.store.list("42"), [])
        self.assertEqual([action.id for action in await self.store.list_all()], ["b"])
        await self.store.clear("999")

    async def test_list_all_orders_equal_times_lexicographically_by_scope(self) -> None:
        await self.store.add(_action("7", action_id="seven", run_at=NOW))
        await self.store.add(_action("42", action_id="forty-two", run_at=NOW))

        self.assertEqual([action.id for action in await self.store.list_all()], ["forty-two", "seven"])

    async def test_string_scopes_are_isolated(self) -> None:
        await self.store.add(_action("C012345", action_id="a", run_at=NOW + timedelta(hours=1)))
        await self.store.add(_action("C999", action_id="b", run_at=NOW))
        await self.store.add(_action("C012345", action_id="c", run_at=NOW))

        self.assertEqual([action.id for action in await self.store.list("C012345")], ["c", "a"])
        self.assertEqual([action.id for action in await self.store.list_all()], ["c", "a", "b"])
        self.assertIsNone(await self.store.claim_one("C012345", "b", NOW))
        self.assertIsNone(await self.store.claim_one("C999", "a", NOW))
        self.assertTrue(await self.store.delete("C012345", "c"))
        self.assertEqual([action.id for action in await self.store.list("C012345")], ["a"])
        self.assertEqual([action.id for action in await self.store.list("C999")], ["b"])

    async def test_claim_due_orders_equal_times_lexicographically_by_scope(self) -> None:
        await self.store.add(_action("7", action_id="seven", run_at=NOW - timedelta(minutes=1)))
        await self.store.add(_action("42", action_id="forty-two", run_at=NOW - timedelta(minutes=1)))

        claimed = await self.store.claim_due(NOW, limit=2)

        self.assertEqual([action.id for action in claimed], ["forty-two", "seven"])


if __name__ == "__main__":
    unittest.main()
