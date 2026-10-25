"""Tests for the filesystem scheduler store."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from manolo_bot.scheduler.base import (
    STATUS_ACTIVE,
    STATUS_FAILED,
    STATUS_FIRING,
    ScheduledAction,
    SchedulerLimitError,
)
from manolo_bot.scheduler.stores.filesystem_store import FilesystemSchedulerStore, schedules_path

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


class TestFilesystemSchedulerStore(unittest.IsolatedAsyncioTestCase):
    """Persistence, locking and state-machine behaviour of the filesystem store."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "store"
        self.store = FilesystemSchedulerStore("test-bot", self.root)
        await self.store.build_or_load()

    async def _new_store(self, **kwargs) -> FilesystemSchedulerStore:
        store = FilesystemSchedulerStore("test-bot", self.root, **kwargs)
        await store.build_or_load()
        return store

    async def test_build_creates_namespace_lock_and_no_snapshot(self) -> None:
        self.assertTrue(self.store.path.parent.is_dir())
        self.assertTrue(self.store.lock_path.is_file())
        self.assertFalse(self.store.path.exists())
        self.assertEqual(await self.store.list_all(), [])

    async def test_add_writes_versioned_snapshot(self) -> None:
        await self.store.add(_action("1", action_id="a"))
        payload = json.loads(self.store.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 1)
        self.assertEqual([entry["id"] for entry in payload["actions"]], ["a"])
        self.assertEqual(payload["actions"][0]["chat_id"], "1")
        self.assertIsInstance(payload["actions"][0]["chat_id"], str)

    async def test_state_survives_a_new_store_instance(self) -> None:
        await self.store.add(_action("1", action_id="a", run_at=NOW + timedelta(hours=1)))
        await self.store.add(_action("2", action_id="b"))

        fresh = await self._new_store()
        self.assertEqual([action.id for action in await fresh.list_all()], ["a", "b"])
        self.assertEqual([action.id for action in await fresh.list("1")], ["a"])
        self.assertEqual([action.id for action in await fresh.list("2")], ["b"])

    async def test_crud_and_sorting_persist(self) -> None:
        await self.store.add(_action("1", action_id="late", run_at=NOW + timedelta(hours=2)))
        await self.store.add(_action("1", action_id="early", run_at=NOW))

        fresh = await self._new_store()
        listed = await fresh.list("1")
        self.assertEqual([action.id for action in listed], ["early", "late"])
        # An integer scope addresses the same records as its string form.
        self.assertEqual([action.id for action in await fresh.list(1)], ["early", "late"])

        self.assertTrue(await self.store.delete("1", "late"))
        self.assertFalse(await self.store.delete("1", "late"))
        self.assertEqual([action.id for action in await (await self._new_store()).list("1")], ["early"])

    async def test_chat_isolation_and_same_id_across_chats(self) -> None:
        await self.store.add(_action("1", action_id="shared", prompt="chat one"))
        await self.store.add(_action("2", action_id="shared", prompt="chat two"))

        self.assertEqual([action.prompt for action in await self.store.list("1")], ["chat one"])
        self.assertEqual([action.prompt for action in await self.store.list("2")], ["chat two"])
        self.assertFalse(await self.store.delete("1", "missing"))

        self.assertTrue(await self.store.delete("1", "shared"))
        self.assertEqual(await self.store.list("1"), [])
        self.assertEqual([action.prompt for action in await self.store.list("2")], ["chat two"])

    async def test_quota_ignores_failed_records(self) -> None:
        store = await self._new_store(max_actions_per_chat=1)
        first = await store.add(_action("1", action_id="first"))
        with self.assertRaises(SchedulerLimitError):
            await store.add(_action("1", action_id="second"))

        await store.record_failure(first, "boom", NOW)
        await store.add(_action("1", action_id="second"))
        self.assertEqual([action.id for action in await store.list("1")], ["first", "second"])

    async def test_corrupt_snapshot_loads_empty_and_is_repairable(self) -> None:
        self.store.path.parent.mkdir(parents=True, exist_ok=True)
        self.store.path.write_text("{not valid json", encoding="utf-8")

        fresh = await self._new_store()
        self.assertEqual(await fresh.list_all(), [])
        await fresh.add(_action("1", action_id="repaired"))
        self.assertEqual([action.id for action in await (await self._new_store()).list_all()], ["repaired"])

    async def test_crash_recovery_on_load_persists_recovered_records(self) -> None:
        self.store.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "actions": [
                _action("1", action_id="once", status=STATUS_FIRING).to_dict(),
                _action("2", action_id="cron", status=STATUS_FIRING, cron="*/5 * * * *").to_dict(),
            ],
        }
        self.store.path.write_text(json.dumps(payload), encoding="utf-8")

        fresh = await self._new_store()
        once = (await fresh.list("1"))[0]
        cron = (await fresh.list("2"))[0]
        self.assertEqual(once.status, STATUS_FAILED)
        self.assertEqual(cron.status, STATUS_ACTIVE)
        self.assertIn("interrupted", once.last_error or "")
        self.assertIn("interrupted", cron.last_error or "")

        stored = json.loads(self.store.path.read_text(encoding="utf-8"))
        statuses = {entry["id"]: entry["status"] for entry in stored["actions"]}
        self.assertEqual(statuses, {"once": STATUS_FAILED, "cron": STATUS_ACTIVE})

    async def test_one_shot_claim_is_persisted_before_return(self) -> None:
        await self.store.add(_action("1", action_id="once", run_at=NOW - timedelta(minutes=1)))
        claimed = await self.store.claim_one("1", "once", NOW)
        assert claimed is not None

        # The claim hits disk before claim_one returns; a fresh instance would
        # then treat the firing record as crash-interrupted on build_or_load.
        payload = json.loads(self.store.path.read_text(encoding="utf-8"))
        self.assertEqual([entry["status"] for entry in payload["actions"]], [STATUS_FIRING])

        await self.store.mark_delivered(claimed, NOW)
        self.assertEqual(await (await self._new_store()).list("1"), [])

    async def test_cron_claim_and_delivery_persist(self) -> None:
        await self.store.add(_action("1", action_id="cron", run_at=NOW - timedelta(minutes=1), cron="*/5 * * * *"))
        claimed = await self.store.claim_one("1", "cron", NOW)
        assert claimed is not None
        self.assertGreater(claimed.run_at, NOW)
        self.assertEqual(claimed.run_count, 1)

        payload = json.loads(self.store.path.read_text(encoding="utf-8"))
        persisted = payload["actions"][0]
        self.assertEqual(persisted["status"], STATUS_FIRING)
        self.assertEqual(persisted["run_count"], 1)
        self.assertEqual(datetime.fromisoformat(persisted["run_at"]), claimed.run_at)

        await self.store.mark_delivered(claimed, NOW)
        reactivated = (await (await self._new_store()).list("1"))[0]
        self.assertEqual(reactivated.status, STATUS_ACTIVE)
        self.assertEqual(reactivated.run_count, 1)
        self.assertEqual(reactivated.attempts, 0)

    async def test_failure_persists(self) -> None:
        await self.store.add(_action("1", action_id="once", run_at=NOW - timedelta(minutes=1)))
        claimed = await self.store.claim_one("1", "once", NOW)
        assert claimed is not None

        fresh = await self._new_store()
        await fresh.record_failure(claimed, "provider down", NOW)
        failed = (await (await self._new_store()).list("1"))[0]
        self.assertEqual(failed.status, STATUS_FAILED)
        self.assertEqual(failed.attempts, 1)
        self.assertEqual(failed.last_error, "provider down")

    async def test_claim_due_respects_limit_and_sorting(self) -> None:
        await self.store.add(_action("2", action_id="z", run_at=NOW - timedelta(minutes=3)))
        await self.store.add(_action("1", action_id="b", run_at=NOW - timedelta(minutes=2)))
        await self.store.add(_action("1", action_id="a", run_at=NOW - timedelta(minutes=2)))
        await self.store.add(_action("1", action_id="future", run_at=NOW + timedelta(minutes=1)))

        claimed = await self.store.claim_due(NOW, limit=2)
        self.assertEqual([action.id for action in claimed], ["z", "a"])
        self.assertEqual(await self.store.claim_due(NOW, limit=0), [])

        payload = json.loads(self.store.path.read_text(encoding="utf-8"))
        statuses = {entry["id"]: entry["status"] for entry in payload["actions"]}
        self.assertEqual(statuses["z"], STATUS_FIRING)
        self.assertEqual(statuses["a"], STATUS_FIRING)
        self.assertEqual(statuses["b"], STATUS_ACTIVE)
        self.assertEqual(statuses["future"], STATUS_ACTIVE)

    async def test_catchup_grace_skips_overdue_actions(self) -> None:
        store = await self._new_store(catchup_grace_seconds=60)
        await store.add(_action("1", action_id="once", run_at=NOW - timedelta(minutes=2)))
        await store.add(_action("2", action_id="cron", run_at=NOW - timedelta(minutes=2), cron="*/5 * * * *"))

        self.assertEqual(await store.claim_due(NOW), [])
        fresh = await self._new_store(catchup_grace_seconds=60)
        self.assertEqual((await fresh.list("1"))[0].status, STATUS_FAILED)
        recurring = (await fresh.list("2"))[0]
        self.assertEqual(recurring.status, STATUS_ACTIVE)
        self.assertGreater(recurring.run_at, NOW)

    async def test_cron_max_runs_removes_record(self) -> None:
        await self.store.add(
            _action(
                "1",
                action_id="cron",
                run_at=NOW - timedelta(minutes=1),
                cron="*/5 * * * *",
                max_runs=1,
            )
        )
        claimed = await self.store.claim_one("1", "cron", NOW)
        assert claimed is not None
        await self.store.mark_delivered(claimed, NOW)
        self.assertEqual(await (await self._new_store()).list_all(), [])

    async def test_clear_only_affects_one_chat(self) -> None:
        await self.store.add(_action("1", action_id="a"))
        await self.store.add(_action("2", action_id="b"))
        await self.store.clear("1")
        self.assertEqual([action.id for action in await (await self._new_store()).list_all()], ["b"])
        await self.store.clear("999")

    async def test_list_all_sorted_across_scopes_lexicographically(self) -> None:
        await self.store.add(_action("42", action_id="b", run_at=NOW))
        await self.store.add(_action("7", action_id="c", run_at=NOW + timedelta(hours=1)))
        await self.store.add(_action("7", action_id="a", run_at=NOW))
        # Scopes sort lexicographically ("42" before "7"), then by run time.
        self.assertEqual(
            [action.id for action in await self.store.list_all()],
            ["b", "a", "c"],
        )

    async def test_list_all_orders_equal_times_lexicographically_by_scope(self) -> None:
        await self.store.add(_action("7", action_id="seven", run_at=NOW))
        await self.store.add(_action("42", action_id="forty-two", run_at=NOW))

        self.assertEqual(
            [action.id for action in await self.store.list_all()],
            ["forty-two", "seven"],
        )

    async def test_string_scopes_persist_and_stay_isolated(self) -> None:
        await self.store.add(_action("C012345", action_id="a"))
        await self.store.add(_action("C999", action_id="b"))

        fresh = await self._new_store()
        self.assertEqual([action.id for action in await fresh.list("C012345")], ["a"])
        self.assertEqual([action.id for action in await fresh.list("C999")], ["b"])
        self.assertEqual([action.id for action in await fresh.list_all()], ["a", "b"])
        self.assertIsNone(await fresh.claim_one("C012345", "b", NOW))
        self.assertTrue(await fresh.delete("C012345", "a"))
        self.assertEqual([action.id for action in await fresh.list_all()], ["b"])

    async def test_claim_due_orders_equal_times_lexicographically_by_scope(self) -> None:
        await self.store.add(_action("7", action_id="seven", run_at=NOW - timedelta(minutes=1)))
        await self.store.add(_action("42", action_id="forty-two", run_at=NOW - timedelta(minutes=1)))

        claimed = await self.store.claim_due(NOW)

        self.assertEqual([action.id for action in claimed], ["forty-two", "seven"])

    async def test_no_temporary_files_left_behind(self) -> None:
        await self.store.add(_action("1", action_id="a"))
        await self.store.delete("1", "a")
        leftovers = [path for path in self.store.path.parent.iterdir() if path.suffix == ".tmp"]
        self.assertEqual(leftovers, [])

    async def test_unknown_delivery_and_failure_are_no_ops(self) -> None:
        ghost = _action("1", action_id="ghost")
        await self.store.mark_delivered(ghost, NOW)
        await self.store.record_failure(ghost, "boom", NOW)
        self.assertEqual(await self.store.list_all(), [])

    def test_schedules_path_layout(self) -> None:
        self.assertEqual(
            schedules_path("/tmp/store", "bot"),
            Path("/tmp/store") / "bot" / "schedules.json",
        )


if __name__ == "__main__":
    unittest.main()
