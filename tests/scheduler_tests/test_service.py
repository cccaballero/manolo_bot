"""Tests for SchedulerService composition and its in-memory fake store.

``FakeStore`` implements the store contract in memory using the shared state
helpers from :mod:`manolo_bot.scheduler.base`, so these tests exercise the
service without depending on the concrete stores lane.
"""

from __future__ import annotations

import itertools
import unittest
from datetime import UTC, datetime, timedelta

from manolo_bot.scheduler.base import (
    STATUS_ACTIVE,
    STATUS_FAILED,
    STATUS_FIRING,
    BaseSchedulerEngine,
    BaseSchedulerStore,
    ScheduledAction,
    SchedulerDispatch,
    SchedulerError,
    SchedulerLimitError,
    SchedulerValidationError,
    apply_claim,
    apply_delivered,
    apply_failure,
    ensure_chat_limit,
    is_open,
)
from manolo_bot.scheduler.service import SchedulerService

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
_DEFAULT_HANDLER = object()


class FakeStore(BaseSchedulerStore):
    """In-memory store implementing the full contract with the base helpers."""

    def __init__(
        self,
        *,
        timezone_name: str = "UTC",
        max_actions_per_chat: int = 20,
        max_consecutive_failures: int = 5,
    ) -> None:
        super().__init__(
            max_actions_per_chat=max_actions_per_chat,
            max_consecutive_failures=max_consecutive_failures,
            default_timezone=timezone_name,
        )
        self.actions: dict[str, ScheduledAction] = {}
        self.build_calls = 0
        self.added: list[ScheduledAction] = []
        self.deleted: list[tuple[int, str]] = []
        self.delivered: list[str] = []
        self.failures: list[tuple[str, str]] = []

    async def build_or_load(self) -> None:
        self.build_calls += 1

    async def add(self, action: ScheduledAction) -> ScheduledAction:
        ensure_chat_limit(await self.list(action.chat_id), self.max_actions_per_chat)
        self.actions[action.id] = action.clone()
        self.added.append(action.clone())
        return action.clone()

    async def list(self, chat_id: int) -> list[ScheduledAction]:
        return sorted(
            (action.clone() for action in self.actions.values() if action.chat_id == chat_id),
            key=lambda action: (action.run_at, action.id),
        )

    async def delete(self, chat_id: int, action_id: str) -> bool:
        self.deleted.append((chat_id, action_id))
        action = self.actions.get(action_id)
        if action is None or action.chat_id != chat_id:
            return False
        del self.actions[action_id]
        return True

    async def claim_due(self, now: datetime, limit: int = 10) -> list[ScheduledAction]:
        due = sorted(
            (action for action in self.actions.values() if is_open(action) and action.run_at <= now),
            key=lambda action: (action.run_at, action.id),
        )[:limit]
        claimed = []
        for action in due:
            snapshot = apply_claim(action, now, self.default_timezone)
            self.actions[action.id] = snapshot
            claimed.append(snapshot.clone())
        return claimed

    async def claim_one(self, chat_id: int, action_id: str, now: datetime) -> ScheduledAction | None:
        action = self.actions.get(action_id)
        if action is None or action.chat_id != chat_id or not is_open(action) or action.run_at > now:
            return None
        snapshot = apply_claim(action, now, self.default_timezone)
        self.actions[action_id] = snapshot
        return snapshot.clone()

    async def mark_delivered(self, action: ScheduledAction, now: datetime) -> None:
        self.delivered.append(action.id)
        replacement = apply_delivered(action)
        if replacement is None:
            self.actions.pop(action.id, None)
        else:
            self.actions[action.id] = replacement

    async def record_failure(self, action: ScheduledAction, error: str, now: datetime) -> None:
        self.failures.append((action.id, error))
        replacement = apply_failure(action, error, self.max_consecutive_failures)
        if replacement is None:
            self.actions.pop(action.id, None)
        else:
            self.actions[action.id] = replacement

    async def clear(self, chat_id: int) -> None:
        for action_id in [key for key, action in self.actions.items() if action.chat_id == chat_id]:
            del self.actions[action_id]

    async def list_all(self) -> list[ScheduledAction]:
        return [action.clone() for action in self.actions.values()]


class RecordingHandler:
    """Async handler that records calls and optionally raises one error."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[ScheduledAction] = []

    async def __call__(self, action: ScheduledAction) -> None:
        self.calls.append(action)
        if self.error is not None:
            raise self.error


class RecordingEngine(BaseSchedulerEngine):
    """Engine double recording lifecycle calls and notification snapshots."""

    wants_action_snapshots = True

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.dispatch: SchedulerDispatch | None = None
        self.notifications: list[list[ScheduledAction]] = []

    async def start(self, dispatch: SchedulerDispatch) -> None:
        self.started += 1
        self.dispatch = dispatch

    async def stop(self) -> None:
        self.stopped += 1
        self.dispatch = None

    async def on_actions_changed(self, actions) -> None:
        self.notifications.append([action.clone() for action in actions])


class ServiceFixture:
    """Shared builders for service tests."""

    def setUp(self) -> None:
        self._ids = itertools.count(1)

    def make_service(
        self,
        *,
        handler: RecordingHandler | None | object = _DEFAULT_HANDLER,
        store: FakeStore | None = None,
        engine: BaseSchedulerEngine | None = None,
        batch_size: int = 10,
    ) -> SchedulerService:
        self.store = store if store is not None else FakeStore()
        self.engine = engine if engine is not None else RecordingEngine()
        if handler is _DEFAULT_HANDLER:
            handler = RecordingHandler()
        self.handler = handler
        self.service = SchedulerService(
            self.store,
            self.engine,
            handler=handler,
            batch_size=batch_size,
            clock=lambda: NOW,
        )
        return self.service

    async def seed(
        self,
        *,
        prompt: str = "do the thing",
        run_at: datetime = NOW,
        chat_id: int = 42,
        cron: str | None = None,
        max_runs: int | None = None,
        status: str = STATUS_ACTIVE,
        last_error: str | None = None,
    ) -> ScheduledAction:
        action = ScheduledAction(
            id=f"action-{next(self._ids)}",
            chat_id=chat_id,
            prompt=prompt,
            run_at=run_at,
            created_at=NOW - timedelta(hours=1),
            cron=cron,
            max_runs=max_runs,
            status=status,
            last_error=last_error,
        )
        await self.store.add(action)
        return action


class TestServiceLifecycle(ServiceFixture, unittest.IsolatedAsyncioTestCase):
    async def test_start_loads_store_starts_engine_and_notifies(self) -> None:
        service = self.make_service()
        await service.start()
        self.assertEqual(self.store.build_calls, 1)
        self.assertEqual(self.engine.started, 1)
        self.assertIs(self.engine.dispatch, service)
        self.assertEqual(self.engine.notifications, [[]])

    async def test_start_is_idempotent(self) -> None:
        service = self.make_service()
        await service.start()
        await service.start()
        self.assertEqual(self.store.build_calls, 1)
        self.assertEqual(self.engine.started, 1)

    async def test_start_rejects_invalid_store_timezone(self) -> None:
        service = self.make_service(store=FakeStore(timezone_name="Mars/Phobos"))
        with self.assertRaises(SchedulerValidationError) as ctx:
            await service.start()
        self.assertIn("Mars/Phobos", str(ctx.exception))
        self.assertEqual(self.store.build_calls, 0)
        self.assertEqual(self.engine.started, 0)

    async def test_stop_delegates_to_engine(self) -> None:
        service = self.make_service()
        await service.start()
        await service.stop()
        self.assertEqual(self.engine.stopped, 1)
        self.assertIsNone(self.engine.dispatch)


class TestSweep(ServiceFixture, unittest.IsolatedAsyncioTestCase):
    async def test_sweep_claims_dispatches_and_delivers_in_run_order(self) -> None:
        service = self.make_service()
        older = await self.seed(prompt="older", run_at=NOW - timedelta(minutes=5))
        newer = await self.seed(prompt="newer", run_at=NOW - timedelta(minutes=1))
        future = await self.seed(prompt="future", run_at=NOW + timedelta(hours=1))

        count = await service.sweep()

        self.assertEqual(count, 2)
        self.assertEqual([action.id for action in self.handler.calls], [older.id, newer.id])
        self.assertEqual(self.store.delivered, [older.id, newer.id])
        self.assertNotIn(older.id, self.store.actions)
        self.assertNotIn(newer.id, self.store.actions)
        self.assertIn(future.id, self.store.actions)
        self.assertEqual(len(self.engine.notifications), 1)

    async def test_sweep_without_due_actions_does_not_notify(self) -> None:
        service = self.make_service()
        await self.seed(run_at=NOW + timedelta(hours=1))
        self.assertEqual(await service.sweep(), 0)
        self.assertEqual(self.handler.calls, [])
        self.assertEqual(self.engine.notifications, [])

    async def test_sweep_honours_batch_size(self) -> None:
        service = self.make_service(batch_size=2)
        for offset in range(3):
            await self.seed(run_at=NOW - timedelta(minutes=offset + 1))

        self.assertEqual(await service.sweep(), 2)
        self.assertEqual(await service.sweep(), 1)
        self.assertEqual(await service.sweep(), 0)
        self.assertEqual(len(self.store.delivered), 3)

    async def test_sweep_records_handler_failure_and_continues(self) -> None:
        service = self.make_service(handler=RecordingHandler(error=RuntimeError("boom")))
        first = await self.seed(run_at=NOW - timedelta(minutes=2))
        second = await self.seed(run_at=NOW - timedelta(minutes=1))

        count = await service.sweep()

        self.assertEqual(count, 2)
        self.assertEqual([action.id for action in self.handler.calls], [first.id, second.id])
        self.assertEqual([failure[0] for failure in self.store.failures], [first.id, second.id])
        self.assertIn("RuntimeError: boom", self.store.failures[0][1])
        self.assertEqual(self.store.delivered, [])
        self.assertEqual(self.store.actions[first.id].status, STATUS_FAILED)
        self.assertEqual(self.store.actions[second.id].status, STATUS_FAILED)
        self.assertEqual(len(self.engine.notifications), 1)

    async def test_sweep_without_handler_raises(self) -> None:
        service = self.make_service(handler=None)
        with self.assertRaises(SchedulerError) as ctx:
            await service.sweep()
        self.assertIn("no handler", str(ctx.exception))


class TestFire(ServiceFixture, unittest.IsolatedAsyncioTestCase):
    async def test_fire_delivers_due_action(self) -> None:
        service = self.make_service()
        action = await self.seed(run_at=NOW - timedelta(seconds=1))

        self.assertTrue(await service.fire(42, action.id))

        self.assertEqual(self.store.delivered, [action.id])
        self.assertNotIn(action.id, self.store.actions)
        self.assertEqual(len(self.engine.notifications), 1)

    async def test_fire_returns_false_for_future_unknown_or_foreign_actions(self) -> None:
        service = self.make_service()
        future = await self.seed(run_at=NOW + timedelta(hours=1), chat_id=42)
        foreign = await self.seed(run_at=NOW - timedelta(minutes=1), chat_id=99)

        self.assertFalse(await service.fire(42, future.id))
        self.assertFalse(await service.fire(42, "missing"))
        self.assertFalse(await service.fire(42, foreign.id))

        self.assertEqual(self.store.delivered, [])
        self.assertEqual(self.engine.notifications, [])
        self.assertIn(foreign.id, self.store.actions)

    async def test_fire_records_handler_failure_and_returns_true(self) -> None:
        service = self.make_service(handler=RecordingHandler(error=ValueError("nope")))
        action = await self.seed(run_at=NOW - timedelta(seconds=1))

        self.assertTrue(await service.fire(42, action.id))

        self.assertEqual(self.store.failures, [(action.id, "ValueError: nope")])
        self.assertEqual(self.store.actions[action.id].status, STATUS_FAILED)
        self.assertEqual(self.store.delivered, [])
        self.assertEqual(len(self.engine.notifications), 1)

    async def test_fire_without_handler_raises(self) -> None:
        service = self.make_service(handler=None)
        with self.assertRaises(SchedulerError) as ctx:
            await service.fire(42, "action-1")
        self.assertIn("no handler", str(ctx.exception))


class TestAddAction(ServiceFixture, unittest.IsolatedAsyncioTestCase):
    async def test_add_one_shot_from_offset_string(self) -> None:
        service = self.make_service()
        action = await service.add_action(42, "  ping  ", run_at="2026-10-20T15:00:00+02:00")

        self.assertEqual(action.prompt, "ping")
        self.assertEqual(action.chat_id, 42)
        self.assertEqual(action.run_at, datetime(2026, 10, 20, 13, 0, tzinfo=UTC))
        self.assertIsNone(action.cron)
        self.assertEqual(action.created_at, NOW)
        self.assertEqual(self.store.actions[action.id], action)
        self.assertEqual(len(self.engine.notifications), 1)

    async def test_add_parses_z_suffix(self) -> None:
        service = self.make_service()
        action = await service.add_action(42, "ping", run_at="2026-10-20T13:00:00Z")
        self.assertEqual(action.run_at, datetime(2026, 10, 20, 13, 0, tzinfo=UTC))

    async def test_add_naive_string_uses_store_timezone(self) -> None:
        service = self.make_service(store=FakeStore(timezone_name="Europe/Madrid"))
        action = await service.add_action(42, "ping", run_at="2026-10-20T15:00:00")
        self.assertEqual(action.run_at, datetime(2026, 10, 20, 13, 0, tzinfo=UTC))

    async def test_add_naive_datetime_uses_store_timezone(self) -> None:
        service = self.make_service()
        action = await service.add_action(42, "ping", run_at=datetime(2026, 10, 20, 13, 0))
        self.assertEqual(action.run_at, datetime(2026, 10, 20, 13, 0, tzinfo=UTC))

    async def test_add_rejects_past_and_current_run_at(self) -> None:
        service = self.make_service()
        for value in ("2026-10-20T11:59:59+00:00", NOW):
            with self.subTest(value=value):
                with self.assertRaises(SchedulerValidationError) as ctx:
                    await service.add_action(42, "ping", run_at=value)
                self.assertIn("future", str(ctx.exception))
                self.assertIn(NOW.isoformat(), str(ctx.exception))
        self.assertEqual(self.store.added, [])

    async def test_add_rejects_both_or_neither_schedule(self) -> None:
        service = self.make_service()
        with self.assertRaises(SchedulerValidationError) as ctx:
            await service.add_action(42, "ping")
        self.assertIn("exactly one", str(ctx.exception))
        with self.assertRaises(SchedulerValidationError) as ctx:
            await service.add_action(42, "ping", run_at="2026-10-20T13:00:00Z", cron="0 9 * * *")
        self.assertIn("exactly one", str(ctx.exception))

    async def test_add_rejects_empty_and_oversized_prompts(self) -> None:
        service = self.make_service()
        for prompt in ("", "   \n\t "):
            with self.subTest(prompt=prompt):
                with self.assertRaises(SchedulerValidationError) as ctx:
                    await service.add_action(42, prompt, run_at="2026-10-20T13:00:00Z")
                self.assertIn("empty", str(ctx.exception))
        with self.assertRaises(SchedulerValidationError) as ctx:
            await service.add_action(42, "x" * 4001, run_at="2026-10-20T13:00:00Z")
        self.assertIn("too long", str(ctx.exception))

    async def test_add_rejects_unparseable_run_at(self) -> None:
        service = self.make_service()
        with self.assertRaises(SchedulerValidationError) as ctx:
            await service.add_action(42, "ping", run_at="tomorrow")
        self.assertIn("ISO-8601", str(ctx.exception))

    async def test_add_normalizes_cron_and_computes_next_run(self) -> None:
        service = self.make_service()
        action = await service.add_action(42, "ping", cron=" 0   9  *  *  1-5 ")

        self.assertEqual(action.cron, "0 9 * * 1-5")
        self.assertEqual(action.run_at, datetime(2026, 10, 21, 9, 0, tzinfo=UTC))

    async def test_add_rejects_invalid_cron(self) -> None:
        service = self.make_service()
        for expression in ("not a cron", "* * * *", ""):
            with self.subTest(expression=expression):
                with self.assertRaises(SchedulerValidationError):
                    await service.add_action(42, "ping", cron=expression)

    async def test_add_rejects_invalid_store_timezone(self) -> None:
        service = self.make_service(store=FakeStore(timezone_name="Nowhere/Nothing"))
        with self.assertRaises(SchedulerValidationError) as ctx:
            await service.add_action(42, "ping", cron="0 9 * * *")
        self.assertIn("Nowhere/Nothing", str(ctx.exception))
        with self.assertRaises(SchedulerValidationError):
            await service.add_action(42, "ping", run_at=datetime(2026, 10, 20, 13, 0))

    async def test_add_validates_max_runs(self) -> None:
        service = self.make_service()
        action = await service.add_action(42, "ping", run_at="2026-10-20T13:00:00Z", max_runs=3)
        self.assertEqual(action.max_runs, 3)
        unlimited = await service.add_action(42, "pong", run_at="2026-10-20T13:00:00Z")
        self.assertIsNone(unlimited.max_runs)
        for value in (0, -5, True):
            with self.subTest(max_runs=value):
                with self.assertRaises(SchedulerValidationError) as ctx:
                    await service.add_action(42, "ping", run_at="2026-10-20T14:00:00Z", max_runs=value)
                self.assertIn("max_runs", str(ctx.exception))

    async def test_add_propagates_store_quota_error(self) -> None:
        service = self.make_service(store=FakeStore(max_actions_per_chat=1))
        await service.add_action(42, "first", run_at="2026-10-20T13:00:00Z")
        with self.assertRaises(SchedulerLimitError):
            await service.add_action(42, "second", run_at="2026-10-20T14:00:00Z")

    async def test_add_returns_an_independent_copy(self) -> None:
        service = self.make_service()
        created = await service.add_action(42, "ping", run_at="2026-10-20T13:00:00Z")
        created.prompt = "mutated"
        self.assertEqual(self.store.actions[created.id].prompt, "ping")


class TestListAndDelete(ServiceFixture, unittest.IsolatedAsyncioTestCase):
    async def test_list_is_chat_scoped_and_returns_copies(self) -> None:
        service = self.make_service()
        later = await self.seed(prompt="later", run_at=NOW + timedelta(hours=2), chat_id=42)
        sooner = await self.seed(prompt="sooner", run_at=NOW + timedelta(hours=1), chat_id=42)
        await self.seed(prompt="other chat", run_at=NOW, chat_id=7)

        actions = await service.list_actions(42)

        self.assertEqual([action.id for action in actions], [sooner.id, later.id])
        actions[0].prompt = "mutated"
        self.assertEqual(self.store.actions[sooner.id].prompt, "sooner")

    async def test_delete_is_chat_scoped_and_notifies_on_success(self) -> None:
        service = self.make_service()
        own = await self.seed(chat_id=42)
        foreign = await self.seed(chat_id=7)

        self.assertFalse(await service.delete_action(42, foreign.id))
        self.assertIn(foreign.id, self.store.actions)
        self.assertEqual(self.engine.notifications, [])

        self.assertTrue(await service.delete_action(42, own.id))
        self.assertNotIn(own.id, self.store.actions)
        self.assertEqual(len(self.engine.notifications), 1)

    async def test_delete_missing_returns_false_without_notifying(self) -> None:
        service = self.make_service()
        self.assertFalse(await service.delete_action(42, "missing"))
        self.assertEqual(self.engine.notifications, [])

    async def test_clear_actions_is_chat_scoped_and_reports_count(self) -> None:
        service = self.make_service()
        first = await self.seed(chat_id=42, run_at=NOW + timedelta(hours=1))
        second = await self.seed(chat_id=42, run_at=NOW + timedelta(hours=2))
        foreign = await self.seed(chat_id=7, run_at=NOW + timedelta(hours=1))
        notifications_before = len(self.engine.notifications)

        self.assertEqual(await service.clear_actions(42), 2)

        self.assertNotIn(first.id, self.store.actions)
        self.assertNotIn(second.id, self.store.actions)
        self.assertIn(foreign.id, self.store.actions)
        self.assertEqual(len(self.engine.notifications), notifications_before + 1)

    async def test_clear_actions_empty_chat_returns_zero_without_notifying(self) -> None:
        service = self.make_service()
        self.assertEqual(await service.clear_actions(42), 0)
        self.assertEqual(self.engine.notifications, [])


class TestServiceInstructions(ServiceFixture, unittest.IsolatedAsyncioTestCase):
    def test_as_instructions_uses_store_settings(self) -> None:
        service = self.make_service()
        text = service.as_instructions()
        self.assertIn("UTC", text)
        self.assertIn("20", text)

    async def test_active_instructions_is_chat_scoped(self) -> None:
        service = self.make_service()
        own = await self.seed(chat_id=42, run_at=NOW + timedelta(hours=1))
        foreign = await self.seed(chat_id=7, run_at=NOW + timedelta(hours=1))

        text = await service.active_instructions(42)

        self.assertIn(own.id, text)
        self.assertNotIn(foreign.id, text)


class SweepOnlyEngine(RecordingEngine):
    """Engine that consumes no action snapshots (the shipped engines)."""

    wants_action_snapshots = False


class CountingStore(FakeStore):
    """FakeStore counting engine-facing snapshots."""

    def __init__(self) -> None:
        super().__init__()
        self.list_all_calls = 0

    async def list_all(self) -> list[ScheduledAction]:
        self.list_all_calls += 1
        return await super().list_all()


class FailingDeliveryStore(FakeStore):
    """FakeStore whose first delivery write fails."""

    def __init__(self) -> None:
        super().__init__()
        self.delivery_failures_left = 1

    async def mark_delivered(self, action: ScheduledAction, now: datetime) -> None:
        if self.delivery_failures_left:
            self.delivery_failures_left -= 1
            raise OSError("disk full")
        await super().mark_delivered(action, now)


class FailingFailureStore(FakeStore):
    """FakeStore whose failure write always fails."""

    async def record_failure(self, action: ScheduledAction, error: str, now: datetime) -> None:
        raise OSError("disk full")


class TestBookkeepingResilience(ServiceFixture, unittest.IsolatedAsyncioTestCase):
    async def test_delivery_write_failure_does_not_abort_the_batch(self) -> None:
        store = FailingDeliveryStore()
        service = self.make_service(store=store)
        first = await self.seed(run_at=NOW - timedelta(minutes=2))
        second = await self.seed(run_at=NOW - timedelta(minutes=1))

        self.assertEqual(await service.sweep(), 2)
        self.assertEqual([action.id for action in self.handler.calls], [first.id, second.id])
        self.assertEqual(store.delivered, [second.id])
        # The first claim could not be finalized; restart recovery handles it.
        self.assertEqual(store.actions[first.id].status, STATUS_FIRING)

    async def test_failure_write_failure_does_not_abort_the_batch(self) -> None:
        store = FailingFailureStore()
        service = self.make_service(store=store, handler=RecordingHandler(error=RuntimeError("boom")))
        first = await self.seed(run_at=NOW - timedelta(minutes=2))
        second = await self.seed(run_at=NOW - timedelta(minutes=1))

        self.assertEqual(await service.sweep(), 2)
        self.assertEqual([action.id for action in self.handler.calls], [first.id, second.id])
        self.assertEqual(store.actions[first.id].status, STATUS_FIRING)
        self.assertEqual(store.actions[second.id].status, STATUS_FIRING)


class TestEngineSnapshotCapability(ServiceFixture, unittest.IsolatedAsyncioTestCase):
    async def test_sweep_engines_skip_snapshot_reads(self) -> None:
        store = CountingStore()
        service = self.make_service(store=store, engine=SweepOnlyEngine())

        await service.start()
        await service.add_action(42, "hi", run_at=NOW + timedelta(hours=1))
        await service.delete_action(42, (await service.list_actions(42))[0].id)

        self.assertEqual(store.list_all_calls, 0)
