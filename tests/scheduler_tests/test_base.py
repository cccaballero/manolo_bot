"""Tests for scheduler models and the shared state machine."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta, timezone

from manolo_bot.scheduler.base import (
    BASE_SCHEDULER_TOOL_NAMES,
    MAX_PROMPT_CHARS,
    SCHEDULED_TRIGGER_PREFIX,
    STATUS_ACTIVE,
    STATUS_FAILED,
    STATUS_FIRING,
    ScheduledAction,
    SchedulerLimitError,
    apply_catchup_grace,
    apply_claim,
    apply_delivered,
    apply_failure,
    apply_interrupted,
    ensure_chat_limit,
    ensure_utc,
    is_open,
    new_action_id,
    normalize_scope,
)

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def make_action(**overrides) -> ScheduledAction:
    data = {
        "id": "action-1",
        "chat_id": 42,
        "prompt": "send the report",
        "run_at": NOW,
        "created_at": NOW - timedelta(minutes=5),
    }
    data.update(overrides)
    return ScheduledAction(**data)


class TestScheduledAction(unittest.TestCase):
    def test_naive_datetimes_are_treated_as_utc(self) -> None:
        action = make_action(run_at=datetime(2026, 10, 20, 12, 0), created_at=datetime(2026, 10, 20, 11, 0))
        self.assertEqual(action.run_at, NOW)
        self.assertEqual(action.created_at.tzinfo, UTC)

    def test_serialization_round_trip(self) -> None:
        action = make_action(cron="*/15 * * * *", max_runs=3, run_count=1, attempts=2, last_error="boom")
        restored = ScheduledAction.from_dict(action.to_dict())
        self.assertEqual(restored, action)

    def test_from_dict_defaults_for_optional_fields(self) -> None:
        payload = make_action().to_dict()
        for key in ("cron", "max_runs", "run_count", "attempts", "status", "last_error"):
            payload.pop(key)
        restored = ScheduledAction.from_dict(payload)
        self.assertIsNone(restored.cron)
        self.assertIsNone(restored.max_runs)
        self.assertEqual(restored.run_count, 0)
        self.assertEqual(restored.status, STATUS_ACTIVE)

    def test_clone_is_independent(self) -> None:
        action = make_action()
        clone = action.clone()
        clone.prompt = "changed"
        self.assertNotEqual(action.prompt, clone.prompt)

    def test_is_recurring(self) -> None:
        self.assertFalse(make_action().is_recurring)
        self.assertTrue(make_action(cron="0 9 * * *").is_recurring)

    def test_ids_are_unique(self) -> None:
        self.assertNotEqual(new_action_id(), new_action_id())

    def test_constants_are_consistent(self) -> None:
        self.assertEqual(len(set(BASE_SCHEDULER_TOOL_NAMES)), 3)
        self.assertTrue(SCHEDULED_TRIGGER_PREFIX.startswith("["))
        self.assertGreater(MAX_PROMPT_CHARS, 0)


class TestScheduledActionScope(unittest.TestCase):
    def test_int_chat_id_normalizes_to_string(self) -> None:
        self.assertEqual(make_action(chat_id=42).chat_id, "42")

    def test_string_chat_id_is_preserved(self) -> None:
        self.assertEqual(make_action(chat_id="C012345").chat_id, "C012345")

    def test_to_dict_emits_the_string_scope(self) -> None:
        payload = make_action(chat_id=42).to_dict()
        self.assertEqual(payload["chat_id"], "42")
        self.assertIsInstance(payload["chat_id"], str)

    def test_from_dict_accepts_int_and_str_scopes(self) -> None:
        payload = make_action(chat_id=42).to_dict()
        payload["chat_id"] = 42
        self.assertEqual(ScheduledAction.from_dict(payload).chat_id, "42")
        payload["chat_id"] = "C012345"
        self.assertEqual(ScheduledAction.from_dict(payload).chat_id, "C012345")

    def test_padded_scopes_are_canonicalized(self) -> None:
        self.assertEqual(make_action(chat_id=" 42 ").chat_id, "42")
        self.assertEqual(make_action(chat_id="\tC012345\n").chat_id, "C012345")

    def test_empty_scopes_are_rejected(self) -> None:
        for value in (None, "", "   ", "\n\t "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_action(chat_id=value)


class TestNormalizeScope(unittest.TestCase):
    def test_numeric_values_become_strings(self) -> None:
        self.assertEqual(normalize_scope(42), "42")
        self.assertEqual(normalize_scope("42"), "42")

    def test_opaque_values_are_preserved(self) -> None:
        self.assertEqual(normalize_scope("C012345"), "C012345")
        self.assertEqual(normalize_scope("dashboard:user:agent:uuid"), "dashboard:user:agent:uuid")

    def test_padded_values_are_stripped(self) -> None:
        self.assertEqual(normalize_scope(" 42 "), "42")
        self.assertEqual(normalize_scope(" 42 "), normalize_scope(42))
        self.assertEqual(normalize_scope("\tC012345\n"), "C012345")

    def test_blank_values_raise(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_scope(value)


class TestEnsureUtc(unittest.TestCase):
    def test_aware_is_converted(self) -> None:
        madrid = timezone(timedelta(hours=2))
        value = datetime(2026, 10, 20, 14, 0, tzinfo=madrid)
        self.assertEqual(ensure_utc(value), NOW)

    def test_naive_gets_utc(self) -> None:
        self.assertEqual(ensure_utc(datetime(2026, 10, 20, 12, 0)), NOW)


class TestChatLimit(unittest.TestCase):
    def test_under_limit_passes(self) -> None:
        ensure_chat_limit([make_action()], 2)

    def test_at_limit_raises(self) -> None:
        with self.assertRaises(SchedulerLimitError):
            ensure_chat_limit([make_action(id="a"), make_action(id="b")], 2)

    def test_failed_records_do_not_hold_a_slot(self) -> None:
        actions = [make_action(id="a"), make_action(id="b", status=STATUS_FAILED)]
        ensure_chat_limit(actions, 2)

    def test_firing_holds_a_slot(self) -> None:
        actions = [make_action(id="a"), make_action(id="b", status=STATUS_FIRING)]
        with self.assertRaises(SchedulerLimitError):
            ensure_chat_limit(actions, 2)

    def test_is_open(self) -> None:
        self.assertTrue(is_open(make_action()))
        self.assertTrue(is_open(make_action(status=STATUS_FIRING)))
        self.assertFalse(is_open(make_action(status=STATUS_FAILED)))


class TestClaim(unittest.TestCase):
    def test_one_shot_is_marked_firing(self) -> None:
        claimed = apply_claim(make_action(), NOW, "UTC")
        self.assertEqual(claimed.status, STATUS_FIRING)
        self.assertEqual(claimed.run_count, 1)
        self.assertEqual(claimed.run_at, NOW)

    def test_cron_advances_to_next_occurrence(self) -> None:
        claimed = apply_claim(make_action(cron="*/15 * * * *"), NOW, "UTC")
        self.assertEqual(claimed.run_at, NOW + timedelta(minutes=15))
        self.assertEqual(claimed.status, STATUS_FIRING)

    def test_cron_uses_configured_timezone(self) -> None:
        # 09:00 Europe/Madrid == 07:00 UTC.
        action = make_action(cron="0 9 * * *")
        claimed = apply_claim(action, NOW, "Europe/Madrid")
        self.assertEqual(claimed.run_at, datetime(2026, 10, 21, 7, 0, tzinfo=UTC))


class TestCatchupGrace(unittest.TestCase):
    def test_disabled_returns_none(self) -> None:
        self.assertIsNone(apply_catchup_grace(make_action(run_at=NOW - timedelta(days=1)), NOW, "UTC", 0))

    def test_within_window_returns_none(self) -> None:
        action = make_action(run_at=NOW - timedelta(minutes=5))
        self.assertIsNone(apply_catchup_grace(action, NOW, "UTC", 3600))

    def test_overdue_one_shot_fails(self) -> None:
        action = make_action(run_at=NOW - timedelta(days=1))
        replacement = apply_catchup_grace(action, NOW, "UTC", 60)
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.status, STATUS_FAILED)
        self.assertIn("Missed run", replacement.last_error or "")

    def test_overdue_cron_advances_and_stays_active(self) -> None:
        action = make_action(cron="*/15 * * * *", run_at=NOW - timedelta(days=1))
        replacement = apply_catchup_grace(action, NOW, "UTC", 60)
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.status, STATUS_ACTIVE)
        self.assertEqual(replacement.run_at, NOW + timedelta(minutes=15))


class TestDelivered(unittest.TestCase):
    def test_one_shot_is_removed(self) -> None:
        self.assertIsNone(apply_delivered(make_action(status=STATUS_FIRING)))

    def test_cron_reactivates_and_resets_attempts(self) -> None:
        action = make_action(cron="*/15 * * * *", status=STATUS_FIRING, attempts=3, last_error="x")
        delivered = apply_delivered(action)
        self.assertIsNotNone(delivered)
        self.assertEqual(delivered.status, STATUS_ACTIVE)
        self.assertEqual(delivered.attempts, 0)
        self.assertIsNone(delivered.last_error)

    def test_cron_is_removed_at_max_runs(self) -> None:
        action = make_action(cron="*/15 * * * *", status=STATUS_FIRING, run_count=2, max_runs=2)
        self.assertIsNone(apply_delivered(action))


class TestFailure(unittest.TestCase):
    def test_one_shot_fails_permanently(self) -> None:
        failed = apply_failure(make_action(status=STATUS_FIRING), "boom", 5)
        self.assertEqual(failed.status, STATUS_FAILED)
        self.assertEqual(failed.attempts, 1)
        self.assertEqual(failed.last_error, "boom")

    def test_cron_stays_active_until_cap(self) -> None:
        action = make_action(cron="*/15 * * * *", status=STATUS_FIRING, attempts=0)
        failed = apply_failure(action, "boom", 5)
        self.assertEqual(failed.status, STATUS_ACTIVE)
        self.assertEqual(failed.attempts, 1)

    def test_cron_fails_at_cap(self) -> None:
        action = make_action(cron="*/15 * * * *", status=STATUS_FIRING, attempts=4)
        failed = apply_failure(action, "boom", 5)
        self.assertEqual(failed.status, STATUS_FAILED)
        self.assertEqual(failed.attempts, 5)

    def test_cron_fails_when_run_budget_exhausted(self) -> None:
        action = make_action(cron="*/15 * * * *", status=STATUS_FIRING, run_count=2, max_runs=2)
        failed = apply_failure(action, "boom", 5)
        self.assertEqual(failed.status, STATUS_FAILED)


class TestInterrupted(unittest.TestCase):
    def test_one_shot_becomes_failed(self) -> None:
        recovered = apply_interrupted(make_action(status=STATUS_FIRING))
        self.assertEqual(recovered.status, STATUS_FAILED)
        self.assertIn("interrupted", (recovered.last_error or "").lower())

    def test_cron_returns_to_active(self) -> None:
        recovered = apply_interrupted(make_action(cron="*/15 * * * *", status=STATUS_FIRING))
        self.assertEqual(recovered.status, STATUS_ACTIVE)

    def test_other_statuses_are_untouched(self) -> None:
        action = make_action()
        self.assertIs(apply_interrupted(action), action)


if __name__ == "__main__":
    unittest.main()
