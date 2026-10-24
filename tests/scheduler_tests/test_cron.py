"""Tests for cron parsing and next-occurrence calculation."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from manolo_bot.scheduler.cron import next_cron_run, validate_cron, validate_timezone


class TestValidateCron(unittest.TestCase):
    def test_normalizes_whitespace(self) -> None:
        self.assertEqual(validate_cron("  0   9  *  *  1-5 "), "0 9 * * 1-5")

    def test_accepts_five_fields(self) -> None:
        self.assertEqual(validate_cron("*/15 * * * *"), "*/15 * * * *")

    def test_rejects_wrong_field_count(self) -> None:
        with self.assertRaises(ValueError):
            validate_cron("* * * *")
        with self.assertRaises(ValueError):
            validate_cron("* * * * * *")

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            validate_cron("")

    def test_rejects_garbage(self) -> None:
        with self.assertRaises(ValueError):
            validate_cron("not a cron")


class TestNextCronRun(unittest.TestCase):
    def test_returns_strictly_after_reference(self) -> None:
        now = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
        self.assertEqual(next_cron_run("*/15 * * * *", now, "UTC"), now + timedelta(minutes=15))

    def test_naive_reference_is_utc(self) -> None:
        now = datetime(2026, 10, 20, 12, 0)
        self.assertEqual(
            next_cron_run("*/15 * * * *", now, "UTC"),
            datetime(2026, 10, 20, 12, 15, tzinfo=UTC),
        )

    def test_timezone_is_respected(self) -> None:
        # 09:00 in Madrid (CEST, UTC+2) is 07:00 UTC.
        now = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
        self.assertEqual(
            next_cron_run("0 9 * * *", now, "Europe/Madrid"),
            datetime(2026, 10, 21, 7, 0, tzinfo=UTC),
        )

    def test_dst_transition_uses_local_wall_clock(self) -> None:
        # Madrid leaves DST on 2026-10-25 at 03:00 local. 09:00 is still CEST
        # on the 24th and CET on the 26th.
        now = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
        self.assertEqual(
            next_cron_run("0 9 * * *", now, "Europe/Madrid"),
            datetime(2026, 10, 24, 7, 0, tzinfo=UTC),
        )
        self.assertEqual(
            next_cron_run("0 9 * * *", datetime(2026, 10, 25, 12, 0, tzinfo=UTC), "Europe/Madrid"),
            datetime(2026, 10, 26, 8, 0, tzinfo=UTC),
        )

    def test_impossible_expression_raises(self) -> None:
        now = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
        with self.assertRaises(ValueError):
            next_cron_run("0 0 31 2 *", now, "UTC")

    def test_unknown_timezone_raises(self) -> None:
        with self.assertRaises(ValueError):
            next_cron_run("0 9 * * *", datetime(2026, 10, 20, 12, 0, tzinfo=UTC), "Mars/Olympus")


class TestValidateTimezone(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(str(validate_timezone("Europe/Madrid")), "Europe/Madrid")

    def test_invalid(self) -> None:
        with self.assertRaises(ValueError):
            validate_timezone("Not/AZone")


if __name__ == "__main__":
    unittest.main()
