"""Scheduler store factory."""

from __future__ import annotations

from pathlib import Path

from manolo_bot.scheduler.base import BaseSchedulerStore
from manolo_bot.scheduler.stores.filesystem_store import FilesystemSchedulerStore
from manolo_bot.scheduler.stores.inmemory_store import InMemorySchedulerStore


def build_scheduler_store(
    name: str,
    *,
    bot_uuid: str = "default",
    store_path: str | Path | None = None,
    max_actions_per_chat: int = 20,
    max_consecutive_failures: int = 5,
    catchup_grace_seconds: int = 0,
    default_timezone: str = "UTC",
) -> BaseSchedulerStore:
    """Build a scheduler store by name.

    :param name: ``"in_memory"`` (ephemeral, tests/single-process) or
        ``"local_fs"`` (single-node JSON snapshot persistence).
    :param bot_uuid: Bot namespace used for filesystem isolation.
    :param store_path: Root store dir for ``"local_fs"``; defaults to a tmp
        ``manolo_bot/scheduler`` directory.
    :param max_actions_per_chat: Quota of open actions per chat.
    :param max_consecutive_failures: Retry budget before a recurring action
        is marked failed.
    :param catchup_grace_seconds: How late an overdue action may still fire.
    :param default_timezone: Timezone used to evaluate cron expressions.
    :raises ValueError: For any unsupported store name.
    """
    normalized = (name or "").strip().lower()
    common = {
        "bot_uuid": bot_uuid,
        "max_actions_per_chat": max_actions_per_chat,
        "max_consecutive_failures": max_consecutive_failures,
        "catchup_grace_seconds": catchup_grace_seconds,
        "default_timezone": default_timezone,
    }
    if normalized == "in_memory":
        return InMemorySchedulerStore(**common)
    if normalized == "local_fs":
        return FilesystemSchedulerStore(store_path=store_path, **common)
    raise ValueError(f"Unsupported scheduler store: {name!r}")


__all__ = ["build_scheduler_store"]
