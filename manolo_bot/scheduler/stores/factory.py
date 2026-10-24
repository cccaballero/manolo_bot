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
    redis_url: str | None = None,
) -> BaseSchedulerStore:
    """Build a scheduler store by name.

    :param name: ``"in_memory"`` (ephemeral, tests/single-process),
        ``"local_fs"`` (single-node JSON snapshot persistence) or ``"redis"``
        (shared durable snapshot with cross-process locking).
    :param bot_uuid: Bot namespace used for storage isolation.
    :param store_path: Root store dir for ``"local_fs"``; defaults to a tmp
        ``manolo_bot/scheduler`` directory.
    :param max_actions_per_chat: Quota of open actions per chat.
    :param max_consecutive_failures: Retry budget before a recurring action
        is marked failed.
    :param catchup_grace_seconds: How late an overdue action may still fire.
    :param default_timezone: Timezone used to evaluate cron expressions.
    :param redis_url: Redis URL for the ``"redis"`` store (falls back to the
        default localhost URL when unset).
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
    if normalized == "redis":
        from manolo_bot.scheduler.stores.redis_store import RedisSchedulerStore

        return RedisSchedulerStore(
            redis_url=redis_url or "redis://localhost:6379/0",
            **common,
        )
    raise ValueError(f"Unsupported scheduler store: {name!r}")


__all__ = ["build_scheduler_store"]
