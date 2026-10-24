"""Redis-backed scheduler store (shared, durable snapshot)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import redis

from manolo_bot.scheduler.base import (
    STATUS_ACTIVE,
    BaseSchedulerStore,
    ScheduledAction,
    apply_catchup_grace,
    apply_claim,
    apply_delivered,
    apply_failure,
    apply_interrupted,
    ensure_chat_limit,
    ensure_utc,
)

logger = logging.getLogger(__name__)

#: Snapshot schema version. Unknown versions fail loudly instead of being
#: overwritten with an empty state.
_SNAPSHOT_VERSION = 1


def _as_str(payload: str | bytes | bytearray | memoryview) -> str:
    """Decode a Redis payload (redis-py returns bytes by default)."""
    if isinstance(payload, str):
        return payload
    return bytes(payload).decode("utf-8")


class RedisSchedulerStore(BaseSchedulerStore):
    """Scheduler store persisted as one JSON snapshot per bot namespace.

    Storage is a single Redis string (``{bot_uuid}:scheduler:snapshot``) shared
    by every process using the same namespace. Mutations take a short-lived
    Redis lock (``{bot_uuid}:scheduler:lock``, ``SET NX PX`` + compare-and-delete)
    so read-modify-write cycles and claims are atomic across processes. That is
    enough for correct claims and durable schedules; it does **not** by itself
    make multi-instance agent delivery safe (a distributed per-chat lock and a
    shared message store are separate concerns).

    The snapshot mirrors the filesystem store schema. A corrupt or
    unknown-version snapshot raises instead of being replaced, so data is never
    silently destroyed.

    :param redis_url: Redis URL used when ``redis_client`` is not supplied.
    :param redis_client: Explicit async client (``get``/``set``/``eval``) for
        tests or dependency injection.
    :param lock_timeout_seconds: How long to wait for the distributed lock.
    :param lock_ttl_ms: Lock lease duration; keep it above one snapshot write.
    """

    def __init__(
        self,
        bot_uuid: str = "default",
        *,
        max_actions_per_chat: int = 20,
        max_consecutive_failures: int = 5,
        catchup_grace_seconds: int = 0,
        default_timezone: str = "UTC",
        redis_url: str = "redis://localhost:6379/0",
        redis_client: Any = None,
        lock_timeout_seconds: float = 5.0,
        lock_ttl_ms: int = 5000,
    ) -> None:
        super().__init__(
            bot_uuid=bot_uuid,
            max_actions_per_chat=max_actions_per_chat,
            max_consecutive_failures=max_consecutive_failures,
            catchup_grace_seconds=catchup_grace_seconds,
            default_timezone=default_timezone,
        )
        self._redis_url = redis_url
        self._redis = redis_client
        self._lock_timeout_seconds = lock_timeout_seconds
        self._lock_ttl_ms = lock_ttl_ms
        self._lock = asyncio.Lock()

    @property
    def snapshot_key(self) -> str:
        """Redis key holding this bot's action snapshot."""
        return f"{self.bot_uuid}:scheduler:snapshot"

    @property
    def lock_key(self) -> str:
        """Redis key used as the cross-process mutation lock."""
        return f"{self.bot_uuid}:scheduler:lock"

    async def _client(self):
        """Lazily connected async client (explicit client wins, for tests/DI)."""
        if self._redis is None:
            self._redis = redis.asyncio.Redis.from_url(self._redis_url)
        return self._redis

    async def _acquire_lock(self, client) -> str:
        token = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._lock_timeout_seconds
        while True:
            acquired = await client.set(self.lock_key, token, nx=True, px=self._lock_ttl_ms)
            if acquired:
                return token
            if loop.time() >= deadline:
                raise TimeoutError(f"Could not acquire scheduler lock {self.lock_key!r}")
            await asyncio.sleep(0.05)

    async def _release_lock(self, client, token: str) -> None:
        """Delete the lock only while it still holds our token (atomic WATCH)."""
        try:
            async with client.pipeline(transaction=True) as pipe:
                await pipe.watch(self.lock_key)
                current = await pipe.get(self.lock_key)
                if current is None or _as_str(current) != token:
                    await pipe.unwatch()
                    return
                pipe.multi()
                pipe.delete(self.lock_key)
                await pipe.execute()
        except Exception:
            logger.exception("Failed to release scheduler lock %s", self.lock_key)

    @asynccontextmanager
    async def _locked(self) -> AsyncIterator[Any]:
        """Serialize one read-modify-write across processes."""
        client = await self._client()
        async with self._lock:
            token = await self._acquire_lock(client)
            try:
                yield client
            finally:
                await self._release_lock(client, token)

    async def _load_actions(self, client) -> dict[str, ScheduledAction]:
        raw = await client.get(self.snapshot_key)
        if raw is None:
            return {}
        try:
            payload = json.loads(_as_str(raw))
        except ValueError as exc:
            raise ValueError(f"Corrupt scheduler snapshot at {self.snapshot_key!r}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != _SNAPSHOT_VERSION:
            raise ValueError(
                f"Unsupported scheduler snapshot version at {self.snapshot_key!r}; refusing to overwrite it"
            )
        actions: dict[str, ScheduledAction] = {}
        for item in payload.get("actions", []):
            action = ScheduledAction.from_dict(item)
            actions[action.id] = action
        return actions

    async def _save_actions(self, client, actions: dict[str, ScheduledAction]) -> None:
        payload = {
            "version": _SNAPSHOT_VERSION,
            "actions": [action.to_dict() for action in actions.values()],
        }
        await client.set(self.snapshot_key, json.dumps(payload))

    async def build_or_load(self) -> None:
        """Load the shared snapshot and recover records left in ``firing``."""
        async with self._locked() as client:
            actions = await self._load_actions(client)
            changed = False
            for action_id, action in list(actions.items()):
                recovered = apply_interrupted(action)
                if recovered != action:
                    actions[action_id] = recovered
                    changed = True
            if changed:
                await self._save_actions(client, actions)

    async def add(self, action: ScheduledAction) -> ScheduledAction:
        """Persist a new action after enforcing the per-chat quota."""
        async with self._locked() as client:
            actions = await self._load_actions(client)
            chat_actions = [item for item in actions.values() if item.chat_id == action.chat_id]
            ensure_chat_limit(chat_actions, self.max_actions_per_chat)
            actions[action.id] = action.clone()
            await self._save_actions(client, actions)
            return action.clone()

    async def list(self, chat_id: int) -> list[ScheduledAction]:
        """Return one chat's records sorted by ``(run_at, id)``."""
        client = await self._client()
        actions = await self._load_actions(client)
        return sorted(
            (action.clone() for action in actions.values() if action.chat_id == chat_id),
            key=lambda action: (action.run_at, action.id),
        )

    async def delete(self, chat_id: int, action_id: str) -> bool:
        """Delete one action from one chat, returning whether it existed."""
        async with self._locked() as client:
            actions = await self._load_actions(client)
            action = actions.get(action_id)
            if action is None or action.chat_id != chat_id:
                return False
            del actions[action_id]
            await self._save_actions(client, actions)
            return True

    async def claim_due(self, now: datetime, limit: int = 10) -> list[ScheduledAction]:
        """Atomically claim up to ``limit`` due actions across every chat."""
        now = ensure_utc(now)
        async with self._locked() as client:
            actions = await self._load_actions(client)
            due = sorted(
                (action for action in actions.values() if action.status == STATUS_ACTIVE and action.run_at <= now),
                key=lambda action: (action.run_at, action.chat_id, action.id),
            )
            claimed: list[ScheduledAction] = []
            changed = False
            for action in due[: max(0, int(limit))]:
                skipped = apply_catchup_grace(action, now, self.default_timezone, self.catchup_grace_seconds)
                if skipped is not None:
                    actions[action.id] = skipped
                    changed = True
                    continue
                snapshot = apply_claim(action, now, self.default_timezone)
                actions[action.id] = snapshot
                claimed.append(snapshot.clone())
                changed = True
            if changed:
                await self._save_actions(client, actions)
            return claimed

    async def claim_one(self, chat_id: int, action_id: str, now: datetime) -> ScheduledAction | None:
        """Atomically claim one specific action, or return None if not due."""
        now = ensure_utc(now)
        async with self._locked() as client:
            actions = await self._load_actions(client)
            action = actions.get(action_id)
            if action is None or action.chat_id != chat_id or action.status != STATUS_ACTIVE or action.run_at > now:
                return None
            skipped = apply_catchup_grace(action, now, self.default_timezone, self.catchup_grace_seconds)
            if skipped is not None:
                actions[action_id] = skipped
                await self._save_actions(client, actions)
                return None
            snapshot = apply_claim(action, now, self.default_timezone)
            actions[action_id] = snapshot
            await self._save_actions(client, actions)
            return snapshot.clone()

    async def mark_delivered(self, action: ScheduledAction, now: datetime) -> None:
        """Finalize a successful delivery (remove one-shots, reactivate cron)."""
        async with self._locked() as client:
            actions = await self._load_actions(client)
            stored = actions.get(action.id)
            if stored is None or stored.chat_id != action.chat_id:
                logger.warning(
                    "Scheduler delivery for unknown action %s (chat %s); ignoring",
                    action.id,
                    action.chat_id,
                )
                return
            updated = apply_delivered(action)
            if updated is None:
                del actions[action.id]
            else:
                actions[action.id] = updated
            await self._save_actions(client, actions)

    async def record_failure(self, action: ScheduledAction, error: str, now: datetime) -> None:
        """Record a failed delivery and apply the failure policy."""
        async with self._locked() as client:
            actions = await self._load_actions(client)
            stored = actions.get(action.id)
            if stored is None or stored.chat_id != action.chat_id:
                logger.warning(
                    "Scheduler failure for unknown action %s (chat %s); ignoring",
                    action.id,
                    action.chat_id,
                )
                return
            actions[action.id] = apply_failure(action, error, self.max_consecutive_failures)
            await self._save_actions(client, actions)

    async def clear(self, chat_id: int) -> None:
        """Remove every scheduled action belonging to one chat."""
        async with self._locked() as client:
            actions = await self._load_actions(client)
            remaining = {action_id: action for action_id, action in actions.items() if action.chat_id != chat_id}
            if len(remaining) != len(actions):
                await self._save_actions(client, remaining)

    async def list_all(self) -> list[ScheduledAction]:
        """Return every action across chats, sorted by ``(chat_id, run_at, id)``."""
        client = await self._client()
        actions = await self._load_actions(client)
        return sorted(
            (action.clone() for action in actions.values()),
            key=lambda action: (action.chat_id, action.run_at, action.id),
        )


__all__ = ["RedisSchedulerStore"]
