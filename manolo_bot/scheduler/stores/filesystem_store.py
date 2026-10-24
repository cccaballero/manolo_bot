"""Filesystem-backed scheduler store (single-node JSON snapshot)."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

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

#: JSON schema version written to ``schedules.json``.
_FORMAT_VERSION = 1

#: Default root for persisted scheduler state.
DEFAULT_STORE_PATH = Path(tempfile.gettempdir()) / "manolo_bot" / "scheduler"


def schedules_path(store_path: str | Path, bot_uuid: str) -> Path:
    """Return the persisted schedule file for a bot namespace."""
    return Path(store_path) / bot_uuid / "schedules.json"


def schedules_lock_path(path: str | Path) -> Path:
    """Return the sidecar lock file used to serialize access to ``path``."""
    return Path(f"{path}.lock")


class FilesystemSchedulerStore(BaseSchedulerStore):
    """Scheduler store persisted as one JSON snapshot per bot namespace.

    Layout: ``<store_path>/<bot_uuid>/schedules.json`` plus a sidecar
    ``schedules.json.lock``. Every read-modify-write takes an ``fcntl.flock``
    on the sidecar (multi-process safety) and a process-local
    :class:`asyncio.Lock`; blocking IO runs in a worker thread so the event
    loop is never blocked. Writes are atomic (temp file + ``os.replace``).

    This is simple single-node persistence, not a horizontally scalable store.
    """

    def __init__(
        self,
        bot_uuid: str = "default",
        store_path: str | Path | None = None,
        *,
        max_actions_per_chat: int = 20,
        max_consecutive_failures: int = 5,
        catchup_grace_seconds: int = 0,
        default_timezone: str = "UTC",
    ) -> None:
        super().__init__(
            bot_uuid=bot_uuid,
            max_actions_per_chat=max_actions_per_chat,
            max_consecutive_failures=max_consecutive_failures,
            catchup_grace_seconds=catchup_grace_seconds,
            default_timezone=default_timezone,
        )
        self.store_path = Path(store_path) if store_path is not None else DEFAULT_STORE_PATH
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        """Path of this namespace's ``schedules.json`` file."""
        return schedules_path(self.store_path, self.bot_uuid)

    @property
    def lock_path(self) -> Path:
        """Path of this namespace's sidecar lock file."""
        return schedules_lock_path(self.path)

    async def build_or_load(self) -> None:
        """Create the store directory, load the snapshot and recover ``firing`` records."""
        async with self._lock:
            await asyncio.to_thread(self._build_or_load_sync)

    async def add(self, action: ScheduledAction) -> ScheduledAction:
        """Persist a new action after enforcing the per-chat quota."""
        async with self._lock:
            return await asyncio.to_thread(self._add_sync, action)

    async def list(self, chat_id: int) -> list[ScheduledAction]:
        """Return one chat's records sorted by ``(run_at, id)``."""
        async with self._lock:
            return await asyncio.to_thread(self._list_sync, chat_id)

    async def delete(self, chat_id: int, action_id: str) -> bool:
        """Delete one action from one chat, returning whether it existed."""
        async with self._lock:
            return await asyncio.to_thread(self._delete_sync, chat_id, action_id)

    async def claim_due(self, now: datetime, limit: int = 10) -> list[ScheduledAction]:
        """Atomically claim up to ``limit`` due actions across every chat.

        The claim is written to disk before this method returns.
        """
        now = ensure_utc(now)
        async with self._lock:
            return await asyncio.to_thread(self._claim_due_sync, now, limit)

    async def claim_one(self, chat_id: int, action_id: str, now: datetime) -> ScheduledAction | None:
        """Atomically claim one specific action, or return None if not due."""
        now = ensure_utc(now)
        async with self._lock:
            return await asyncio.to_thread(self._claim_one_sync, chat_id, action_id, now)

    async def mark_delivered(self, action: ScheduledAction, now: datetime) -> None:
        """Finalize a successful delivery (remove one-shots, reactivate cron)."""
        async with self._lock:
            await asyncio.to_thread(self._mark_delivered_sync, action)

    async def record_failure(self, action: ScheduledAction, error: str, now: datetime) -> None:
        """Record a failed delivery and apply the retry/failure policy."""
        async with self._lock:
            await asyncio.to_thread(self._record_failure_sync, action, error)

    async def clear(self, chat_id: int) -> None:
        """Remove every scheduled action belonging to one chat."""
        async with self._lock:
            await asyncio.to_thread(self._clear_sync, chat_id)

    async def list_all(self) -> list[ScheduledAction]:
        """Return every action across chats, sorted by ``(chat_id, run_at, id)``."""
        async with self._lock:
            return await asyncio.to_thread(self._list_all_sync)

    # ------------------------------------------------------------------ sync internals

    def _build_or_load_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock():
            records = self._read_sync()
            changed = False
            for chat_actions in records.values():
                for action_id, action in list(chat_actions.items()):
                    recovered = apply_interrupted(action)
                    if recovered is not action:
                        chat_actions[action_id] = recovered
                        changed = True
            if changed:
                self._write_sync(records)

    def _add_sync(self, action: ScheduledAction) -> ScheduledAction:
        with self._file_lock():
            records = self._read_sync()
            chat_actions = records.setdefault(action.chat_id, {})
            ensure_chat_limit(list(chat_actions.values()), self.max_actions_per_chat)
            stored = action.clone()
            chat_actions[stored.id] = stored
            self._write_sync(records)
            return stored.clone()

    def _list_sync(self, chat_id: int) -> list[ScheduledAction]:
        with self._file_lock():
            records = self._read_sync()
        return self._sorted_clones(records.get(chat_id, {}))

    def _delete_sync(self, chat_id: int, action_id: str) -> bool:
        with self._file_lock():
            records = self._read_sync()
            chat_actions = records.get(chat_id)
            if chat_actions is None or action_id not in chat_actions:
                return False
            del chat_actions[action_id]
            if not chat_actions:
                del records[chat_id]
            self._write_sync(records)
            return True

    def _claim_due_sync(self, now: datetime, limit: int) -> list[ScheduledAction]:
        if limit <= 0:
            return []
        with self._file_lock():
            records = self._read_sync()
            candidates = [
                action
                for chat_actions in records.values()
                for action in chat_actions.values()
                if action.status == STATUS_ACTIVE and action.run_at <= now
            ]
            candidates.sort(key=lambda action: (action.run_at, action.chat_id, action.id))
            claimed: list[ScheduledAction] = []
            changed = False
            for action in candidates:
                if len(claimed) >= limit:
                    break
                replacement = apply_catchup_grace(action, now, self.default_timezone, self.catchup_grace_seconds)
                if replacement is not None:
                    records[action.chat_id][action.id] = replacement
                    changed = True
                    continue
                claimed_action = apply_claim(action, now, self.default_timezone)
                records[action.chat_id][action.id] = claimed_action
                claimed.append(claimed_action.clone())
                changed = True
            if changed:
                self._write_sync(records)
            return claimed

    def _claim_one_sync(self, chat_id: int, action_id: str, now: datetime) -> ScheduledAction | None:
        with self._file_lock():
            records = self._read_sync()
            chat_actions = records.get(chat_id)
            action = chat_actions.get(action_id) if chat_actions else None
            if action is None or action.status != STATUS_ACTIVE or action.run_at > now:
                return None
            replacement = apply_catchup_grace(action, now, self.default_timezone, self.catchup_grace_seconds)
            if replacement is not None:
                chat_actions[action_id] = replacement
                self._write_sync(records)
                return None
            claimed = apply_claim(action, now, self.default_timezone)
            chat_actions[action_id] = claimed
            self._write_sync(records)
            return claimed.clone()

    def _mark_delivered_sync(self, action: ScheduledAction) -> None:
        with self._file_lock():
            records = self._read_sync()
            chat_actions = records.get(action.chat_id)
            if chat_actions is None or action.id not in chat_actions:
                logger.warning(
                    "Scheduler delivery for unknown action %s (chat %s); ignoring",
                    action.id,
                    action.chat_id,
                )
                return
            updated = apply_delivered(action)
            if updated is None:
                del chat_actions[action.id]
                if not chat_actions:
                    del records[action.chat_id]
            else:
                chat_actions[action.id] = updated
            self._write_sync(records)

    def _record_failure_sync(self, action: ScheduledAction, error: str) -> None:
        with self._file_lock():
            records = self._read_sync()
            chat_actions = records.get(action.chat_id)
            if chat_actions is None or action.id not in chat_actions:
                logger.warning(
                    "Scheduler failure for unknown action %s (chat %s); ignoring",
                    action.id,
                    action.chat_id,
                )
                return
            updated = apply_failure(action, error, self.max_consecutive_failures)
            if updated is None:
                del chat_actions[action.id]
                if not chat_actions:
                    del records[action.chat_id]
            else:
                chat_actions[action.id] = updated
            self._write_sync(records)

    def _clear_sync(self, chat_id: int) -> None:
        with self._file_lock():
            records = self._read_sync()
            if chat_id not in records:
                return
            del records[chat_id]
            self._write_sync(records)

    def _list_all_sync(self) -> list[ScheduledAction]:
        with self._file_lock():
            records = self._read_sync()
        return [action.clone() for action in self._all_actions(records)]

    def _read_sync(self) -> dict[int, dict[str, ScheduledAction]]:
        """Load and parse the snapshot; missing or corrupt files read as empty."""
        if not self.path.is_file():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("Scheduler state at %s is unreadable; starting empty: %s", self.path, exc)
            return {}
        if not isinstance(payload, dict) or payload.get("version") != _FORMAT_VERSION:
            logger.warning("Scheduler state at %s has an unsupported format; starting empty", self.path)
            return {}
        entries = payload.get("actions")
        if not isinstance(entries, list):
            logger.warning("Scheduler state at %s has no action list; starting empty", self.path)
            return {}
        records: dict[int, dict[str, ScheduledAction]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                logger.warning("Skipping malformed scheduler record in %s", self.path)
                continue
            try:
                action = ScheduledAction.from_dict(entry)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed scheduler record in %s: %s", self.path, exc)
                continue
            records.setdefault(action.chat_id, {})[action.id] = action
        return records

    def _write_sync(self, records: dict[int, dict[str, ScheduledAction]]) -> None:
        """Atomically replace the snapshot with ``records``."""
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "actions": [action.to_dict() for action in self._all_actions(records)],
        }
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        """Hold an exclusive OS lock on the sidecar file for one operation."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _all_actions(records: dict[int, dict[str, ScheduledAction]]) -> list[ScheduledAction]:
        """Flatten every chat's records, sorted by ``(chat_id, run_at, id)``."""
        return sorted(
            (action for chat_actions in records.values() for action in chat_actions.values()),
            key=lambda action: (action.chat_id, action.run_at, action.id),
        )

    @staticmethod
    def _sorted_clones(chat_actions: dict[str, ScheduledAction]) -> list[ScheduledAction]:
        """Return clones of one chat's records sorted by ``(run_at, id)``."""
        return sorted(
            (action.clone() for action in chat_actions.values()),
            key=lambda action: (action.run_at, action.id),
        )
