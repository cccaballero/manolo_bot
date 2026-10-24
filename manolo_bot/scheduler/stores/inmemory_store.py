"""Ephemeral in-memory scheduler store."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

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


class InMemorySchedulerStore(BaseSchedulerStore):
    """Scheduler store kept in process memory.

    State is lost on restart, so this backend is intended for tests and
    single-process debugging. An :class:`asyncio.Lock` makes claims atomic
    within the process, and every public read returns clones so callers can
    never mutate internal records by accident.
    """

    def __init__(
        self,
        bot_uuid: str = "default",
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
        self._actions: dict[int, dict[str, ScheduledAction]] = {}
        self._lock = asyncio.Lock()

    async def build_or_load(self) -> None:
        """Recover records left in ``firing``; safe to call more than once."""
        async with self._lock:
            for chat_actions in self._actions.values():
                for action_id, action in list(chat_actions.items()):
                    chat_actions[action_id] = apply_interrupted(action)

    async def add(self, action: ScheduledAction) -> ScheduledAction:
        """Persist a new action after enforcing the per-chat quota."""
        async with self._lock:
            chat_actions = self._actions.setdefault(action.chat_id, {})
            ensure_chat_limit(list(chat_actions.values()), self.max_actions_per_chat)
            stored = action.clone()
            chat_actions[stored.id] = stored
            return stored.clone()

    async def list(self, chat_id: int) -> list[ScheduledAction]:
        """Return one chat's records sorted by ``(run_at, id)``."""
        async with self._lock:
            return self._sorted_clones(self._actions.get(chat_id, {}))

    async def delete(self, chat_id: int, action_id: str) -> bool:
        """Delete one action from one chat, returning whether it existed."""
        async with self._lock:
            chat_actions = self._actions.get(chat_id)
            if chat_actions is None or action_id not in chat_actions:
                return False
            del chat_actions[action_id]
            if not chat_actions:
                del self._actions[chat_id]
            return True

    async def claim_due(self, now: datetime, limit: int = 10) -> list[ScheduledAction]:
        """Atomically claim up to ``limit`` due actions across every chat."""
        now = ensure_utc(now)
        async with self._lock:
            if limit <= 0:
                return []
            candidates = [
                action
                for chat_actions in self._actions.values()
                for action in chat_actions.values()
                if action.status == STATUS_ACTIVE and action.run_at <= now
            ]
            candidates.sort(key=lambda action: (action.run_at, action.chat_id, action.id))
            claimed: list[ScheduledAction] = []
            for action in candidates:
                if len(claimed) >= limit:
                    break
                replacement = apply_catchup_grace(action, now, self.default_timezone, self.catchup_grace_seconds)
                if replacement is not None:
                    self._actions[action.chat_id][action.id] = replacement
                    continue
                claimed_action = apply_claim(action, now, self.default_timezone)
                self._actions[action.chat_id][action.id] = claimed_action
                claimed.append(claimed_action.clone())
            return claimed

    async def claim_one(self, chat_id: int, action_id: str, now: datetime) -> ScheduledAction | None:
        """Atomically claim one specific action, or return None if not due."""
        now = ensure_utc(now)
        async with self._lock:
            action = self._actions.get(chat_id, {}).get(action_id)
            if action is None or action.status != STATUS_ACTIVE or action.run_at > now:
                return None
            replacement = apply_catchup_grace(action, now, self.default_timezone, self.catchup_grace_seconds)
            if replacement is not None:
                self._actions[chat_id][action_id] = replacement
                return None
            claimed = apply_claim(action, now, self.default_timezone)
            self._actions[chat_id][action_id] = claimed
            return claimed.clone()

    async def mark_delivered(self, action: ScheduledAction, now: datetime) -> None:
        """Finalize a successful delivery (remove one-shots, reactivate cron)."""
        async with self._lock:
            chat_actions = self._actions.get(action.chat_id)
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
                    del self._actions[action.chat_id]
            else:
                chat_actions[action.id] = updated

    async def record_failure(self, action: ScheduledAction, error: str, now: datetime) -> None:
        """Record a failed delivery and apply the retry/failure policy."""
        async with self._lock:
            chat_actions = self._actions.get(action.chat_id)
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
                    del self._actions[action.chat_id]
            else:
                chat_actions[action.id] = updated

    async def clear(self, chat_id: int) -> None:
        """Remove every scheduled action belonging to one chat."""
        async with self._lock:
            self._actions.pop(chat_id, None)

    async def list_all(self) -> list[ScheduledAction]:
        """Return every action across chats, sorted by ``(chat_id, run_at, id)``."""
        async with self._lock:
            return sorted(
                (action.clone() for chat_actions in self._actions.values() for action in chat_actions.values()),
                key=lambda action: (action.chat_id, action.run_at, action.id),
            )

    @staticmethod
    def _sorted_clones(chat_actions: dict[str, ScheduledAction]) -> list[ScheduledAction]:
        """Return clones of one chat's records sorted by ``(run_at, id)``."""
        return sorted(
            (action.clone() for action in chat_actions.values()),
            key=lambda action: (action.run_at, action.id),
        )
