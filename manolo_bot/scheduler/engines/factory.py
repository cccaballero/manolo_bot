"""Engine selection from configuration values."""

from __future__ import annotations

from manolo_bot.scheduler.base import BaseSchedulerEngine
from manolo_bot.scheduler.engines.asyncio_engine import AsyncioSchedulerEngine
from manolo_bot.scheduler.engines.manual_engine import ManualSchedulerEngine

ASYNCIO_ENGINE_NAME = "asyncio"
MANUAL_ENGINE_NAME = "manual"

_ENGINE_NAMES = (ASYNCIO_ENGINE_NAME, MANUAL_ENGINE_NAME)


def build_scheduler_engine(name: str, *, interval_seconds: float = 5.0) -> BaseSchedulerEngine:
    """Instantiate the engine selected by ``name``.

    :param name: ``"asyncio"`` (background sweep loop) or ``"manual"``.
    :param interval_seconds: Sweep interval, used only by the asyncio engine.
    :raises ValueError: When ``name`` is unknown.
    """
    normalized = (name or "").strip().lower()
    if normalized == ASYNCIO_ENGINE_NAME:
        return AsyncioSchedulerEngine(interval_seconds=interval_seconds)
    if normalized == MANUAL_ENGINE_NAME:
        return ManualSchedulerEngine()
    expected = ", ".join(repr(candidate) for candidate in _ENGINE_NAMES)
    raise ValueError(f"Unknown scheduler engine {name!r}; expected one of {expected}")


__all__ = ["ASYNCIO_ENGINE_NAME", "MANUAL_ENGINE_NAME", "build_scheduler_engine"]
