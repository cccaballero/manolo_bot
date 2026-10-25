"""Canonical keys for the storage layer.

Storage keys identify a conversation, and they are not necessarily numeric
Telegram chat ids: a library consumer may key history by ``"C012345"`` (Slack),
by a shared agent key, or by ``"dashboard:{user}:{agent}:{uuid}"``. Keys are
therefore opaque, non-empty strings; ``42`` and ``"42"`` address the same
records.

Path-based backends (file documents and the deep-agent local filesystem) need a
safe single path segment. Keys that are already safe are used verbatim, so
existing numeric chat directories keep working with no migration; every other
key maps to a stable ``h.<sha256>`` segment, which is traversal-free, length
bounded and safe on case-insensitive filesystems.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import quote

#: Keys matching this need no transformation: no separators, no dots (so no
#: ``.``/``..``), no case to collide on a case-insensitive filesystem, and short
#: enough for any filesystem's per-component limit.
_SAFE_SEGMENT = re.compile(r"^[a-z0-9_-]{1,64}$")

#: Windows reserved device names cannot be directories even though they match
#: :data:`_SAFE_SEGMENT`; they are hashed like any other unsafe key.
_RESERVED_SEGMENTS = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{index}" for index in range(1, 10)} | {f"lpt{index}" for index in range(1, 10)}
)

#: Hashed segments are prefixed so they can never collide with a safe key
#: (``.`` is not an allowed character in :data:`_SAFE_SEGMENT`).
_HASH_PREFIX = "h."

#: SHA-256 hex prefix; 40 hex chars is far below the 255-byte filename limit.
_HASH_LENGTH = 40


def normalize_storage_key(value: str | int | None) -> str:
    """Return the canonical storage key for ``value``.

    A numeric Telegram chat id becomes its decimal representation, so ``42`` and
    ``"42"`` address the same records, and surrounding whitespace is stripped so
    ``" 42 "`` does too. ``None`` and blank values are rejected because they
    cannot identify a conversation.
    """
    if value is None:
        raise ValueError("Storage key must not be empty.")
    key = str(value).strip()
    if not key:
        raise ValueError("Storage key must not be empty.")
    return key


def escape_key_component(value: str) -> str:
    """Percent-encode ``value`` for one component of a colon-delimited key.

    Storage keys are opaque and may contain ``:``, while Redis keys are built by
    joining components with ``:``. Without escaping, one component sequence
    could impersonate another — e.g. a message key ``documents:42`` would
    address chat ``42``'s document set. Percent-encoding keeps components
    unambiguous while leaving numeric chat ids and sanitized filenames
    byte-identical, so existing numeric keys need no migration.
    """
    return quote(value, safe="")


def storage_path_segment(value: str | int) -> str:
    """Return a safe single path segment for the storage key ``value``.

    Safe keys (``[a-z0-9_-]+``, at most 64 characters) are returned unchanged so
    existing directories such as ``42`` or ``-1001234567890`` keep resolving.
    Any other key — uppercase, punctuation, separators, traversal attempts,
    over-long values or Windows reserved device names — is replaced by a
    deterministic ``h.<sha256>`` segment.
    """
    key = normalize_storage_key(value)
    if _SAFE_SEGMENT.fullmatch(key) and key not in _RESERVED_SEGMENTS:
        return key
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    return f"{_HASH_PREFIX}{digest}"


__all__ = ["escape_key_component", "normalize_storage_key", "storage_path_segment"]
