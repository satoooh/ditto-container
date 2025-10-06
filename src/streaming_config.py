# streaming_config.py — shared helpers for streaming parameter parsing and validation.
# Provides utilities to normalize chunk configuration strings and enforce safety limits.

from __future__ import annotations

from typing import Sequence, Tuple


def parse_chunk_config(
    raw: Sequence[int] | str, *, fallback: Tuple[int, int, int] = (3, 5, 2)
) -> Tuple[int, int, int]:
    """Parse chunk configuration input into a (pre, main, post) tuple.

    Accepts a comma-separated string like "3,5,2" or any iterable of three integers.
    Values are clamped to at least 1 to avoid zero-length buffers.
    """

    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = list(raw)

    if len(parts) != 3:
        return fallback

    try:
        pre, main, post = (max(1, int(value)) for value in parts)
    except (TypeError, ValueError):
        return fallback

    return pre, main, post


def clamp_sampling_timesteps(
    value: int | None, *, default: int = 30, minimum: int = 5, maximum: int = 100
) -> int:
    """Ensure sampling timesteps stay within an acceptable range."""

    if value is None:
        return default
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, numeric))


def to_chunk_list(chunks: Tuple[int, int, int]) -> list[int]:
    """Convert chunk tuple to list for JSON serialization."""

    return [int(value) for value in chunks]
