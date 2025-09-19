# File: src/server_paths.py
# Purpose: Share TensorRT engine path resolution logic across runtime and tests.
# Why: Decouple lightweight path handling from heavy streaming_server imports.

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT_CANDIDATES: tuple[str, ...] = (
    "/app/checkpoints/ditto_trt_universal",
    "/app/checkpoints/ditto_trt_Ampere_Plus",
)


def _normalized(path: str) -> str:
    return os.path.abspath(path)


def resolve_data_root(requested_path: Optional[str]) -> str:
    """Pick the first available TensorRT engine directory, preferring the universal build."""
    candidates: List[str] = []

    if requested_path:
        candidates.append(_normalized(requested_path))

    for candidate in DEFAULT_DATA_ROOT_CANDIDATES:
        abs_candidate = _normalized(candidate)
        if abs_candidate not in candidates:
            candidates.append(abs_candidate)

    for candidate in candidates:
        if os.path.isdir(candidate):
            logger.info("Resolved TensorRT engine directory: %s", candidate)
            return candidate

    raise FileNotFoundError(
        "No TensorRT engine directory found. Checked: " + ", ".join(candidates)
    )
