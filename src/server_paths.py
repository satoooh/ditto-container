# File: src/server_paths.py
# Purpose: Share TensorRT engine path resolution logic across runtime and tests.
# Why: Decouple lightweight path handling from heavy streaming_server imports.

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# (path, min_major, max_major)
DEFAULT_DATA_ROOT_CANDIDATES: Tuple[Tuple[str, Optional[int], Optional[int]], ...] = (
    ("/app/checkpoints/ditto_trt_blackwell", 12, None),
    ("/app/checkpoints/ditto_trt_Ampere_Plus", 8, 11),
)


def _normalized(path: str) -> str:
    return os.path.abspath(path)


def _current_cc() -> Optional[Tuple[int, int]]:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover - advisory only
        logger.debug("Failed to query CUDA device capability", exc_info=True)
    return None


def _iter_candidates(requested_path: Optional[str]) -> Sequence[Tuple[str, Optional[int], Optional[int]]]:
    seen: set[str] = set()
    ordered: List[Tuple[str, Optional[int], Optional[int]]] = []

    if requested_path:
        resolved = _normalized(requested_path)
        ordered.append((resolved, None, None))
        seen.add(resolved)

    for path, min_major, max_major in DEFAULT_DATA_ROOT_CANDIDATES:
        normalized = _normalized(path)
        if normalized not in seen:
            ordered.append((normalized, min_major, max_major))
            seen.add(normalized)

    return ordered


def resolve_data_root(requested_path: Optional[str]) -> str:
    """Pick the best TensorRT engine directory for the current GPU."""
    cc = _current_cc()
    candidates = _iter_candidates(requested_path)

    def _compatible(entry: Tuple[str, Optional[int], Optional[int]]) -> bool:
        path, min_major, max_major = entry
        if not os.path.isdir(path):
            return False
        if cc is None or min_major is None:
            return True
        major, _ = cc
        if major < min_major:
            return False
        if max_major is not None and major > max_major:
            return False
        return True

    for entry in candidates:
        if _compatible(entry):
            logger.info("Resolved TensorRT engine directory: %s", entry[0])
            return entry[0]

    fallback_paths = [path for path, _, _ in candidates if os.path.isdir(path)]
    if fallback_paths:
        chosen = fallback_paths[0]
        logger.warning(
            "No engine matched compute capability %s; falling back to %s", cc, chosen
        )
        return chosen

    checked = [_normalized(path) for path, _, _ in candidates]
    raise FileNotFoundError(
        "No TensorRT engine directory found. Checked: " + ", ".join(checked)
    )
