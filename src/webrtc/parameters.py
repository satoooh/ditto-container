"""Parameter normalization utilities for streaming sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from streaming_config import (
    clamp_sampling_timesteps,
    clamp_scale,
    parse_chunk_config,
)


@dataclass
class NormalizedParams:
    sampling_timesteps: int
    frame_scale: float
    chunk_config: Tuple[int, int, int]
    chunk_sleep_s: float | None
    warnings: List[str] = field(default_factory=list)


class ParameterNormalizer:
    """Normalize and clamp incoming session parameters."""

    def __init__(self) -> None:
        self.default_sampling = clamp_sampling_timesteps(None)
        self.default_scale = clamp_scale(None)
        self.default_chunk = (3, 5, 2)

    def normalize(self, params: Dict[str, Any]) -> NormalizedParams:
        warnings: List[str] = []
        frame_scale = self._coerce_scale(params.get("frame_scale"), warnings)
        sampling = self._coerce_sampling(params.get("sampling_timesteps"), warnings)

        raw_chunk = params.get("chunk_config") or params.get("chunksize")
        if raw_chunk is None:
            chunk_config = self.default_chunk
        else:
            chunk_config = parse_chunk_config(raw_chunk, fallback=self.default_chunk)
            if chunk_config == self.default_chunk and raw_chunk not in (
                self.default_chunk,
                list(self.default_chunk),
                "3,5,2",
            ):
                warnings.append("chunk_config invalid; fallback to (3,5,2)")

        chunk_sleep_s = self._coerce_chunk_sleep(params, warnings)

        return NormalizedParams(
            sampling_timesteps=sampling,
            frame_scale=frame_scale,
            chunk_config=chunk_config,
            chunk_sleep_s=chunk_sleep_s,
            warnings=warnings,
        )

    def _coerce_scale(self, value: Any, warnings: List[str]) -> float:
        if value is None:
            return self.default_scale
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            warnings.append("frame_scale invalid; using default")
            return self.default_scale
        return clamp_scale(numeric)

    def _coerce_sampling(self, value: Any, warnings: List[str]) -> int:
        if value is None:
            return self.default_sampling
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            warnings.append("sampling_timesteps invalid; using default")
            return self.default_sampling
        return clamp_sampling_timesteps(numeric)

    def _coerce_chunk_sleep(self, params: Dict[str, Any], warnings: List[str]) -> float | None:
        if params.get("chunk_sleep_s") is not None:
            try:
                return max(0.0, float(params["chunk_sleep_s"]))
            except (TypeError, ValueError):
                warnings.append("chunk_sleep_s invalid; ignoring")
                return None
        if params.get("chunk_sleep_ms") is not None:
            try:
                return max(0.0, float(params["chunk_sleep_ms"]) / 1000.0)
            except (TypeError, ValueError):
                warnings.append("chunk_sleep_ms invalid; ignoring")
                return None
        return None
