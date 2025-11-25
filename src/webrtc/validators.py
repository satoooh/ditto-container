"""Payload validation helpers for /webrtc/offer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from streaming_config import parse_chunk_config
from webrtc.parameters import ParameterNormalizer


_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class ValidatedOffer:
    sdp: str
    type: str
    audio_path: str
    source_path: str
    sampling_timesteps: int | None
    frame_scale: float
    chunk_config: Tuple[int, int, int]
    chunk_sleep_s: float | None
    warnings: List[str] = field(default_factory=list)


class OfferValidator:
    """Validate and normalize /webrtc/offer payloads."""

    def __init__(self) -> None:
        self.audio_ext = _AUDIO_EXT
        self.image_ext = _IMAGE_EXT
        self.normalizer = ParameterNormalizer()

    def validate(self, payload: Dict[str, Any]) -> ValidatedOffer:
        warnings: List[str] = []

        # Required top-level fields
        sdp = payload.get("sdp")
        typ = payload.get("type")
        audio_path = payload.get("audio_path")
        source_path = payload.get("source_path")
        if not sdp or not typ or not audio_path or not source_path:
            raise ValueError("Missing required fields: sdp, type, audio_path, source_path")

        self._check_path(audio_path, self.audio_ext, "audio")
        self._check_path(source_path, self.image_ext, "source")

        setup_kwargs = payload.get("setup_kwargs") or {}
        run_kwargs = payload.get("run_kwargs") or {}

        normalized = self.normalizer.normalize(
            {
                "frame_scale": run_kwargs.get("frame_scale"),
                "sampling_timesteps": setup_kwargs.get("sampling_timesteps"),
                "chunk_config": run_kwargs.get("chunk_config")
                or run_kwargs.get("chunksize")
                or run_kwargs.get("chunk_cfg"),
                "chunk_sleep_s": run_kwargs.get("chunk_sleep_s"),
                "chunk_sleep_ms": run_kwargs.get("chunk_sleep_ms"),
            }
        )
        warnings.extend(normalized.warnings)

        return ValidatedOffer(
            sdp=str(sdp),
            type=str(typ),
            audio_path=str(audio_path),
            source_path=str(source_path),
            sampling_timesteps=normalized.sampling_timesteps,
            frame_scale=normalized.frame_scale,
            chunk_config=normalized.chunk_config,
            chunk_sleep_s=normalized.chunk_sleep_s,
            warnings=warnings,
        )

    def _check_path(self, path_str: str, allowed_exts: set[str], label: str) -> None:
        path = Path(path_str)
        if path.suffix.lower() not in allowed_exts:
            raise ValueError(f"Unsupported {label} extension: {path.suffix}")
        if not path.is_file():
            raise ValueError(f"{label} file not found: {path}")
