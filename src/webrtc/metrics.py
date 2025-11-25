"""Drift metrics helpers for audio/video synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DriftMetrics:
    """Track latest audio/video timestamps and evaluate drift."""

    max_drift_s: float = 0.04
    last_video_ts: Optional[float] = None
    last_audio_ts: Optional[float] = None

    def record_video(self, pts: int, time_base) -> None:
        self.last_video_ts = float(pts) * float(time_base)

    def record_audio(self, pts: int, time_base) -> None:
        self.last_audio_ts = float(pts) * float(time_base)

    def drift_ok(self) -> bool:
        if self.last_video_ts is None or self.last_audio_ts is None:
            return True
        return abs(self.last_video_ts - self.last_audio_ts) <= self.max_drift_s

    def last_drift(self) -> Optional[float]:
        if self.last_video_ts is None or self.last_audio_ts is None:
            return None
        return abs(self.last_video_ts - self.last_audio_ts)
