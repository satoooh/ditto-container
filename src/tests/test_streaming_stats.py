# Where: pytest suite for streaming client metrics.
# What: Validates decode timing, latency sampling, and hardware-metric fallbacks.
# Why: Guard against regressions in telemetry output adjustments.

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streaming_client  # noqa: E402  # isort:skip


def _make_test_frame() -> bytes:
    # Use a deterministic pattern to keep encode cost low but non-trivial.
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = [255, 0, 0]
    image[:, 16:] = [0, 255, 0]
    success, encoded = streaming_client.cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_streaming_stats_records_decode_and_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure hardware metrics do not rely on actual system dependencies during tests.
    monkeypatch.setattr(streaming_client, "psutil", None)
    monkeypatch.setattr(streaming_client.shutil, "which", lambda _: None)

    cpu_samples = [
        {"overall": (100.0, 40.0), "per_cpu": [(50.0, 20.0), (50.0, 20.0)]},
        {"overall": (250.0, 70.0), "per_cpu": [(120.0, 30.0), (130.0, 40.0)]},
    ]
    net_samples = [(1_000, 2_000), (6_000, 9_000)]
    cpu_calls = {"count": 0}
    net_calls = {"count": 0}

    def fake_cpu_totals(self):  # type: ignore[override]
        idx = min(cpu_calls["count"], len(cpu_samples) - 1)
        cpu_calls["count"] += 1
        return cpu_samples[idx]

    def fake_net_totals(self):  # type: ignore[override]
        idx = min(net_calls["count"], len(net_samples) - 1)
        net_calls["count"] += 1
        return net_samples[idx]

    monkeypatch.setattr(streaming_client.StreamingStats, "_read_proc_cpu_totals", fake_cpu_totals)
    monkeypatch.setattr(streaming_client.StreamingStats, "_read_proc_net_counters", fake_net_totals)

    memory_stub = {
        "available": True,
        "total_gb": 32.0,
        "used_gb": 12.0,
        "percent": 37.5,
        "source": "stub",
    }
    monkeypatch.setattr(streaming_client.StreamingStats, "_memory_metrics_procfs", lambda self: memory_stub)

    client = streaming_client.StreamingClient("ws://example", "client")
    client.stats.mark_streaming_start()

    frame_bytes = _make_test_frame()

    original_imdecode: Callable = streaming_client.cv2.imdecode

    def traced_imdecode(buffer, flags):
        time.sleep(0.002)
        return original_imdecode(buffer, flags)

    monkeypatch.setattr(streaming_client.cv2, "imdecode", traced_imdecode)

    client._record_frame(frame_id=1, timestamp=time.time(), frame_bytes=frame_bytes)
    client.stats.record_latency(0.05)
    client.stats.streaming_start_time -= 10  # simulate longer run for throughput

    stats = client.stats.get_stats()

    assert stats["total_frames"] == 1
    assert stats["decode_times"]["max"] > 0
    assert stats["latency"]["count"] == 1
    assert stats["latency"]["mean"] == pytest.approx(0.05)
    assert stats["system_metrics"]["cpu"]["available"] is True
    assert stats["system_metrics"]["cpu"]["source"] == "procfs"
    assert stats["system_metrics"]["memory"] == memory_stub
    assert stats["system_metrics"]["network"]["available"] is True
    assert stats["system_metrics"]["network"]["bytes_recv"] == 5_000
    assert stats["system_metrics"]["network"]["bytes_sent"] == 7_000
    assert stats["system_metrics"]["gpu"] == []
