# Where: src/tests/test_drift_metrics.py
# What: Unit tests for DriftMetrics (task 4.2).

from webrtc.metrics import DriftMetrics


def test_drift_ok_within_threshold():
    metrics = DriftMetrics(max_drift_s=0.04)
    metrics.record_video(pts=0, time_base=1 / 30)  # 0s
    metrics.record_audio(pts=960, time_base=1 / 48000)  # 0.02s
    assert metrics.drift_ok() is True
    assert metrics.last_drift() == 0.02


def test_drift_exceeds_threshold():
    metrics = DriftMetrics(max_drift_s=0.04)
    metrics.record_video(pts=0, time_base=1 / 30)  # 0s
    metrics.record_audio(pts=4000, time_base=1 / 48000)  # ~0.083s
    assert metrics.drift_ok() is False
    assert metrics.last_drift() > 0.04
