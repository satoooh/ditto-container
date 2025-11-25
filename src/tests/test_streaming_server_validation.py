# Where: src/tests/test_streaming_server_validation.py
# What: Ensure StreamingServer uses OfferValidator before pipeline startup.
# Why: Task 1.3 requirement to apply validation and 4xx without resource allocation.

import sys
import tempfile
from pathlib import Path

import pytest

# Skip if heavy deps absent
pytest.importorskip("uvicorn")
pytest.importorskip("aiortc")

# Stub heavy deps before importing streaming_server
import types

cv2_stub = types.SimpleNamespace(
    cvtColor=lambda *a, **k: a[0],
    COLOR_RGB2BGR=0,
    resize=lambda img, size, interpolation=None: img,
)
librosa_stub = types.SimpleNamespace(
    load=lambda path, sr=16000: ([0.0] * sr, sr),
    resample=lambda audio, orig_sr, target_sr: audio,
)
numpy_stub = types.SimpleNamespace(
    zeros=lambda shape, dtype=None: [0.0] * (shape if isinstance(shape, int) else shape[0]),
    float32=float,
)
sys.modules.setdefault("cv2", cv2_stub)
sys.modules.setdefault("librosa", librosa_stub)
sys.modules.setdefault("numpy", numpy_stub)

from streaming_server import StreamingServer


def _temp_file(suffix: str) -> Path:
    fd, path_str = tempfile.mkstemp(suffix=suffix)
    Path(path_str).write_bytes(b"test")
    Path(path_str).chmod(0o644)
    return Path(path_str)


def _server():
    # cfg/data paths are placeholders; validation should not touch them.
    return StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data")


def test_validate_offer_happy_path():
    server = _server()
    audio = _temp_file(".wav")
    image = _temp_file(".png")

    payload = {
        "sdp": "v=0",
        "type": "offer",
        "audio_path": str(audio),
        "source_path": str(image),
        "setup_kwargs": {"sampling_timesteps": 15},
        "run_kwargs": {"frame_scale": 0.4, "chunk_config": [2, 3, 2]},
    }

    validated = server._validate_offer(payload)

    assert validated.frame_scale == 0.4
    assert validated.sampling_timesteps == 15
    assert validated.chunk_config == (2, 3, 2)


def test_validate_offer_rejects_bad_audio_ext():
    server = _server()
    audio = _temp_file(".txt")
    image = _temp_file(".png")

    payload = {
        "sdp": "v=0",
        "type": "offer",
        "audio_path": str(audio),
        "source_path": str(image),
    }

    with pytest.raises(Exception):
        server._validate_offer(payload)
