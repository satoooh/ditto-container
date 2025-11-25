# Where: src/tests/test_upload_validation.py
# What: Validate /upload returns 400 for missing or bad extensions (task 2.1).

import sys
import tempfile
from pathlib import Path

import pytest

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
uvicorn_stub = types.SimpleNamespace(run=lambda *a, **k: None)
aiortc_stub = types.SimpleNamespace(
    RTCPeerConnection=type("PC", (), {}) ,
    RTCSessionDescription=lambda *a, **k: None,
)
rtc_config_stub = types.SimpleNamespace(RTCConfiguration=lambda *a, **k: None, RTCIceServer=lambda *a, **k: None)
mediastreams_stub = types.SimpleNamespace(MediaStreamTrack=type("MST", (), {}))
av_stub = types.SimpleNamespace(VideoFrame=type("VF", (), {}))
class _StreamSDKStub:
    def __init__(self, *a, **k): ...
sys.modules.setdefault("stream_pipeline_online", types.SimpleNamespace(StreamSDK=_StreamSDKStub))
# Stub deep model imports to avoid torch dependency
sys.modules.setdefault("core", types.SimpleNamespace())
sys.modules.setdefault("core.atomic_components", types.SimpleNamespace())
sys.modules.setdefault("core.models", types.SimpleNamespace())

for name, module in {
    "cv2": cv2_stub,
    "librosa": librosa_stub,
    "numpy": numpy_stub,
    "uvicorn": uvicorn_stub,
    "aiortc": aiortc_stub,
    "aiortc.rtcconfiguration": rtc_config_stub,
    "aiortc.mediastreams": mediastreams_stub,
    "av": av_stub,
}.items():
    sys.modules.setdefault(name, module)

from streaming_server import StreamingServer
from fastapi.testclient import TestClient


def _server():
    server = StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data")
    return server.app


def test_upload_rejects_unsupported_extension():
    app = _server()
    client = TestClient(app)

    fd, path = tempfile.mkstemp(suffix=".txt")
    Path(path).write_bytes(b"dummy")
    with open(path, "rb") as f:
        response = client.post("/upload", files={"audio": ("bad.txt", f, "text/plain")})

    assert response.status_code == 400
    assert "Unsupported extension" in response.json()["detail"]


def test_upload_requires_at_least_one_file():
    app = _server()
    client = TestClient(app)

    response = client.post("/upload", files={})

    assert response.status_code == 400
    assert "no files provided" in response.json()["detail"]
