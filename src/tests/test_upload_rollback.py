# Where: src/tests/test_upload_rollback.py
# What: Ensure upload rolls back on write failure (task 2.2).

import sys
import os
import tempfile
from pathlib import Path
from unittest import mock

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
rtc_config_stub = types.SimpleNamespace(RTCConfiguration=lambda *a, **k: None, RTCIceServer=lambda *a, **k: None)
aiortc_stub = types.SimpleNamespace(
    RTCPeerConnection=type("PC", (), {}),
    RTCSessionDescription=lambda *a, **k: None,
)
mediastreams_stub = types.SimpleNamespace(MediaStreamTrack=type("MST", (), {}))
av_stub = types.SimpleNamespace(VideoFrame=type("VF", (), {}))
class _StreamSDKStub:
    def __init__(self, *a, **k): ...

for name, module in {
    "cv2": cv2_stub,
    "librosa": librosa_stub,
    "numpy": numpy_stub,
    "uvicorn": uvicorn_stub,
    "aiortc": aiortc_stub,
    "aiortc.rtcconfiguration": rtc_config_stub,
    "aiortc.mediastreams": mediastreams_stub,
    "av": av_stub,
    "stream_pipeline_online": types.SimpleNamespace(StreamSDK=_StreamSDKStub),
    "core": types.SimpleNamespace(),
    "core.atomic_components": types.SimpleNamespace(),
    "core.models": types.SimpleNamespace(),
}.items():
    sys.modules.setdefault(name, module)

from fastapi.testclient import TestClient
from streaming_server import StreamingServer


def _server(tmpdir: Path):
    return StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data").app


def test_upload_rollback_on_write_error(tmp_path):
    app = _server(tmp_path)
    client = TestClient(app)

    fd, path = tempfile.mkstemp(suffix=".wav")
    Path(path).write_bytes(b"dummy")
    dest_dir = tmp_path / "uploads"
    dest_dir.mkdir()

    # Force destination.open to raise for our path
    def _fake_open(self, mode="r", *args, **kwargs):
        if self.parent == dest_dir:
            raise PermissionError("no write")
        return original_open(self, mode, *args, **kwargs)

    original_open = Path.open
    try:
        with mock.patch("pathlib.Path.open", _fake_open), mock.patch(
            "streaming_server.os.getenv", return_value=str(dest_dir)
        ):
            with open(path, "rb") as f:
                response = client.post("/upload", files={"audio": ("a.wav", f, "audio/wav")})

        assert response.status_code == 500
        assert not list(dest_dir.iterdir())
    finally:
        os.chmod(dest_dir, 0o700)
        os.environ.pop("UPLOAD_DIR", None)
