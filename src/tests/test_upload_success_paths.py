# Where: src/tests/test_upload_success_paths.py
# What: Verify /upload returns usable paths and files are written (task 2.3).

import os
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


def _client(tmpdir: Path) -> TestClient:
    os.environ["UPLOAD_DIR"] = str(tmpdir)
    app = StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data").app
    return TestClient(app)


def test_upload_audio_only_returns_path(tmp_path):
    client = _client(tmp_path)
    fd, path = tempfile.mkstemp(suffix=".wav")
    Path(path).write_bytes(b"dummy")

    with open(path, "rb") as f:
        resp = client.post("/upload", files={"audio": ("a.wav", f, "audio/wav")})

    assert resp.status_code == 200
    data = resp.json()
    assert "audio_path" in data and data["audio_path"].endswith("a.wav")
    assert Path(data["audio_path"]).is_file()


def test_upload_image_only_returns_path(tmp_path):
    client = _client(tmp_path)
    fd, path = tempfile.mkstemp(suffix=".png")
    Path(path).write_bytes(b"img")

    with open(path, "rb") as f:
        resp = client.post("/upload", files={"source": ("b.png", f, "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert "source_path" in data and data["source_path"].endswith("b.png")
    assert Path(data["source_path"]).is_file()


def test_upload_both_returns_both_paths(tmp_path):
    client = _client(tmp_path)
    fa, pa = tempfile.mkstemp(suffix=".wav")
    fi, pi = tempfile.mkstemp(suffix=".jpeg")
    Path(pa).write_bytes(b"aud")
    Path(pi).write_bytes(b"img")

    with open(pa, "rb") as fa_f, open(pi, "rb") as fi_f:
        resp = client.post(
            "/upload",
            files={
                "audio": ("c.wav", fa_f, "audio/wav"),
                "source": ("d.jpeg", fi_f, "image/jpeg"),
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"audio_path", "source_path"}
    assert Path(data["audio_path"]).is_file()
    assert Path(data["source_path"]).is_file()

