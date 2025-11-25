# Where: src/tests/test_offer_exception_cleanup.py
# What: Integration-ish test to ensure /webrtc/offer returns 500 and cleans up when StreamSDK.setup raises (task 4.3).

import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Stubs to avoid heavy deps
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
    ndarray=list,
)
uvicorn_stub = types.SimpleNamespace(run=lambda *a, **k: None)
rtc_config_stub = types.SimpleNamespace(RTCConfiguration=lambda *a, **k: None, RTCIceServer=lambda *a, **k: None)
mediastreams_stub = types.SimpleNamespace(MediaStreamTrack=type("MST", (), {}))
av_stub = types.SimpleNamespace(VideoFrame=type("VF", (), {}))
class _FakeAudioTrack:
    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate
    def finalize(self):
        return None

class _FakeVideoTrack:
    def __init__(self, *a, **k): ...
    def enqueue(self, *a, **k): ...

for name, module in {
    "cv2": cv2_stub,
    "librosa": librosa_stub,
    "numpy": numpy_stub,
    "uvicorn": uvicorn_stub,
    "aiortc": types.SimpleNamespace(RTCPeerConnection=lambda *a, **k: FakePC(), RTCSessionDescription=lambda *a, **k: types.SimpleNamespace(sdp="ans", type="answer")),
    "aiortc.rtcconfiguration": rtc_config_stub,
    "aiortc.mediastreams": mediastreams_stub,
    "av": av_stub,
    "core": types.SimpleNamespace(),
    "core.atomic_components": types.SimpleNamespace(audio2motion=None),
    "core.models": types.SimpleNamespace(),
    "stream_pipeline_online": types.SimpleNamespace(StreamSDK=None),  # patched later
    "webrtc.tracks": types.SimpleNamespace(AudioArrayTrack=_FakeAudioTrack, VideoFrameTrack=_FakeVideoTrack),
}.items():
    sys.modules.setdefault(name, module)

from streaming_server import StreamingServer


class FakePC:
    def __init__(self):
        self.handlers = {}
        self.connectionState = "connected"
        self.localDescription = types.SimpleNamespace(sdp="ans", type="answer")

    def on(self, event):
        def deco(fn):
            self.handlers[event] = fn
            return fn

        return deco

    async def setRemoteDescription(self, *a, **k):
        return None

    async def createAnswer(self):
        return self.localDescription

    async def setLocalDescription(self, *a, **k):
        return None

    def addTrack(self, *a, **k):
        return None

    async def close(self):
        return None


class BoomSDK:
    def __init__(self, *a, **k):
        self.writer = None

    def setup(self, *a, **k):
        raise RuntimeError("boom")

    def setup_Nd(self, *a, **k):
        return None

    def close(self):
        return None


def test_offer_returns_500_on_sdk_setup_error(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    img = tmp_path / "b.png"
    audio.write_bytes(b"a")
    img.write_bytes(b"b")

    # Fast-fail connection wait
    async def _fail_wait(*args, **kwargs):
        return False

    server = StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data")
    monkeypatch.setattr("streaming_server.StreamSDK", BoomSDK)
    monkeypatch.setattr(server, "_wait_for_connection", _fail_wait, raising=False)
    api = TestClient(server.app)

    resp = api.post(
        "/webrtc/offer",
        json={
            "sdp": "v=0",
            "type": "offer",
            "audio_path": str(audio),
            "source_path": str(img),
        },
    )

    assert resp.status_code == 502
    assert "peer connection failed" in resp.json()["detail"]
    assert len(server._pcs) == 0
