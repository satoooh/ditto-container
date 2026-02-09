# Where: src/tests/test_handle_offer_timeout.py
# What: Verify /webrtc/offer still returns answer and pipeline cleanup happens on connection timeout (task 3.3).

import sys
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Lightweight stubs to avoid heavy deps
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


class FakePC:
    def __init__(self):
        self.handlers = {}
        self.connectionState = "new"
        self.localDescription = types.SimpleNamespace(sdp="ans", type="answer")
        self.closed = False

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
        self.closed = True


class _FakeAudioTrack:
    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate
    def finalize(self):
        return None


class _FakeVideoTrack:
    def __init__(self, *a, **k): ...
    def enqueue(self, *a, **k): ...


class _StreamSDKStub:
    def __init__(self, *a, **k): ...


for name, module in {
    "cv2": cv2_stub,
    "librosa": librosa_stub,
    "numpy": numpy_stub,
    "uvicorn": uvicorn_stub,
    "aiortc": types.SimpleNamespace(RTCPeerConnection=lambda *a, **k: FakePC(), RTCSessionDescription=lambda *a, **k: types.SimpleNamespace(sdp="ans", type="answer")),
    "aiortc.rtcconfiguration": rtc_config_stub,
    "aiortc.mediastreams": types.SimpleNamespace(MediaStreamTrack=type("MST", (), {})),
    "av": types.SimpleNamespace(VideoFrame=type("VF", (), {})),
    "stream_pipeline_online": types.SimpleNamespace(StreamSDK=_StreamSDKStub),
    "webrtc.tracks": types.SimpleNamespace(AudioArrayTrack=_FakeAudioTrack, VideoFrameTrack=_FakeVideoTrack),
    "core": types.SimpleNamespace(),
    "core.atomic_components": types.SimpleNamespace(),
    "core.models": types.SimpleNamespace(),
}.items():
    sys.modules.setdefault(name, module)

from streaming_server import StreamingServer


@pytest.fixture
def client(monkeypatch, tmp_path):
    async def _fail_wait(*args, **kwargs):
        return False

    app_server = StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data")
    # make sure pcs tracking is visible for assertions
    monkeypatch.setattr("streaming_server.RTCPeerConnection", lambda *a, **k: FakePC())
    monkeypatch.setattr("streaming_server.StreamingServer._wait_for_connection", _fail_wait, raising=False)
    return app_server, TestClient(app_server.app)


def test_handle_offer_cleans_up_on_timeout(tmp_path, client):
    server, api = client
    audio = tmp_path / "a.wav"
    img = tmp_path / "b.png"
    audio.write_bytes(b"a")
    img.write_bytes(b"b")

    payload = {
        "sdp": "v=0",
        "type": "offer",
        "audio_path": str(audio),
        "source_path": str(img),
    }

    resp = api.post("/webrtc/offer", json=payload)
    assert resp.status_code == 200
    for _ in range(20):
        if len(server._pcs) == 0:
            break
        time.sleep(0.01)
    assert len(server._pcs) == 0
