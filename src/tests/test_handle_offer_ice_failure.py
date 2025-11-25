# Where: src/tests/test_handle_offer_ice_failure.py
# What: Ensure /webrtc/offer returns 502 when peer connection fails early (task 3.2).

import sys
import types

import pytest
from fastapi.testclient import TestClient

# Stub heavy deps
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
aiortc_stub = types.SimpleNamespace(
    RTCPeerConnection=lambda *a, **k: FakePC(),
    RTCSessionDescription=lambda *a, **k: types.SimpleNamespace(sdp="ans", type="answer"),
)
mediastreams_stub = types.SimpleNamespace(MediaStreamTrack=type("MST", (), {}))
av_stub = types.SimpleNamespace(VideoFrame=type("VF", (), {}))
class _StreamSDKStub:
    def __init__(self, *a, **k): ...

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
    "aiortc": aiortc_stub,
    "aiortc.rtcconfiguration": rtc_config_stub,
    "aiortc.mediastreams": mediastreams_stub,
    "av": av_stub,
    "stream_pipeline_online": types.SimpleNamespace(StreamSDK=_StreamSDKStub),
    "webrtc.tracks": types.SimpleNamespace(AudioArrayTrack=_FakeAudioTrack, VideoFrameTrack=_FakeVideoTrack),
    "core": types.SimpleNamespace(),
    "core.atomic_components": types.SimpleNamespace(),
    "core.models": types.SimpleNamespace(),
}.items():
    sys.modules.setdefault(name, module)

from streaming_server import StreamingServer


class FakePC:
    def __init__(self):
        self.handlers = {}
        self.connectionState = "failed"
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


@pytest.fixture
def client(tmp_path, monkeypatch):
    app = StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data").app
    # force wait_for_connection to fail immediately
    async def _fail_wait(*args, **kwargs):
        return False
    monkeypatch.setattr("streaming_server.StreamingServer._wait_for_connection", _fail_wait)
    return TestClient(app)


def test_offer_returns_502_when_connection_fails(tmp_path, client):
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

    resp = client.post("/webrtc/offer", json=payload)
    assert resp.status_code == 502
    assert "peer connection failed" in resp.json()["detail"]
