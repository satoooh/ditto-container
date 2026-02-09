# Where: src/tests/test_error_handling_json.py
# What: Ensure errors return consistent JSON detail for validation and async pipeline handling (task 5.2).

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

class FakePC:
    def __init__(self, state="connected"):
        self.handlers = {}
        self.connectionState = state
        self.localDescription = types.SimpleNamespace(sdp="ans", type="answer")
        self.transceivers = []

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

    def addTransceiver(self, kind, direction="recvonly"):
        self.transceivers.append(types.SimpleNamespace(kind=kind, direction=direction))
        return self.transceivers[-1]

    async def close(self):
        return None


class StubSDK:
    def __init__(self, *a, **k):
        self.writer = None
    def setup(self, *a, **k):
        return None
    def setup_Nd(self, *a, **k):
        return None
    def close(self):
        return None


class BoomSDK(StubSDK):
    def setup(self, *a, **k):
        raise RuntimeError("boom")


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
    "stream_pipeline_online": types.SimpleNamespace(StreamSDK=StubSDK),
    "webrtc.tracks": types.SimpleNamespace(AudioArrayTrack=lambda *a, **k: None, VideoFrameTrack=lambda *a, **k: None),
}.items():
    sys.modules.setdefault(name, module)

from streaming_server import StreamingServer


@pytest.fixture
def client(monkeypatch, tmp_path):
    async def _wait_ok(*a, **k):
        return True
    async def _noop_run(*a, **k):
        return None

    server = StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data")
    monkeypatch.setattr(server, "_wait_for_connection", _wait_ok, raising=False)
    monkeypatch.setattr(server, "_run_streaming_pipeline", _noop_run, raising=False)
    return server, TestClient(server.app)


def test_validation_error_returns_400_json(client, tmp_path):
    _, api = client
    resp = api.post("/webrtc/offer", json={"sdp": "v=0", "type": "offer", "source_path": str(tmp_path/"b.png")})
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_connection_wait_fail_does_not_block_offer_response(client, monkeypatch, tmp_path):
    server, api = client
    async def _wait_fail(*a, **k):
        return False
    monkeypatch.setattr(server, "_wait_for_connection", _wait_fail, raising=False)

    audio = tmp_path / "a.wav"
    img = tmp_path / "b.png"
    audio.write_bytes(b"a")
    img.write_bytes(b"b")

    resp = api.post("/webrtc/offer", json={"sdp": "v=0", "type": "offer", "audio_path": str(audio), "source_path": str(img)})
    assert resp.status_code == 200
    assert resp.json().get("type") == "answer"


def test_pipeline_fail_keeps_offer_response_200(client, monkeypatch, tmp_path):
    server, api = client
    async def _wait_ok(*a, **k):
        return True
    async def _boom_pipeline(*a, **k):
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Streaming pipeline failed")
    monkeypatch.setattr(server, "_wait_for_connection", _wait_ok, raising=False)
    monkeypatch.setattr(server, "_run_streaming_pipeline", _boom_pipeline, raising=False)

    audio = tmp_path / "a.wav"
    img = tmp_path / "b.png"
    audio.write_bytes(b"a")
    img.write_bytes(b"b")

    resp = api.post("/webrtc/offer", json={"sdp": "v=0", "type": "offer", "audio_path": str(audio), "source_path": str(img)})
    # Pipeline is executed asynchronously; /webrtc/offer should still return the SDP answer.
    assert resp.status_code == 200
