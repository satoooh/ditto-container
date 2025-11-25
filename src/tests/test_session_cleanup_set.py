# Where: src/tests/test_session_cleanup_set.py
# What: Ensure peer connections are removed from _pcs on completion or early fail (task 5.3).

import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Stubs
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
        tx = types.SimpleNamespace(kind=kind, direction=direction)
        self.transceivers.append(tx)
        return tx

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


for name, module in {
    "cv2": cv2_stub,
    "librosa": librosa_stub,
    "numpy": numpy_stub,
    "uvicorn": uvicorn_stub,
    "aiortc": types.SimpleNamespace(RTCPeerConnection=lambda *a, **k: FakePC(), RTCSessionDescription=lambda *a, **k: types.SimpleNamespace(sdp="ans", type="answer")),
    "aiortc.rtcconfiguration": rtc_config_stub,
    "aiortc.mediastreams": mediastreams_stub,
    "av": av_stub,
    "stream_pipeline_online": types.SimpleNamespace(StreamSDK=StubSDK),
    "core": types.SimpleNamespace(),
    "core.atomic_components": types.SimpleNamespace(audio2motion=None),
    "core.models": types.SimpleNamespace(),
    "webrtc.tracks": types.SimpleNamespace(AudioArrayTrack=lambda *a, **k: None, VideoFrameTrack=lambda *a, **k: None),
}.items():
    sys.modules.setdefault(name, module)

from streaming_server import StreamingServer


@pytest.fixture
def api(monkeypatch, tmp_path):
    async def _wait_ok(*a, **k):
        return True
    async def _cleanup_run(**kwargs):
        await server._cleanup_peer(kwargs["pc"])

    server = StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data")
    monkeypatch.setattr(server, "_wait_for_connection", _wait_ok, raising=False)
    monkeypatch.setattr(server, "_run_streaming_pipeline", _cleanup_run, raising=False)
    return server, TestClient(server.app)


def test_peer_removed_after_success(api, tmp_path):
    server, client = api
    audio = tmp_path / "a.wav"
    img = tmp_path / "b.png"
    audio.write_bytes(b"a")
    img.write_bytes(b"b")

    resp = client.post(
        "/webrtc/offer",
        json={"sdp": "v=0", "type": "offer", "audio_path": str(audio), "source_path": str(img)},
    )

    assert resp.status_code == 200
    assert len(server._pcs) == 0


def test_peer_removed_after_connection_fail(api, monkeypatch, tmp_path):
    server, client = api
    async def _wait_fail(*a, **k):
        return False
    monkeypatch.setattr(server, "_wait_for_connection", _wait_fail, raising=False)

    audio = tmp_path / "a.wav"
    img = tmp_path / "b.png"
    audio.write_bytes(b"a")
    img.write_bytes(b"b")

    resp = client.post(
        "/webrtc/offer",
        json={"sdp": "v=0", "type": "offer", "audio_path": str(audio), "source_path": str(img)},
    )

    assert resp.status_code == 502
    assert len(server._pcs) == 0
