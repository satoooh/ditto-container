# Where: src/tests/test_e2e_smoke.py
# What: Stubbed E2E-ish smoke to verify frame_scale handling and basic SDP path (task 6.3).

import sys
import types
from pathlib import Path

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
    def __init__(self):
        self.handlers = {}
        self.connectionState = "connected"
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
    "core": types.SimpleNamespace(),
    "core.atomic_components": types.SimpleNamespace(audio2motion=None),
    "core.models": types.SimpleNamespace(),
    "stream_pipeline_online": types.SimpleNamespace(StreamSDK=StubSDK),
    "webrtc.tracks": types.SimpleNamespace(AudioArrayTrack=lambda *a, **k: None, VideoFrameTrack=lambda *a, **k: None),
}.items():
    sys.modules.setdefault(name, module)

from streaming_server import StreamingServer


def test_offer_smoke_frame_scale(tmp_path, monkeypatch):
    async def _wait_ok(*a, **k):
        return True
    async def _cleanup_run(**kwargs):
        server = kwargs.get("self") or kwargs.get("server")
        pc = kwargs.get("pc")
        if server and pc:
            await server._cleanup_peer(pc)

    server = StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data")
    monkeypatch.setattr(server, "_wait_for_connection", _wait_ok, raising=False)
    monkeypatch.setattr(server, "_run_streaming_pipeline", _cleanup_run, raising=False)

    api = TestClient(server.app)

    audio = tmp_path / "a.wav"
    img = tmp_path / "b.png"
    audio.write_bytes(b"a")
    img.write_bytes(b"b")

    resp = api.post(
        "/webrtc/offer",
        json={
            "sdp": "v=0",
            "type": "offer",
            "audio_path": str(audio),
            "source_path": str(img),
            "run_kwargs": {"frame_scale": 0.3},
            "setup_kwargs": {"sampling_timesteps": 10},
        },
    )

    assert resp.status_code == 200
    assert resp.json()["type"] == "answer"
