# Where: src/tests/test_audio_resample_to_48k.py
# What: Ensure /webrtc/offer resamples 16k audio to 48k and passes to AudioArrayTrack (task 4.4).

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
    resample=lambda audio, orig_sr, target_sr: [0.0] * len(audio),
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
    "stream_pipeline_online": types.SimpleNamespace(StreamSDK=None),
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


class FakeAudioTrack:
    last_sample_rate = None
    last_samples = None

    def __init__(self, samples, sample_rate, *a, **k):
        FakeAudioTrack.last_samples = samples
        FakeAudioTrack.last_sample_rate = sample_rate

    def finalize(self):
        return None


class FakeVideoTrack:
    def __init__(self, *a, **k): ...
    def enqueue(self, *a, **k): ...


class StubSDK:
    def __init__(self, *a, **k):
        self.writer = None
    def setup(self, *a, **k):
        return None
    def setup_Nd(self, *a, **k):
        return None
    def close(self):
        return None


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Patch dependencies
    monkeypatch.setattr("streaming_server.StreamSDK", StubSDK)
    monkeypatch.setattr("streaming_server.AudioArrayTrack", FakeAudioTrack)
    monkeypatch.setattr("streaming_server.VideoFrameTrack", FakeVideoTrack)

    resample_called = {}

    def fake_resample(audio, orig_sr, target_sr):
        resample_called["orig_sr"] = orig_sr
        resample_called["target_sr"] = target_sr
        return [0.0] * len(audio)

    monkeypatch.setattr("streaming_server.librosa.resample", fake_resample, raising=False)
    monkeypatch.setattr("streaming_server.librosa.load", lambda p, sr=16000: ([0.0] * sr, sr), raising=False)
    async def _wait_ok(*a, **k):
        return True
    async def _noop_run(*a, **k):
        return None
    monkeypatch.setattr("streaming_server.StreamingServer._wait_for_connection", _wait_ok, raising=False)
    monkeypatch.setattr("streaming_server.StreamingServer._run_streaming_pipeline", _noop_run, raising=False)

    server = StreamingServer(cfg_pkl="/tmp/cfg.pkl", data_root="/tmp/data")
    client = TestClient(server.app)
    client._resample_called = resample_called
    return client


def test_audio_resampled_to_48k_and_track_uses_it(tmp_path, client):
    audio = tmp_path / "a.wav"
    img = tmp_path / "b.png"
    audio.write_bytes(b"a")
    img.write_bytes(b"b")

    resp = client.post(
        "/webrtc/offer",
        json={"sdp": "v=0", "type": "offer", "audio_path": str(audio), "source_path": str(img)},
    )

    assert resp.status_code == 200
    assert client._resample_called["orig_sr"] == 16000
    assert client._resample_called["target_sr"] == 48000
    assert FakeAudioTrack.last_sample_rate == 48000
    assert FakeAudioTrack.last_samples is not None
