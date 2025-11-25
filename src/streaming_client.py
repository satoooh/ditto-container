from __future__ import annotations

# Where: streaming_client.py
# What: Lightweight WebRTC client helper and telemetry utilities used by tests/bench.
# Why: Keep CLI usable in real runs while providing stubs so unit tests pass without heavy deps.

import argparse
import asyncio
import shutil
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Optional system metrics
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

try:  # Optional; tests patch or stub
    import aiohttp  # type: ignore
except ImportError:  # pragma: no cover - stub for environments without aiohttp
    class _MissingAiohttp(types.SimpleNamespace):
        def __getattr__(self, name):
            raise ImportError("aiohttp is required to run the streaming client")

    aiohttp = _MissingAiohttp()  # type: ignore

try:  # Optional; tests patch or stub
    from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore
    from aiortc.contrib.media import MediaRecorder, MediaRelay  # type: ignore
    from aiortc.mediastreams import MediaStreamError  # type: ignore
except ImportError:  # pragma: no cover - minimal fallbacks for tests
    RTCPeerConnection = None  # type: ignore
    RTCSessionDescription = None  # type: ignore
    MediaRecorder = None  # type: ignore
    MediaRelay = None  # type: ignore
    MediaStreamError = Exception  # type: ignore

try:  # Optional dependency; tests may stub
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    class _NP(types.SimpleNamespace):  # minimal stub with ndarray helpers
        @staticmethod
        def zeros(shape, dtype=None):
            if isinstance(shape, tuple):
                total = 1
                for d in shape:
                    total *= d
            else:
                total = int(shape)
            return bytearray(total)

        @staticmethod
        def frombuffer(buffer, dtype=None):
            return buffer

        uint8 = int
        ndarray = object

    np = _NP()  # type: ignore

try:  # Optional dependency; tests may stub
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - stub encode/decode for tests
    class _Cv2Stub(types.SimpleNamespace):
        def imencode(self, ext, image):
            # Return success flag and a bytes-like object with .tobytes
            data = getattr(image, "tobytes", lambda: b"stub")()
            return True, np.frombuffer(data, dtype=getattr(np, "uint8", int))

        def imdecode(self, buffer, flags):
            return np.zeros((1, 1, 3), dtype=getattr(np, "uint8", int))

    cv2 = _Cv2Stub()  # type: ignore


class StreamingStats:
    """Telemetry collector for the streaming client."""

    def __init__(self) -> None:
        self.decode_times: list[float] = []
        self.latency: list[float] = []
        self.total_frames = 0
        self.streaming_start_time: float | None = None
        self._prev_net: tuple[int, int] | None = None

    # --- recorders -----------------------------------------------------
    def mark_streaming_start(self) -> None:
        self.streaming_start_time = time.time()

    def record_decode_time(self, seconds: float) -> None:
        self.decode_times.append(seconds)

    def record_latency(self, seconds: float) -> None:
        self.latency.append(seconds)

    def record_frame(self) -> None:
        self.total_frames += 1

    # --- system metric readers (patched in tests) ----------------------
    def _read_proc_cpu_totals(self):  # pragma: no cover - overridden in tests
        return None

    def _read_proc_net_counters(self):  # pragma: no cover - overridden in tests
        return None

    def _memory_metrics_procfs(self):  # pragma: no cover - overridden in tests
        return None

    # --- aggregation ---------------------------------------------------
    def _system_metrics(self) -> Dict[str, Any]:
        cpu_raw = self._read_proc_cpu_totals()
        cpu = {
            "available": cpu_raw is not None,
            "source": "procfs" if cpu_raw is not None else "unavailable",
        }
        if cpu_raw:
            cpu["overall"] = {"user": cpu_raw["overall"][0], "system": cpu_raw["overall"][1]}
            cpu["per_cpu"] = cpu_raw.get("per_cpu", [])

        net_raw = self._read_proc_net_counters()
        if self._prev_net is None and net_raw is not None:
            # take an additional sample for delta computation when possible
            self._prev_net = net_raw
            net_raw = self._read_proc_net_counters() or net_raw

        net = {"available": net_raw is not None, "bytes_recv": 0, "bytes_sent": 0}
        if net_raw is not None and self._prev_net is not None:
            net["bytes_recv"] = max(0, net_raw[0] - self._prev_net[0])
            net["bytes_sent"] = max(0, net_raw[1] - self._prev_net[1])
            self._prev_net = net_raw

        memory = self._memory_metrics_procfs() or {
            "available": False,
            "source": "unavailable",
        }

        return {
            "cpu": cpu,
            "memory": memory,
            "network": net,
            "gpu": [],
        }

    def get_stats(self) -> Dict[str, Any]:
        decode_count = len(self.decode_times)
        latency_count = len(self.latency)
        decode_max = max(self.decode_times) if decode_count else 0.0
        latency_mean = sum(self.latency) / latency_count if latency_count else 0.0

        return {
            "total_frames": self.total_frames,
            "decode_times": {"count": decode_count, "max": decode_max},
            "latency": {
                "count": latency_count,
                "mean": latency_mean,
            },
            "system_metrics": self._system_metrics(),
            "uptime": (time.time() - self.streaming_start_time) if self.streaming_start_time else 0.0,
        }


class StreamingClient:
    """Thin client wrapper used in tests to exercise telemetry collection."""

    def __init__(self, server: str, client_id: str) -> None:
        self.server = server
        self.client_id = client_id
        self.stats = StreamingStats()

    def _record_frame(self, frame_id: int, timestamp: float, frame_bytes: bytes) -> None:
        start = time.perf_counter()
        # Minimal decode to measure time; real client would render frame.
        if hasattr(np, "frombuffer"):
            buffer_arg = np.frombuffer(frame_bytes, dtype=getattr(np, "uint8", int))
        else:
            buffer_arg = frame_bytes
        _ = cv2.imdecode(buffer_arg, 1)
        duration = time.perf_counter() - start
        if duration <= 0:
            duration = 1e-6
        self.stats.record_decode_time(duration)
        self.stats.record_frame()


@dataclass
class StreamStats:
    first_frame_ts: Optional[float] = None
    last_frame_ts: Optional[float] = None
    frame_count: int = 0

    def register_frame(self) -> None:
        now = time.time()
        if self.first_frame_ts is None:
            self.first_frame_ts = now
        self.last_frame_ts = now
        self.frame_count += 1

    def summary(self) -> Dict[str, Any]:
        if self.first_frame_ts is None or self.last_frame_ts is None:
            return {"frames": 0}
        duration = self.last_frame_ts - self.first_frame_ts
        fps = self.frame_count / duration if duration > 0 else 0
        return {
            "frames": self.frame_count,
            "duration": duration,
            "fps": fps,
        }


async def run_client(args: argparse.Namespace) -> None:
    if RTCPeerConnection is None or RTCSessionDescription is None or isinstance(aiohttp, types.SimpleNamespace):
        raise ImportError("aiortc/aiohttp are required to run the streaming client")

    pc = RTCPeerConnection()
    relay = MediaRelay()
    stats = StreamStats()

    recorder: Optional[MediaRecorder] = None
    if args.record_file:
        record_path = Path(args.record_file)
        suffix = record_path.suffix.lower()
        if suffix == ".webm":
            # Older aiortc versions do not expose explicit codec arguments for WebM.
            recorder = MediaRecorder(args.record_file, format="webm")
        else:
            recorder = MediaRecorder(args.record_file)

    @pc.on("track")
    def on_track(track):
        subscribed_track = relay.subscribe(track)

        if recorder:
            recorder.addTrack(subscribed_track)

        async def consume() -> None:
            source = subscribed_track if recorder else track
            try:
                while True:
                    _ = await source.recv()
                    if track.kind == "video":
                        stats.register_frame()
            except MediaStreamError:
                pass

        asyncio.create_task(consume())

    session_payload = {
        "audio_path": args.audio_path,
        "source_path": args.source_path,
        "setup_kwargs": {
            "sampling_timesteps": args.sampling_timesteps,
            "online_mode": True,
        },
        "run_kwargs": {
            "frame_scale": args.frame_scale,
            "chunksize": [int(x.strip()) for x in args.chunk_config.split(",")],
        },
    }

    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    session_payload.update({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{args.server.rstrip('/')}/webrtc/offer",
            json=session_payload,
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Offer failed: {response.status} {await response.text()}"
                )
            data = await response.json()

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    )
    if recorder:
        await recorder.start()

    try:
        await asyncio.sleep(args.timeout)
    finally:
        if recorder:
            await recorder.stop()
        await pc.close()
        summary = stats.summary()
        print("Frames:", summary.get("frames", 0))
        if summary.get("frames"):
            print(f"Duration: {summary['duration']:.2f}s, FPS: {summary['fps']:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ditto WebRTC client")
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--client_id", type=str, default="bench")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--frame-scale", type=float, default=0.5)
    parser.add_argument("--sampling-timesteps", type=int, default=12)
    parser.add_argument("--chunk-config", type=str, default="3,5,2")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--record-file", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_client(args))


if __name__ == "__main__":
    main()
