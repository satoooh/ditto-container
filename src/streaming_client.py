from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder, MediaRelay
from aiortc.mediastreams import MediaStreamError


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
    pc = RTCPeerConnection()
    relay = MediaRelay()
    stats = StreamStats()

    recorder: Optional[MediaRecorder] = None
    if args.record_file:
        record_path = Path(args.record_file)
        suffix = record_path.suffix.lower()
        if suffix == ".webm":
            recorder = MediaRecorder(
                args.record_file,
                format="webm",
                video_codec="vp8",
                audio_codec="opus",
            )
        else:
            recorder = MediaRecorder(args.record_file)

    @pc.on("track")
    def on_track(track):
        subscribed_track = relay.subscribe(track)

        if recorder:
            recorder.addTrack(subscribed_track)

        async def consume() -> None:
            try:
                while True:
                    _ = await subscribed_track.recv()
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
