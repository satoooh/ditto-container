import asyncio
import json
import time
import base64
import argparse
import logging
from typing import List, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass
import statistics
import cv2
import numpy as np
from streaming_protocol import parse_binary_frame

import websockets
from websockets.exceptions import ConnectionClosed


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameStats:
    frame_id: int
    timestamp: float
    arrival_time: float
    frame_size: int
    decode_time: float


class StreamingStats:
    def __init__(self):
        self.frames: List[FrameStats] = []
        self.start_time: Optional[float] = None
        self.first_frame_time: Optional[float] = None
        self.last_frame_time: Optional[float] = None
        self.streaming_start_time: Optional[float] = None
        self.total_frames = 0
        self.total_bytes = 0
        self.frame_intervals = deque(maxlen=100)  # Keep last 100 intervals

    def add_frame(self, frame_stats: FrameStats):
        self.frames.append(frame_stats)
        self.total_frames += 1
        self.total_bytes += frame_stats.frame_size

        current_time = frame_stats.arrival_time

        if self.first_frame_time is None:
            self.first_frame_time = current_time

        if self.last_frame_time is not None:
            interval = current_time - self.last_frame_time
            self.frame_intervals.append(interval)

        self.last_frame_time = current_time

    def get_stats(self) -> Dict[str, Any]:
        if not self.frames:
            return {"error": "No frames received"}

        total_duration = (
            self.last_frame_time - self.first_frame_time
            if self.first_frame_time and self.last_frame_time
            else 0
        )

        # Calculate streaming latency (time from start to first frame)
        streaming_latency = (
            self.first_frame_time - self.streaming_start_time
            if self.first_frame_time and self.streaming_start_time
            else 0
        )

        # Frame interval statistics
        intervals = list(self.frame_intervals)

        # Decode time statistics
        decode_times = [f.decode_time for f in self.frames]

        return {
            "total_frames": self.total_frames,
            "total_duration": total_duration,
            "streaming_latency": streaming_latency,
            "average_fps": self.total_frames / total_duration
            if total_duration > 0
            else 0,
            "total_bytes": self.total_bytes,
            "average_frame_size": self.total_bytes / self.total_frames
            if self.total_frames > 0
            else 0,
            "frame_intervals": {
                "count": len(intervals),
                "mean": statistics.mean(intervals) if intervals else 0,
                "median": statistics.median(intervals) if intervals else 0,
                "std": statistics.stdev(intervals) if len(intervals) > 1 else 0,
                "min": min(intervals) if intervals else 0,
                "max": max(intervals) if intervals else 0,
            },
            "decode_times": {
                "mean": statistics.mean(decode_times) if decode_times else 0,
                "median": statistics.median(decode_times) if decode_times else 0,
                "std": statistics.stdev(decode_times) if len(decode_times) > 1 else 0,
                "min": min(decode_times) if decode_times else 0,
                "max": max(decode_times) if decode_times else 0,
            },
            "bandwidth_mbps": (self.total_bytes * 8) / (total_duration * 1_000_000)
            if total_duration > 0
            else 0,
        }

    def print_stats(self):
        stats = self.get_stats()

        if "error" in stats:
            print(f"❌ {stats['error']}")
            return

        print("\n" + "=" * 70)
        print("STREAMING CLIENT STATISTICS")
        print("=" * 70)

        print(f"Total Frames Received: {stats['total_frames']}")
        print(f"Total Duration: {stats['total_duration']:.2f}s")
        print(
            f"Streaming Latency: {stats['streaming_latency']:.2f}s (time to first frame)"
        )
        print(f"Average FPS: {stats['average_fps']:.1f}")
        print(f"Total Data: {stats['total_bytes'] / 1024 / 1024:.1f} MB")
        print(f"Average Frame Size: {stats['average_frame_size'] / 1024:.1f} KB")
        print(f"Bandwidth: {stats['bandwidth_mbps']:.1f} Mbps")

        print("\nFRAME INTERVALS:")
        intervals = stats["frame_intervals"]
        print(f"  Count: {intervals['count']}")
        print(f"  Mean: {intervals['mean'] * 1000:.1f}ms")
        print(f"  Median: {intervals['median'] * 1000:.1f}ms")
        print(f"  Std Dev: {intervals['std'] * 1000:.1f}ms")
        print(f"  Min: {intervals['min'] * 1000:.1f}ms")
        print(f"  Max: {intervals['max'] * 1000:.1f}ms")

        print("\nDECODE TIMES:")
        decode = stats["decode_times"]
        print(f"  Mean: {decode['mean'] * 1000:.1f}ms")
        print(f"  Median: {decode['median'] * 1000:.1f}ms")
        print(f"  Std Dev: {decode['std'] * 1000:.1f}ms")
        print(f"  Min: {decode['min'] * 1000:.1f}ms")
        print(f"  Max: {decode['max'] * 1000:.1f}ms")

        print("\nREAL-TIME SUITABILITY:")
        target_fps = 25  # Assuming 25 FPS target
        target_interval = 1.0 / target_fps

        if stats["average_fps"] >= target_fps * 0.9:
            print(
                f"✅ Frame rate suitable for real-time ({stats['average_fps']:.1f} >= {target_fps * 0.9:.1f} FPS)"
            )
        else:
            print(
                f"⚠️  Frame rate below target ({stats['average_fps']:.1f} < {target_fps * 0.9:.1f} FPS)"
            )

        if stats["streaming_latency"] <= 3.0:
            print(
                f"✅ Streaming latency acceptable ({stats['streaming_latency']:.1f}s <= 3.0s)"
            )
        else:
            print(
                f"⚠️  Streaming latency high ({stats['streaming_latency']:.1f}s > 3.0s)"
            )

        if intervals["mean"] <= target_interval * 1.5:
            print(
                f"✅ Frame intervals consistent ({intervals['mean'] * 1000:.1f}ms <= {target_interval * 1.5 * 1000:.1f}ms)"
            )
        else:
            print(
                f"⚠️  Frame intervals inconsistent ({intervals['mean'] * 1000:.1f}ms > {target_interval * 1.5 * 1000:.1f}ms)"
            )

        print("=" * 70)


class StreamingClient:
    def __init__(self, server_url: str, client_id: str, prefer_binary: bool = True):
        self.server_url = server_url
        self.client_id = client_id
        self.stats = StreamingStats()
        self.websocket = None
        self.save_frames = False
        self.frame_save_dir = "received_frames"
        self.prefer_binary = prefer_binary
        self.active_binary = prefer_binary

    async def connect(self):
        """Connect to the streaming server"""
        try:
            self.websocket = await websockets.connect(
                f"{self.server_url}/ws/{self.client_id}"
            )
            logger.info(f"Connected to server as client {self.client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False

    async def start_streaming(
        self, audio_path: str, source_path: str, config: Dict[str, Any] = None
    ):
        """Send start streaming command to server"""
        if not self.websocket:
            logger.error("Not connected to server")
            return False

        if config is None:
            config = {}

        message = {
            "type": "start_streaming",
            "audio_path": audio_path,
            "source_path": source_path,
            "setup_kwargs": config.get("setup_kwargs", {}),
            "run_kwargs": config.get("run_kwargs", {}),
            "binary": self.prefer_binary,
        }

        try:
            await self.websocket.send(json.dumps(message))
            logger.info("Sent start streaming command")
            self.stats.streaming_start_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Failed to send start streaming command: {e}")
            return False

    async def stop_streaming(self):
        """Send stop streaming command to server"""
        if not self.websocket:
            return False

        try:
            await self.websocket.send(json.dumps({"type": "stop_streaming"}))
            logger.info("Sent stop streaming command")
            return True
        except Exception as e:
            logger.error(f"Failed to send stop streaming command: {e}")
            return False

    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages from server"""
        message_type = message.get("type")

        if message_type == "frame":
            await self.handle_json_frame(message)
        elif message_type == "streaming_started":
            self.active_binary = bool(message.get("binary", self.prefer_binary))
            logger.info(
                f"Streaming started: {message.get('audio_path')} -> {message.get('source_path')} "
                f"(binary={self.active_binary})"
            )
        elif message_type == "metadata":
            logger.info(
                f"Metadata: {message.get('audio_duration'):.2f}s, {message.get('expected_frames')} frames"
            )
        elif message_type == "streaming_completed":
            logger.info("Streaming completed")
        elif message_type == "writer_closed":
            logger.info(f"Writer closed. Total frames: {message.get('total_frames')}")
        elif message_type == "error":
            logger.error(f"Server error: {message.get('message')}")
        elif message_type == "pong":
            logger.debug("Received pong")
        else:
            logger.debug(f"Unknown message type: {message_type}")

    async def handle_json_frame(self, message: Dict[str, Any]):
        """Handle incoming frame"""
        frame_id = message.get("frame_id")
        frame_data = message.get("frame_data")
        timestamp = message.get("timestamp")
        try:
            frame_bytes = base64.b64decode(frame_data)
        except Exception as e:
            logger.error(f"Error decoding frame {frame_id}: {e}")
            return

        self._record_frame(frame_id, timestamp, frame_bytes)

    async def handle_binary_frame(self, payload: bytes):
        try:
            frame_id, timestamp, frame_bytes = parse_binary_frame(payload)
        except ValueError as exc:
            logger.error(f"Invalid binary frame received: {exc}")
            return

        self._record_frame(frame_id, timestamp, frame_bytes)

    def _record_frame(
        self, frame_id: int, timestamp: float, frame_bytes: bytes
    ) -> None:
        arrival_time = time.time()

        decode_start = time.time()
        frame_size = len(frame_bytes)

        if self.save_frames:
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            import os

            os.makedirs(self.frame_save_dir, exist_ok=True)
            cv2.imwrite(f"{self.frame_save_dir}/frame_{frame_id:06d}.webp", frame_img)

        decode_time = time.time() - decode_start

        frame_stats = FrameStats(
            frame_id=frame_id,
            timestamp=timestamp,
            arrival_time=arrival_time,
            frame_size=frame_size,
            decode_time=decode_time,
        )

        self.stats.add_frame(frame_stats)

        if frame_id % 25 == 0:
            logger.info(f"Received frame {frame_id}, size: {frame_size / 1024:.1f}KB")

    async def listen(self):
        """Listen for messages from server"""
        if not self.websocket:
            logger.error("Not connected to server")
            return

        try:
            while True:
                incoming = await self.websocket.recv()
                if isinstance(incoming, bytes):
                    await self.handle_binary_frame(incoming)
                else:
                    try:
                        message = json.loads(incoming)
                    except json.JSONDecodeError as exc:
                        logger.error(f"Failed to decode server message: {exc}")
                        continue
                    await self.handle_message(message)
        except ConnectionClosed:
            logger.info("Connection closed by server")
        except Exception as e:
            logger.error(f"Error listening to server: {e}")

    async def ping(self):
        """Send periodic ping to server"""
        if not self.websocket:
            return

        try:
            await self.websocket.send(json.dumps({"type": "ping"}))
        except Exception as e:
            logger.error(f"Error sending ping: {e}")

    async def close(self):
        """Close connection to server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from server")


async def main():
    parser = argparse.ArgumentParser(description="Ditto Streaming Client")
    parser.add_argument(
        "--server", type=str, default="ws://localhost:8000", help="Server WebSocket URL"
    )
    parser.add_argument(
        "--client_id", type=str, default="test_client", help="Client ID"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="./example/audio.wav",
        help="Audio file path on server",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="./example/image.png",
        help="Source image path on server",
    )
    parser.add_argument(
        "--save_frames", action="store_true", help="Save received frames to disk"
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Connection timeout in seconds"
    )
    parser.add_argument(
        "--transport",
        choices=["binary", "json"],
        default="binary",
        help="Preferred frame transport (default: binary)",
    )

    args = parser.parse_args()

    # Create client
    client = StreamingClient(
        args.server, args.client_id, prefer_binary=(args.transport == "binary")
    )
    client.save_frames = args.save_frames

    # Connect to server
    if not await client.connect():
        return

    try:
        # Start streaming
        streaming_config = {
            "setup_kwargs": {},
            "run_kwargs": {"chunksize": (3, 5, 2), "fade_in": -1, "fade_out": -1},
        }

        if not await client.start_streaming(
            args.audio_path, args.source_path, streaming_config
        ):
            return

        # Listen for messages with timeout
        try:
            await asyncio.wait_for(client.listen(), timeout=args.timeout)
        except asyncio.TimeoutError:
            logger.info("Timeout reached, stopping...")

        # Print final statistics
        client.stats.print_stats()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
