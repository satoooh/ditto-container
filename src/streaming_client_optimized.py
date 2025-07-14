import asyncio
import json
import time
import argparse
import logging
import struct
from typing import List, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass
import statistics
import cv2
import numpy as np

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
        self.frame_intervals = deque(maxlen=100)
        
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
        
        total_duration = self.last_frame_time - self.first_frame_time if self.first_frame_time and self.last_frame_time else 0
        streaming_latency = self.first_frame_time - self.streaming_start_time if self.first_frame_time and self.streaming_start_time else 0
        intervals = list(self.frame_intervals)
        decode_times = [f.decode_time for f in self.frames]
        
        return {
            "total_frames": self.total_frames,
            "total_duration": total_duration,
            "streaming_latency": streaming_latency,
            "average_fps": self.total_frames / total_duration if total_duration > 0 else 0,
            "total_bytes": self.total_bytes,
            "average_frame_size": self.total_bytes / self.total_frames if self.total_frames > 0 else 0,
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
            "bandwidth_mbps": (self.total_bytes * 8) / (total_duration * 1_000_000) if total_duration > 0 else 0,
        }
    
    def print_stats(self):
        stats = self.get_stats()
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return
            
        print("\n" + "="*70)
        print("OPTIMIZED STREAMING CLIENT STATISTICS")
        print("="*70)
        
        print(f"Total Frames Received: {stats['total_frames']}")
        print(f"Total Duration: {stats['total_duration']:.2f}s")
        print(f"Streaming Latency: {stats['streaming_latency']:.2f}s (time to first frame)")
        print(f"Average FPS: {stats['average_fps']:.1f}")
        print(f"Total Data: {stats['total_bytes']/1024/1024:.1f} MB")
        print(f"Average Frame Size: {stats['average_frame_size']/1024:.1f} KB")
        print(f"Bandwidth: {stats['bandwidth_mbps']:.1f} Mbps")
        
        print(f"\nFRAME INTERVALS:")
        intervals = stats['frame_intervals']
        print(f"  Count: {intervals['count']}")
        print(f"  Mean: {intervals['mean']*1000:.1f}ms")
        print(f"  Median: {intervals['median']*1000:.1f}ms") 
        print(f"  Std Dev: {intervals['std']*1000:.1f}ms")
        print(f"  Min: {intervals['min']*1000:.1f}ms")
        print(f"  Max: {intervals['max']*1000:.1f}ms")
        
        print(f"\nDECODE TIMES:")
        decode = stats['decode_times']
        print(f"  Mean: {decode['mean']*1000:.1f}ms")
        print(f"  Median: {decode['median']*1000:.1f}ms")
        print(f"  Std Dev: {decode['std']*1000:.1f}ms")
        print(f"  Min: {decode['min']*1000:.1f}ms")
        print(f"  Max: {decode['max']*1000:.1f}ms")
        
        print(f"\nREAL-TIME SUITABILITY:")
        target_fps = 25
        
        if stats['average_fps'] >= target_fps * 0.9:
            print(f"✅ Frame rate suitable for real-time ({stats['average_fps']:.1f} >= {target_fps*0.9:.1f} FPS)")
        else:
            print(f"⚠️  Frame rate below target ({stats['average_fps']:.1f} < {target_fps*0.9:.1f} FPS)")
        
        if stats['streaming_latency'] <= 3.0:
            print(f"✅ Streaming latency acceptable ({stats['streaming_latency']:.1f}s <= 3.0s)")
        else:
            print(f"⚠️  Streaming latency high ({stats['streaming_latency']:.1f}s > 3.0s)")
        
        target_interval = 1.0 / target_fps
        if intervals['mean'] <= target_interval * 1.5:
            print(f"✅ Frame intervals consistent ({intervals['mean']*1000:.1f}ms <= {target_interval*1.5*1000:.1f}ms)")
        else:
            print(f"⚠️  Frame intervals inconsistent ({intervals['mean']*1000:.1f}ms > {target_interval*1.5*1000:.1f}ms)")
        
        print("="*70)


class OptimizedStreamingClient:
    def __init__(self, server_url: str, client_id: str):
        self.server_url = server_url
        self.client_id = client_id
        self.stats = StreamingStats()
        self.websocket = None
        self.save_frames = False
        self.frame_save_dir = "received_frames"
        
    async def connect(self):
        """Connect to the streaming server"""
        try:
            self.websocket = await websockets.connect(f"{self.server_url}/ws/{self.client_id}")
            logger.info(f"Connected to optimized server as client {self.client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    async def start_streaming(self, audio_path: str, source_path: str, config: Dict[str, Any] = None):
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
            "run_kwargs": config.get("run_kwargs", {})
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
    
    async def handle_message(self, message_data):
        """✅ OPTIMIZED: Handle both text and binary messages"""
        if isinstance(message_data, str):
            # Text message (JSON)
            message = json.loads(message_data)
            await self.handle_text_message(message)
        elif isinstance(message_data, bytes):
            # Binary message (frame data)
            await self.handle_binary_frame(message_data)
        else:
            logger.error(f"Unknown message type: {type(message_data)}")
    
    async def handle_text_message(self, message: Dict[str, Any]):
        """Handle text messages (metadata, status, etc.)"""
        message_type = message.get("type")
        
        if message_type == "streaming_started":
            logger.info(f"Streaming started: {message.get('audio_path')} -> {message.get('source_path')}")
        elif message_type == "metadata":
            logger.info(f"Metadata: {message.get('audio_duration'):.2f}s, {message.get('expected_frames')} frames")
        elif message_type == "streaming_completed":
            logger.info("Streaming completed")
        elif message_type == "writer_closed":
            logger.info(f"Writer closed. Total frames: {message.get('total_frames')}")
        elif message_type == "error":
            logger.error(f"Server error: {message.get('message')}")
        elif message_type == "pong":
            logger.debug("Received pong")
        else:
            logger.debug(f"Unknown text message type: {message_type}")
    
    async def handle_text_frame(self, message: Dict[str, Any]):
        """🔍 DEBUG: Handle text+base64 frames (fallback mode)"""
        frame_id = message.get("frame_id")
        frame_data_b64 = message.get("frame_data")
        timestamp = message.get("timestamp")
        arrival_time = time.time()
        
        try:
            # Decode base64 frame data
            import base64
            decode_start = time.time()
            jpeg_data = base64.b64decode(frame_data_b64)
            
            # 🔍 DEBUG: Log frame sizes
            if frame_id % 50 == 0:
                logger.info(f"🔍 Received text frame {frame_id}: Base64={len(frame_data_b64)/1024:.1f}KB, JPEG={len(jpeg_data)/1024:.1f}KB")
            
            # Optional: decode to numpy array for processing
            if self.save_frames:
                frame_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame_img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                # Save frame (optional)
                import os
                os.makedirs(self.frame_save_dir, exist_ok=True)
                cv2.imwrite(f"{self.frame_save_dir}/frame_{frame_id:06d}.jpg", frame_img)
            
            decode_time = time.time() - decode_start
            
            # Record frame statistics
            frame_stats = FrameStats(
                frame_id=frame_id,
                timestamp=timestamp,
                arrival_time=arrival_time,
                frame_size=len(jpeg_data),
                decode_time=decode_time
            )
            
            self.stats.add_frame(frame_stats)
            
            # ✅ OPTIMIZATION: Reduce logging frequency
            if frame_id % 50 == 0:  # Log every 50 frames instead of every 25
                logger.info(f"Received frame {frame_id}, size: {len(jpeg_data)/1024:.1f}KB")
            
        except Exception as e:
            logger.error(f"Error processing text frame {frame_id}: {e}")
    
    async def handle_binary_frame(self, frame_data: bytes):
        """✅ OPTIMIZED: Handle binary frame data"""
        arrival_time = time.time()
        
        try:
            # Parse binary header: frame_id (uint32) + timestamp (double) + data_length (uint32)
            header_size = struct.calcsize('!IdI')
            if len(frame_data) < header_size:
                logger.error(f"Frame data too short: {len(frame_data)} bytes")
                return
            
            frame_id, timestamp, data_length = struct.unpack('!IdI', frame_data[:header_size])
            jpeg_data = frame_data[header_size:]
            
            # 🔍 DEBUG: Monitor binary reception
            if frame_id % 100 == 0:
                logger.info(f"🔍 Received binary frame {frame_id}: {len(jpeg_data)/1024:.1f}KB")
            
            if len(jpeg_data) != data_length:
                logger.error(f"Frame data length mismatch: expected {data_length}, got {len(jpeg_data)}")
                return
            
            # ✅ OPTIMIZATION: Direct JPEG decode without base64 step
            decode_start = time.time()
            
            # Optional: decode to numpy array for processing
            if self.save_frames:
                frame_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame_img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                # Save frame (optional)
                import os
                os.makedirs(self.frame_save_dir, exist_ok=True)
                cv2.imwrite(f"{self.frame_save_dir}/frame_{frame_id:06d}.jpg", frame_img)
            
            decode_time = time.time() - decode_start
            
            # Record frame statistics
            frame_stats = FrameStats(
                frame_id=frame_id,
                timestamp=timestamp,
                arrival_time=arrival_time,
                frame_size=len(jpeg_data),
                decode_time=decode_time
            )
            
            self.stats.add_frame(frame_stats)
            
            # ✅ OPTIMIZATION: Reduce logging frequency
            if frame_id % 100 == 0:  # Log every 100 frames
                logger.info(f"Received binary frame {frame_id}, size: {len(jpeg_data)/1024:.1f}KB")
            
        except Exception as e:
            logger.error(f"Error processing binary frame: {e}")
    
    async def listen(self):
        """✅ OPTIMIZED: Listen for both text and binary messages"""
        if not self.websocket:
            logger.error("Not connected to server")
            return
        
        try:
            while True:
                # Handle both text and binary messages
                message_data = await self.websocket.recv()
                await self.handle_message(message_data)
                
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
    parser = argparse.ArgumentParser(description="Ditto Streaming Client (Optimized)")
    parser.add_argument("--server", type=str, default="ws://localhost:8000", 
                      help="Server WebSocket URL")
    parser.add_argument("--client_id", type=str, default="test_client",
                      help="Client ID")
    parser.add_argument("--audio_path", type=str, default="./example/audio.wav",
                      help="Audio file path on server")
    parser.add_argument("--source_path", type=str, default="./example/image.png",
                      help="Source image path on server")
    parser.add_argument("--save_frames", action="store_true",
                      help="Save received frames to disk")
    parser.add_argument("--timeout", type=int, default=60,
                      help="Connection timeout in seconds")
    
    args = parser.parse_args()
    
    # Create optimized client
    client = OptimizedStreamingClient(args.server, args.client_id)
    client.save_frames = args.save_frames
    
    # Connect to server
    if not await client.connect():
        return
    
    try:
        # Start streaming
        streaming_config = {
            "setup_kwargs": {},
            "run_kwargs": {
                "chunksize": (3, 5, 2),
                "fade_in": -1,
                "fade_out": -1
            }
        }
        
        if not await client.start_streaming(args.audio_path, args.source_path, streaming_config):
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