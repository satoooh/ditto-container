import asyncio
import json
import logging
import os
import time
import struct
from typing import Dict, Any
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from stream_pipeline_online import StreamSDK


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedStreamingServer:
    def __init__(self, cfg_pkl: str, data_root: str):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.active_connections: Dict[str, WebSocket] = {}
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        # ✅ OPTIMIZATION 1: Use asyncio.Queue instead of threading.Queue
        self.frame_queues: Dict[str, asyncio.Queue] = {}
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Ditto Streaming Server (Optimized)", version="1.0.0")
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.get("/")
        async def get_homepage():
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Ditto Streaming Server (Optimized)</title>
            </head>
            <body>
                <h1>Ditto Streaming Server (Optimized)</h1>
                <p>Server is running with performance optimizations.</p>
                <p>WebSocket endpoint: ws://localhost:8000/ws/{client_id}</p>
            </body>
            </html>
            """)
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.handle_websocket_connection(websocket, client_id)
    
    async def handle_websocket_connection(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
        
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                await self.handle_message(client_id, data)
                
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected")
            self.cleanup_client(client_id)
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
            self.cleanup_client(client_id)
    
    async def handle_message(self, client_id: str, data: Dict[str, Any]):
        message_type = data.get("type")
        
        if message_type == "start_streaming":
            await self.start_streaming(client_id, data)
        elif message_type == "stop_streaming":
            await self.stop_streaming(client_id)
        elif message_type == "ping":
            await self.send_message(client_id, {"type": "pong", "timestamp": time.time()})
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def start_streaming(self, client_id: str, config: Dict[str, Any]):
        if client_id in self.active_streams:
            await self.send_message(client_id, {
                "type": "error", 
                "message": "Streaming already active for this client"
            })
            return
        
        # Get streaming parameters
        audio_path = config.get("audio_path", "./example/audio.wav")
        source_path = config.get("source_path", "./example/image.png")
        
        # Validate files exist
        if not os.path.exists(audio_path):
            await self.send_message(client_id, {
                "type": "error", 
                "message": f"Audio file not found: {audio_path}"
            })
            return
            
        if not os.path.exists(source_path):
            await self.send_message(client_id, {
                "type": "error", 
                "message": f"Source file not found: {source_path}"
            })
            return
        
        logger.info(f"Starting streaming for client {client_id}")
        
        # Send streaming started confirmation
        await self.send_message(client_id, {
            "type": "streaming_started",
            "audio_path": audio_path,
            "source_path": source_path,
            "timestamp": time.time()
        })
        
        # ✅ NO THROTTLING: Large queue, let GPU generate at full speed
        # Client pulls at network speed, GPU generates at full speed (~41 FPS)
        # Queue size: 500 frames × 46KB = ~23MB (totally fine with modern memory)
        self.frame_queues[client_id] = asyncio.Queue(maxsize=500)
        
        # Add client to active_streams BEFORE creating frame sender task
        self.active_streams[client_id] = {
            "streaming_task": None,
            "frame_sender_task": None,
            "start_time": time.time(),
            "frame_count": 0
        }
        
        # Start frame sender task
        frame_sender_task = asyncio.create_task(
            self.frame_sender_worker(client_id)
        )
        
        # Start streaming task
        try:
            streaming_task = asyncio.create_task(
                self.run_streaming_pipeline(client_id, audio_path, source_path, config)
            )
            
        except Exception as e:
            logger.error(f"Error creating streaming task for {client_id}: {e}")
            if client_id in self.active_streams:
                del self.active_streams[client_id]
            await self.send_message(client_id, {
                "type": "error",
                "message": f"Failed to create streaming task: {str(e)}"
            })
            return
        
        # Update active_streams with task objects
        self.active_streams[client_id]["streaming_task"] = streaming_task
        self.active_streams[client_id]["frame_sender_task"] = frame_sender_task
    
    async def frame_sender_worker(self, client_id: str):
        """✅ OPTIMIZED: Worker using asyncio.Queue with blocking get()"""
        frames_sent = 0
        
        try:
            frame_queue = self.frame_queues.get(client_id)
            if not frame_queue:
                logger.error(f"No frame queue found for {client_id}")
                return
            
            # ✅ OPTIMIZATION 1: Use blocking queue.get() instead of polling
            while True:
                try:
                    # Blocking get with timeout - much more efficient than polling
                    message = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                    
                    if message is None:  # Poison pill
                        logger.info(f"Streaming completed for {client_id}, sent {frames_sent} frames")
                        break
                    
                    # ✅ OPTIMIZATION 3: Send binary messages instead of JSON
                    if message.get("type") == "frame":
                        await self.send_binary_frame(client_id, message)
                    else:
                        await self.send_message(client_id, message)
                    
                    frames_sent += 1
                    
                    # ✅ OPTIMIZATION 2: Reduce logging frequency
                    if frames_sent % 50 == 0:  # Log every 50 frames instead of every 10
                        logger.info(f"Sent {frames_sent} frames to {client_id}")
                    
                except asyncio.TimeoutError:
                    # Check if client is still connected
                    if client_id not in self.active_streams:
                        break
                    continue
                    
        except Exception as e:
            logger.error(f"Frame sender error for {client_id}: {e}")
        
        logger.info(f"Frame sender exiting for {client_id}, sent {frames_sent} frames")
    
    async def send_binary_frame(self, client_id: str, message: Dict[str, Any]):
        """✅ OPTIMIZATION 3: Send frame as binary WebSocket message"""
        if client_id not in self.active_connections:
            return
        
        try:
            # Pack frame data as binary: frame_id(4 bytes) + timestamp(8 bytes) + jpeg_data
            frame_id = message["frame_id"]
            timestamp = message["timestamp"]
            jpeg_data = message["jpeg_data"]  # Raw JPEG bytes, not base64
            
            # 🔍 DEBUG: Monitor binary transmission
            if frame_id % 100 == 0:
                logger.info(f"🔍 Sending binary frame {frame_id}: {len(jpeg_data)/1024:.1f}KB")
            
            # Create binary header: frame_id (uint32) + timestamp (double) + data_length (uint32)
            header = struct.pack('!IdI', frame_id, timestamp, len(jpeg_data))
            binary_message = header + jpeg_data
            
            await self.active_connections[client_id].send_bytes(binary_message)
            
        except Exception as e:
            logger.error(f"Error sending binary frame to {client_id}: {e}")
            self.cleanup_client(client_id)
    
    async def run_streaming_pipeline(self, client_id: str, audio_path: str, source_path: str, config: Dict[str, Any]):
        try:
            # Initialize SDK
            SDK = StreamSDK(self.cfg_pkl, self.data_root)
            
            # Create optimized writer
            websocket_writer = OptimizedWebSocketFrameWriter(self, client_id)
            
            # Setup the pipeline
            setup_kwargs = config.get("setup_kwargs", {})
            temp_output_path = f"/tmp/streaming_output_{client_id}.mp4"
            
            # Run setup in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, SDK.setup, source_path, temp_output_path, **setup_kwargs)
            
            # Replace writer
            SDK.writer = websocket_writer
            
            # Load and process audio
            import librosa
            import math
            
            audio, sr = librosa.core.load(audio_path, sr=16000)
            num_f = math.ceil(len(audio) / 16000 * 25)
            
            # Setup audio processing
            run_kwargs = config.get("run_kwargs", {})
            fade_in = run_kwargs.get("fade_in", -1)
            fade_out = run_kwargs.get("fade_out", -1)
            ctrl_info = run_kwargs.get("ctrl_info", {})
            
            SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
            
            # Send metadata
            await self.send_message(client_id, {
                "type": "metadata",
                "audio_duration": len(audio) / sr,
                "expected_frames": num_f,
                "timestamp": time.time()
            })
            
            # Process audio chunks
            if SDK.online_mode:
                chunksize = run_kwargs.get("chunksize", (3, 5, 2))
                audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
                split_len = int(sum(chunksize) * 0.04 * 16000) + 80
                
                for i, chunk_start in enumerate(range(0, len(audio), chunksize[1] * 640)):
                    if client_id not in self.active_streams:
                        break
                        
                    audio_chunk = audio[chunk_start:chunk_start + split_len]
                    if len(audio_chunk) < split_len:
                        audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
                    
                    # ✅ OPTIMIZATION 2: Reduce chunk processing logs
                    if i % 20 == 0:  # Log every 20th chunk instead of every chunk
                        logger.info(f"Processing chunk {i+1} for client {client_id}")
                    
                    # Run chunk processing in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, SDK.run_chunk, audio_chunk, chunksize)
                    
                    # ✅ NO THROTTLING: Let GPU run at full speed, client pulls at network speed
            else:
                # Offline mode
                aud_feat = SDK.wav2feat.wav2feat(audio)
                SDK.audio2motion_queue.put(aud_feat)
            
            # Close pipeline
            SDK.close()
            
            # Send completion message
            await self.send_message(client_id, {
                "type": "streaming_completed",
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"Streaming error for client {client_id}: {e}")
            await self.send_message(client_id, {
                "type": "error",
                "message": f"Streaming error: {str(e)}"
            })
        finally:
            # Send poison pill to frame sender
            if client_id in self.frame_queues:
                await self.frame_queues[client_id].put(None)
            
            # Clean up
            if client_id in self.active_streams:
                del self.active_streams[client_id]
    
    async def stop_streaming(self, client_id: str):
        if client_id in self.active_streams:
            stream_info = self.active_streams[client_id]
            stream_info["streaming_task"].cancel()
            stream_info["frame_sender_task"].cancel()
            del self.active_streams[client_id]
            
            # Send poison pill
            if client_id in self.frame_queues:
                await self.frame_queues[client_id].put(None)
            
            await self.send_message(client_id, {
                "type": "streaming_stopped",
                "timestamp": time.time()
            })
            logger.info(f"Stopped streaming for client {client_id}")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.cleanup_client(client_id)
    
    def cleanup_client(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.active_streams:
            stream_info = self.active_streams[client_id]
            if "streaming_task" in stream_info and stream_info["streaming_task"]:
                stream_info["streaming_task"].cancel()
            if "frame_sender_task" in stream_info and stream_info["frame_sender_task"]:
                stream_info["frame_sender_task"].cancel()
            del self.active_streams[client_id]
        if client_id in self.frame_queues:
            del self.frame_queues[client_id]


class OptimizedWebSocketFrameWriter:
    """✅ OPTIMIZED: Writer with reduced logging and binary frame format"""
    
    def __init__(self, server: OptimizedStreamingServer, client_id: str):
        self.server = server
        self.client_id = client_id
        self.frame_count = 0
        
    def __call__(self, frame_rgb: np.ndarray, fmt: str = "rgb"):
        # ✅ OPTIMIZATION 2: Remove per-frame logging
        # logger.info(f"Frame {self.frame_count} generated")  # Removed
        
        # Convert frame format if needed
        if fmt == "rgb":
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_rgb
        
        # ✅ OPTIMIZATION 4: Adaptive JPEG quality based on queue size
        queue_size = self.server.frame_queues[self.client_id].qsize() if self.client_id in self.server.frame_queues else 0
        
        # Reduce quality when queue is backing up to maintain framerate  
        if queue_size > 400:
            quality = 60  # Lower quality when heavily backed up
        elif queue_size > 250:
            quality = 70  # Medium quality when moderately backed up
        else:
            quality = 80  # Normal quality when queue is manageable
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpeg_bytes = buffer.tobytes()
        
        # 🔍 DEBUG: Monitor frame generation 
        if self.frame_count % 100 == 0:
            logger.info(f"🔍 Frame {self.frame_count}: {frame_rgb.shape}, Quality={quality}%, JPEG={len(jpeg_bytes)/1024:.1f}KB")
        
        # ✅ OPTIMIZATION 3: Store raw JPEG bytes instead of base64
        message = {
            "type": "frame",
            "frame_id": self.frame_count,
            "jpeg_data": jpeg_bytes,  # Raw bytes, not base64
            "timestamp": time.time()
        }
        
        # Put message in asyncio queue
        if self.client_id in self.server.frame_queues:
            try:
                self.server.frame_queues[self.client_id].put_nowait(message)
                
                # ✅ OPTIMIZATION 2: Only log occasionally with queue monitoring
                if self.frame_count % 100 == 0:
                    logger.info(f"Frame {self.frame_count} queued for {self.client_id} (quality: {quality}%, queue: {queue_size}/500)")
                    
            except asyncio.QueueFull:
                # ✅ OPTIMIZATION 5: Drop frames when queue is full instead of blocking
                logger.warning(f"Dropping frame {self.frame_count} for {self.client_id} - queue full")
        
        self.frame_count += 1
        
        # Update stream info
        if self.client_id in self.server.active_streams:
            self.server.active_streams[self.client_id]["frame_count"] = self.frame_count
    
    def close(self):
        message = {
            "type": "writer_closed",
            "total_frames": self.frame_count,
            "timestamp": time.time()
        }
        
        if self.client_id in self.server.frame_queues:
            try:
                self.server.frame_queues[self.client_id].put_nowait(message)
            except asyncio.QueueFull:
                pass


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ditto Streaming Server (Optimized)")
    parser.add_argument("--data_root", type=str, default="../checkpoints/ditto_trt_Ampere_Plus/", 
                      help="Path to TRT data_root")
    parser.add_argument("--cfg_pkl", type=str, default="../checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
                      help="Path to cfg_pkl")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Initialize optimized server
    server = OptimizedStreamingServer(args.cfg_pkl, args.data_root)
    
    # Run server
    logger.info(f"Starting Optimized Ditto Streaming Server on {args.host}:{args.port}")
    logger.info(f"Optimizations: asyncio.Queue, reduced logging, binary WebSocket, adaptive quality")
    
    uvicorn.run(
        server.app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main() 