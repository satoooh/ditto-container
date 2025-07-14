import asyncio
import json
import logging
import os
import time
import queue
import threading
from typing import Dict, Any
import base64
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from stream_pipeline_online import StreamSDK


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingServer:
    def __init__(self, cfg_pkl: str, data_root: str):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.active_connections: Dict[str, WebSocket] = {}
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}  # Frame queues for each client
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Ditto Streaming Server", version="1.0.0")
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.get("/")
        async def get_homepage():
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Ditto Streaming Server</title>
            </head>
            <body>
                <h1>Ditto Streaming Server</h1>
                <p>Server is running. Use WebSocket client to connect.</p>
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
                # Wait for messages from client
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
        
        # Test async execution right here
        logger.info(f"🔍 Testing async in handle_message for {client_id}")
        await asyncio.sleep(0.001)
        logger.info(f"✅ Async works in handle_message for {client_id}")
        
        if message_type == "start_streaming":
            logger.info(f"📞 About to call start_streaming for {client_id}")
            await self.start_streaming(client_id, data)
            logger.info(f"📞 start_streaming returned for {client_id}")
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
        logger.info(f"Sent streaming started confirmation for client {client_id}")
        
        # Create frame queue for this client  
        # Increased queue size to handle full streams without dropping frames
        # 410 frames * ~200KB = ~82MB memory per client (reasonable for modern systems)
        self.frame_queues[client_id] = queue.Queue(maxsize=500)
        logger.info(f"Created frame queue for client {client_id}")
        
        # Add client to active_streams BEFORE creating frame sender task
        # This ensures the frame sender worker can find the client in active_streams
        self.active_streams[client_id] = {
            "streaming_task": None,  # Will be set later
            "frame_sender_task": None,  # Will be set later
            "start_time": time.time(),
            "frame_count": 0
        }
        logger.info(f"Added client {client_id} to active_streams")
        
        # Start frame sender task
        frame_sender_task = asyncio.create_task(
            self.frame_sender_worker(client_id)
        )
        logger.info(f"Created frame sender task for client {client_id}")
        
        # Add done callback for frame sender task
        def frame_sender_done_callback(task):
            if task.exception():
                logger.error(f"❌ Frame sender task failed for {client_id}: {task.exception()}")
            else:
                logger.info(f"✅ Frame sender task completed successfully for {client_id}")
        
        frame_sender_task.add_done_callback(frame_sender_done_callback)
        
        # Start streaming task with exception handling
        try:
            streaming_task = asyncio.create_task(
                self.run_streaming_pipeline(client_id, audio_path, source_path, config)
            )
            logger.info(f"Created streaming pipeline task for client {client_id}")
            
            # Add done callback to catch any exceptions
            def task_done_callback(task):
                if task.exception():
                    logger.error(f"Streaming task failed for {client_id}: {task.exception()}")
                else:
                    logger.info(f"Streaming task completed successfully for {client_id}")
            
            streaming_task.add_done_callback(task_done_callback)
            
            # Test that async tasks are working - simple test task
            async def test_task():
                await asyncio.sleep(0.1)
                logger.info(f"Test task executed successfully for {client_id}")
            
            test_task_obj = asyncio.create_task(test_task())
            logger.info(f"Created test task for {client_id}")
            
            # Try to immediately await a simple task to see if it works
            try:
                logger.info(f"🧪 Testing immediate async execution for {client_id}")
                await asyncio.sleep(0.001)
                logger.info(f"✅ Immediate async works for {client_id}")
            except Exception as e:
                logger.error(f"❌ Immediate async failed for {client_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error creating streaming task for {client_id}: {e}")
            # Clean up active_streams if task creation failed
            if client_id in self.active_streams:
                del self.active_streams[client_id]
            await self.send_message(client_id, {
                "type": "error",
                "message": f"Failed to create streaming task: {str(e)}"
            })
            return
        
        # Update active_streams with the actual task objects
        self.active_streams[client_id]["streaming_task"] = streaming_task
        self.active_streams[client_id]["frame_sender_task"] = frame_sender_task
        logger.info(f"Updated active_streams with task objects for {client_id}")
    
    async def frame_sender_worker(self, client_id: str):
        """Worker to send frames from queue via WebSocket"""
        logger.info(f"🚀 STARTING frame_sender_worker for {client_id}")
        frames_sent = 0
        
        try:
            # Continue until poison pill received, regardless of active_streams status
            # This ensures all queued frames are sent even after pipeline completes
            while True:
                try:
                    # Get frame from queue (non-blocking)
                    frame_queue = self.frame_queues.get(client_id)
                    if not frame_queue:
                        logger.error(f"❌ No frame queue found for {client_id}")
                        break
                    
                    queue_size = frame_queue.qsize()
                    if queue_size > 0:
                        logger.info(f"📊 Queue size for {client_id}: {queue_size}")
                        
                    # Use non-blocking get_nowait() instead of blocking get()
                    message = frame_queue.get_nowait()
                    if message is None:  # Poison pill - proper exit condition
                        logger.info(f"💊 Poison pill received for {client_id}, processed all frames")
                        break
                    
                    logger.info(f"📤 Sending message type '{message.get('type')}' to {client_id}")
                    await self.send_message(client_id, message)
                    frames_sent += 1
                    
                    if frames_sent % 10 == 0:
                        logger.info(f"📈 Sent {frames_sent} messages to {client_id}")
                    
                except queue.Empty:
                    # Queue is empty, sleep a bit and continue
                    # Don't exit - pipeline might still be generating frames
                    await asyncio.sleep(0.01)  # Non-blocking sleep
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in frame sender for {client_id}: {e}")
                    logger.error(f"❌ Exception type: {type(e)}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    break
                    
        except Exception as e:
            logger.error(f"❌ Frame sender worker error for {client_id}: {e}")
            import traceback
            logger.error(f"❌ Worker traceback: {traceback.format_exc()}")
        
        logger.info(f"🏁 EXITING frame_sender_worker for {client_id}, sent {frames_sent} messages")
    
    async def run_streaming_pipeline(self, client_id: str, audio_path: str, source_path: str, config: Dict[str, Any]):
        try:
            logger.info(f"🚀 ENTERING run_streaming_pipeline for client {client_id}")
            logger.info(f"Config received: {config}")
            logger.info(f"Audio path: {audio_path}, Source path: {source_path}")
            
            # Immediate test to verify we're executing
            await asyncio.sleep(0.001)
            logger.info(f"✅ Async execution confirmed for client {client_id}")
        
        except Exception as immediate_error:
            logger.error(f"❌ Immediate error in run_streaming_pipeline for {client_id}: {immediate_error}")
            raise
        
        try:
            logger.info(f"About to initialize SDK for client {client_id}")
            logger.info(f"Using cfg_pkl: {self.cfg_pkl}")
            logger.info(f"Using data_root: {self.data_root}")
            
            # Initialize SDK
            SDK = StreamSDK(self.cfg_pkl, self.data_root)
            logger.info(f"SDK initialized successfully for client {client_id}")
            
            # Create custom writer that sends frames via WebSocket
            websocket_writer = WebSocketFrameWriter(self, client_id)
            
            # Setup the pipeline with custom writer
            setup_kwargs = config.get("setup_kwargs", {})
            # Use a temporary output path (won't be used since we replace the writer)
            temp_output_path = f"/tmp/streaming_output_{client_id}.mp4"
            
            logger.info(f"About to call SDK.setup() for client {client_id}")
            logger.info(f"Source path: {source_path}")
            logger.info(f"Temp output path: {temp_output_path}")
            logger.info(f"Setup kwargs: {setup_kwargs}")
            
            # This is where it might be getting stuck - let's run it in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, SDK.setup, source_path, temp_output_path, **setup_kwargs)
            
            logger.info(f"SDK.setup() completed successfully for client {client_id}")
            
            # Replace the writer with our WebSocket writer
            logger.info(f"Replacing writer with WebSocket writer for client {client_id}")
            SDK.writer = websocket_writer
            logger.info(f"Writer replaced successfully for client {client_id}")
            
            # Load and process audio
            logger.info(f"About to load audio file: {audio_path}")
            import librosa
            import math
            
            audio, sr = librosa.core.load(audio_path, sr=16000)
            num_f = math.ceil(len(audio) / 16000 * 25)
            logger.info(f"Audio loaded: {len(audio)} samples, {len(audio)/sr:.2f}s, {num_f} frames")
            
            # Setup audio processing
            run_kwargs = config.get("run_kwargs", {})
            fade_in = run_kwargs.get("fade_in", -1)
            fade_out = run_kwargs.get("fade_out", -1)
            ctrl_info = run_kwargs.get("ctrl_info", {})
            
            logger.info(f"About to call SDK.setup_Nd() for client {client_id}")
            SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
            logger.info(f"SDK.setup_Nd() completed for client {client_id}")
            
            # Send metadata to client
            logger.info(f"Sending metadata to client {client_id}")
            await self.send_message(client_id, {
                "type": "metadata",
                "audio_duration": len(audio) / sr,
                "expected_frames": num_f,
                "timestamp": time.time()
            })
            logger.info(f"Metadata sent to client {client_id}")
            
            # Process audio chunks
            logger.info(f"Starting audio processing for client {client_id}, online_mode: {SDK.online_mode}")
            
            if SDK.online_mode:
                chunksize = run_kwargs.get("chunksize", (3, 5, 2))
                audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
                split_len = int(sum(chunksize) * 0.04 * 16000) + 80
                
                logger.info(f"Processing {len(range(0, len(audio), chunksize[1] * 640))} audio chunks")
                
                for i, chunk_start in enumerate(range(0, len(audio), chunksize[1] * 640)):
                    if client_id not in self.active_streams:
                        logger.info(f"Client {client_id} disconnected, stopping processing")
                        break  # Client disconnected
                        
                    audio_chunk = audio[chunk_start:chunk_start + split_len]
                    if len(audio_chunk) < split_len:
                        audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
                    
                    logger.info(f"Processing chunk {i+1} for client {client_id}")
                    
                    # Run the chunk processing in a thread executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, SDK.run_chunk, audio_chunk, chunksize)
                    
                    # Small delay to prevent overwhelming the WebSocket
                    await asyncio.sleep(0.01)
            else:
                # Offline mode
                logger.info(f"Running offline mode for client {client_id}")
                aud_feat = SDK.wav2feat.wav2feat(audio)
                SDK.audio2motion_queue.put(aud_feat)
            
            # Close pipeline
            logger.info(f"Closing pipeline for client {client_id}")
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
                try:
                    self.frame_queues[client_id].put_nowait(None)
                except queue.Full:
                    pass
            
            # Clean up
            if client_id in self.active_streams:
                del self.active_streams[client_id]
    
    async def stop_streaming(self, client_id: str):
        if client_id in self.active_streams:
            stream_info = self.active_streams[client_id]
            stream_info["streaming_task"].cancel()
            stream_info["frame_sender_task"].cancel()
            del self.active_streams[client_id]
            
            # Send poison pill to frame queue
            if client_id in self.frame_queues:
                try:
                    self.frame_queues[client_id].put_nowait(None)
                except queue.Full:
                    pass
            
            await self.send_message(client_id, {
                "type": "streaming_stopped",
                "timestamp": time.time()
            })
            logger.info(f"Stopped streaming for client {client_id}")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        if client_id in self.active_connections:
            try:
                message_type = message.get("type", "unknown")
                logger.info(f"📡 Sending {message_type} to {client_id}")
                
                await self.active_connections[client_id].send_text(json.dumps(message))
                
                logger.info(f"✅ Successfully sent {message_type} to {client_id}")
            except Exception as e:
                logger.error(f"❌ Error sending message to {client_id}: {e}")
                logger.error(f"❌ Message type was: {message.get('type', 'unknown')}")
                import traceback
                logger.error(f"❌ Send traceback: {traceback.format_exc()}")
                self.cleanup_client(client_id)
        else:
            logger.error(f"❌ Client {client_id} not in active_connections when sending {message.get('type', 'unknown')}")
    
    def cleanup_client(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.active_streams:
            stream_info = self.active_streams[client_id]
            stream_info["streaming_task"].cancel()
            stream_info["frame_sender_task"].cancel()
            del self.active_streams[client_id]
        if client_id in self.frame_queues:
            # Send poison pill and clean up queue
            try:
                self.frame_queues[client_id].put_nowait(None)
            except queue.Full:
                pass
            del self.frame_queues[client_id]


class WebSocketFrameWriter:
    """Custom writer that sends frames via WebSocket instead of writing to file"""
    
    def __init__(self, server: StreamingServer, client_id: str):
        self.server = server
        self.client_id = client_id
        self.frame_count = 0
        
    def __call__(self, frame_rgb: np.ndarray, fmt: str = "rgb"):
        logger.info(f"WebSocketFrameWriter called for frame {self.frame_count} (client: {self.client_id})")
        
        # Convert frame to base64 for WebSocket transmission
        if fmt == "rgb":
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_rgb
            
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame via WebSocket using queue
        message = {
            "type": "frame",
            "frame_id": self.frame_count,
            "frame_data": frame_base64,
            "timestamp": time.time()
        }
        
        # Put message in queue for async processing
        if self.client_id in self.server.frame_queues:
            try:
                self.server.frame_queues[self.client_id].put_nowait(message)
                logger.info(f"Frame {self.frame_count} queued for client {self.client_id}")
            except queue.Full:
                logger.warning(f"Frame queue full for client {self.client_id}, dropping frame {self.frame_count}")
        else:
            logger.error(f"No frame queue found for client {self.client_id}")
        
        self.frame_count += 1
        
        # Update stream info
        if self.client_id in self.server.active_streams:
            self.server.active_streams[self.client_id]["frame_count"] = self.frame_count
    
    def close(self):
        # Send completion message
        message = {
            "type": "writer_closed",
            "total_frames": self.frame_count,
            "timestamp": time.time()
        }
        
        # Put message in queue for async processing
        if self.client_id in self.server.frame_queues:
            try:
                self.server.frame_queues[self.client_id].put_nowait(message)
            except queue.Full:
                logger.warning(f"Frame queue full for client {self.client_id}, dropping close message")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ditto Streaming Server")
    parser.add_argument("--data_root", type=str, default="../checkpoints/ditto_trt_Ampere_Plus/", 
                      help="Path to TRT data_root")
    parser.add_argument("--cfg_pkl", type=str, default="../checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
                      help="Path to cfg_pkl")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Initialize server
    server = StreamingServer(args.cfg_pkl, args.data_root)
    
    # Run server
    logger.info(f"Starting Ditto Streaming Server on {args.host}:{args.port}")
    logger.info(f"Using model: {args.data_root}")
    logger.info(f"Using config: {args.cfg_pkl}")
    
    uvicorn.run(
        server.app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main() 