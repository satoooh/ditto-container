from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
from asyncio import QueueEmpty, QueueFull
from functools import partial
from typing import Any, Dict, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from stream_pipeline_online import StreamSDK
from streaming_config import clamp_sampling_timesteps, parse_chunk_config
from streaming_protocol import build_binary_frame_payload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 50
DEFAULT_CHUNK_SLEEP = 0.0


class StreamingServer:
    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        *,
        default_chunk_config: tuple[int, int, int] = (3, 5, 2),
        default_sampling_steps: int = 30,
        chunk_sleep_s: float = DEFAULT_CHUNK_SLEEP,
    ):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.active_connections: Dict[str, WebSocket] = {}
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.frame_queues: Dict[str, asyncio.Queue] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue_depths: Dict[str, int] = {}
        self._queue_depth_lock = threading.Lock()
        self.default_chunk_config = default_chunk_config
        self.default_sampling_steps = default_sampling_steps
        self.default_chunk_sleep_s = max(0.0, chunk_sleep_s)

        # Initialize FastAPI app
        self.app = FastAPI(title="Ditto Streaming Server", version="1.0.0")
        self.setup_routes()
        # Startup hook for pre-warm
        self.app.add_event_handler("startup", self.on_startup)

    async def on_startup(self):
        """Pre-warm heavy libs (TRT/CUDA) and, if可能, run lightweight setup once."""

        async def _prewarm_async():
            loop = asyncio.get_event_loop()

            def _task():
                try:
                    logger.info("🔧 Prewarming StreamSDK (loading models/libs)...")
                    SDK = StreamSDK(self.cfg_pkl, self.data_root)
                    # Optional: run a light setup with example image if present
                    img = "/app/src/example/image.png"
                    if os.path.exists(img):
                        try:
                            SDK.setup(img, "/tmp/_prewarm.mp4")
                            logger.info("✅ Prewarm setup completed")
                        except Exception as e:
                            logger.warning(f"Prewarm setup skipped: {e}")
                    else:
                        logger.info(
                            "Prewarm: example image not found, skipped setup phase"
                        )
                except Exception as e:
                    logger.warning(f"Prewarm failed/skipped: {e}")

            return await loop.run_in_executor(None, _task)

        try:
            await _prewarm_async()
        except Exception as e:
            logger.warning(f"Prewarm task error: {e}")
        finally:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = None

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
                <p>Try the browser demo at <a href=\"/demo\">/demo</a>.</p>
                <p>WebSocket endpoint: ws(s)://YOUR_HOST/ws/{client_id}</p>
            </body>
            </html>
            """)

        @self.app.post("/upload")
        async def upload_files(
            audio: UploadFile | None = File(None),
            source: UploadFile | None = File(None),
        ):
            import pathlib
            import uuid

            base_dir = pathlib.Path("/app/data/uploads")
            base_dir.mkdir(parents=True, exist_ok=True)
            resp = {}

            async def save(upload: UploadFile, tag: str):
                suffix = pathlib.Path(upload.filename or "").suffix.lower() or (
                    ".wav" if tag == "audio" else ".png"
                )
                # very light validation
                if tag == "audio" and suffix not in [
                    ".wav",
                    ".mp3",
                    ".m4a",
                    ".flac",
                    ".ogg",
                ]:
                    return JSONResponse(
                        {"error": f"unsupported audio extension: {suffix}"},
                        status_code=400,
                    )
                if tag == "source" and suffix not in [".png", ".jpg", ".jpeg", ".webp"]:
                    return JSONResponse(
                        {"error": f"unsupported image extension: {suffix}"},
                        status_code=400,
                    )
                name = f"{uuid.uuid4().hex}_{tag}{suffix}"
                dst = base_dir / name
                with dst.open("wb") as f:
                    f.write(await upload.read())
                return str(dst)

            if audio is not None:
                path = await save(audio, "audio")
                if isinstance(path, JSONResponse):
                    return path
                resp["audio_path"] = path
            if source is not None:
                path = await save(source, "source")
                if isinstance(path, JSONResponse):
                    return path
                resp["source_path"] = path
            if not resp:
                return JSONResponse({"error": "no files provided"}, status_code=400)
            return resp

        @self.app.get("/demo")
        async def get_demo():
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
              <meta charset=\"utf-8\" />
              <title>Ditto Streaming Demo</title>
              <style>
                body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }
                .row { margin: 8px 0; }
                label { display: inline-block; width: 140px; }
                input { width: 520px; }
                #preview { max-width: 720px; border: 1px solid #ddd; }
                #log { height: 160px; width: 720px; overflow: auto; border: 1px solid #ddd; padding: 8px; white-space: pre-wrap; background: #fafafa; }
                button { margin-right: 8px; }
              </style>
            </head>
            <body>
              <h2>Ditto Streaming Demo (Browser)</h2>
              <div class=\"row\"><label>Client ID</label><input id=\"clientId\" value=\"web_demo\"></div>
              <div class=\"row\"><label>Audio Path (server)</label><input id=\"audioPath\" value=\"/app/src/example/audio.wav\"></div>
              <div class=\"row\"><label>Source Path (server)</label><input id=\"sourcePath\" value=\"/app/src/example/image.png\"></div>
              <div class=\"row\"><label>Upload (browser)</label>
                <input type=\"file\" id=\"upAudio\" accept=\"audio/*\"> 
                <input type=\"file\" id=\"upSource\" accept=\"image/*\">
                <button id=\"btnUpload\">Upload</button>
              </div>
              <div class=\"row\">
                <button id=\"btnConnect\">Connect</button>
                <button id=\"btnStart\" disabled>Start</button>
                <button id=\"btnStop\" disabled>Stop</button>
                <button id=\"btnDisconnect\" disabled>Disconnect</button>
              </div>
              <div class=\"row\"><img id=\"preview\" alt=\"frame preview\"/></div>
              <div class=\"row\"><div id=\"log\"></div></div>

              <script>
              const logEl = document.getElementById('log');
              const imgEl = document.getElementById('preview');
              const clientIdEl = document.getElementById('clientId');
              const audioPathEl = document.getElementById('audioPath');
              const sourcePathEl = document.getElementById('sourcePath');
              const btnConnect = document.getElementById('btnConnect');
              const btnStart = document.getElementById('btnStart');
              const btnStop = document.getElementById('btnStop');
              const btnDisconnect = document.getElementById('btnDisconnect');
              const btnUpload = document.getElementById('btnUpload');
              const upAudio = document.getElementById('upAudio');
              const upSource = document.getElementById('upSource');
              
              let ws = null;
              function log(msg){
                const t = new Date().toLocaleTimeString();
                logEl.textContent += `[${t}] ${msg}\n`;
                logEl.scrollTop = logEl.scrollHeight;
              }
              function wsUrlFor(clientId){
                const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
                return `${proto}://${location.host}/ws/${encodeURIComponent(clientId)}`;
              }
              btnUpload.onclick = async () => {
                try {
                  const fd = new FormData();
                  if(upAudio.files[0]) fd.append('audio', upAudio.files[0]);
                  if(upSource.files[0]) fd.append('source', upSource.files[0]);
                  if(!fd.has('audio') && !fd.has('source')){ log('No files selected'); return; }
                  const r = await fetch('/upload', { method: 'POST', body: fd });
                  const j = await r.json();
                  if(!r.ok){ log('Upload failed: ' + (j.error||r.status)); return; }
                  if(j.audio_path){ audioPathEl.value = j.audio_path; }
                  if(j.source_path){ sourcePathEl.value = j.source_path; }
                  log('Upload success');
                } catch (e) { log('Upload error'); console.error(e); }
              };
              btnConnect.onclick = () => {
                const url = wsUrlFor(clientIdEl.value || 'web_demo');
                ws = new WebSocket(url);
                ws.binaryType = 'arraybuffer';
                ws.onopen = () => { log('Connected: ' + url); btnStart.disabled=false; btnDisconnect.disabled=false; btnConnect.disabled=true; };
                ws.onclose = () => { log('Disconnected'); btnStart.disabled=true; btnStop.disabled=true; btnDisconnect.disabled=true; btnConnect.disabled=false; };
                ws.onerror = (e) => { log('WebSocket error'); console.error(e); };
                ws.onmessage = (ev) => {
                  try{
                    if (typeof ev.data === 'string') {
                      const msg = JSON.parse(ev.data);
                      if(msg.type === 'frame' && msg.frame_data){
                        const mime = msg.mime || 'image/webp';
                        imgEl.src = `data:${mime};base64,` + msg.frame_data; // legacy path
                      } else if(msg.type === 'streaming_started'){
                        log('Streaming started');
                      } else if(msg.type === 'streaming_completed'){
                        log('Streaming completed');
                      } else if(msg.type === 'metadata'){
                        log(`Metadata: duration=${msg.audio_duration?.toFixed?.(2)}s, frames=${msg.expected_frames}`);
                      } else if(msg.type === 'error'){
                        log('Server error: ' + msg.message);
                      }
                    } else {
                      // Binary frame: header (uint32 frame_id, double ts, uint32 len) + WebP bytes
                      const buf = ev.data; // ArrayBuffer
                      const dv = new DataView(buf);
                      const frameId = dv.getUint32(0, false);
                      const ts = dv.getFloat64(4, false);
                      const len = dv.getUint32(12, false);
                      const webp = buf.slice(16, 16 + len);
                      const blob = new Blob([webp], {type:'image/webp'});
                      const urlObj = URL.createObjectURL(blob);
                      imgEl.onload = () => URL.revokeObjectURL(urlObj);
                      imgEl.src = urlObj;
                    }
                  }catch(err){ console.error(err); }
                };
              };
              btnStart.onclick = () => {
                if(!ws || ws.readyState !== 1){ log('Not connected'); return; }
                const audio = audioPathEl.value.trim();
                const source = sourcePathEl.value.trim();
                const payload = { type: 'start_streaming', audio_path: audio, source_path: source, setup_kwargs: {}, run_kwargs: {}, binary: true };
                ws.send(JSON.stringify(payload));
                log('Sent start_streaming');
                btnStop.disabled=false;
              };
              btnStop.onclick = () => {
                if(!ws || ws.readyState !== 1){ log('Not connected'); return; }
                ws.send(JSON.stringify({type: 'stop_streaming'}));
                log('Sent stop_streaming');
              };
              btnDisconnect.onclick = () => { if(ws){ ws.close(); } };
              </script>
            </body>
            </html>
            """)

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.handle_websocket_connection(websocket, client_id)

    def _set_queue_depth(self, client_id: str, depth: int) -> None:
        with self._queue_depth_lock:
            if depth <= 0:
                self._queue_depths.pop(client_id, None)
            else:
                self._queue_depths[client_id] = depth

    def get_queue_depth(self, client_id: str) -> int:
        with self._queue_depth_lock:
            return self._queue_depths.get(client_id, 0)

    def enqueue_frame(self, client_id: str, message: Optional[Dict[str, Any]]) -> None:
        frame_queue = self.frame_queues.get(client_id)
        if not frame_queue:
            logger.debug(f"Frame queue missing for {client_id}, dropping message")
            return

        def _put() -> None:
            try:
                if message is None and frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except QueueEmpty:
                        pass
                frame_queue.put_nowait(message)
            except QueueFull:
                if message is not None:
                    logger.warning(
                        f"Dropping frame for {client_id}: queue full ({frame_queue.maxsize})"
                    )
            finally:
                self._set_queue_depth(client_id, frame_queue.qsize())

        target_loop = self.loop
        if target_loop and target_loop.is_running():
            target_loop.call_soon_threadsafe(_put)
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                f"Event loop not running when queuing frame for {client_id}; dropping message"
            )
            return

        if running_loop.is_running():
            _put()  # same thread as loop
        else:
            logger.debug(
                f"Event loop inactive when queuing frame for {client_id}; dropping message"
            )

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

        if message_type == "start_streaming":
            await self.start_streaming(client_id, data)
        elif message_type == "stop_streaming":
            await self.stop_streaming(client_id)
        elif message_type == "ping":
            await self.send_message(
                client_id, {"type": "pong", "timestamp": time.time()}
            )
        else:
            logger.warning(f"Unknown message type: {message_type}")

    async def start_streaming(self, client_id: str, config: Dict[str, Any]):
        if client_id in self.active_streams:
            await self.send_message(
                client_id,
                {
                    "type": "error",
                    "message": "Streaming already active for this client",
                },
            )
            return

        # Get streaming parameters
        audio_path = config.get("audio_path", "./example/audio.wav")
        source_path = config.get("source_path", "./example/image.png")

        # Validate files exist
        if not os.path.exists(audio_path):
            await self.send_message(
                client_id,
                {"type": "error", "message": f"Audio file not found: {audio_path}"},
            )
            return

        if not os.path.exists(source_path):
            await self.send_message(
                client_id,
                {"type": "error", "message": f"Source file not found: {source_path}"},
            )
            return

        logger.info(f"Starting streaming for client {client_id}")

        # Determine frame transport
        prefer_binary = bool(config.get("binary", True))

        # Send streaming started confirmation
        await self.send_message(
            client_id,
            {
                "type": "streaming_started",
                "audio_path": audio_path,
                "source_path": source_path,
                "timestamp": time.time(),
                "binary": prefer_binary,
            },
        )
        logger.info(f"Sent streaming started confirmation for client {client_id}")

        # Create frame queue for this client
        # Small buffer keeps latency low while absorbing short bursts
        self.frame_queues[client_id] = asyncio.Queue(maxsize=DEFAULT_QUEUE_SIZE)
        self._set_queue_depth(client_id, 0)
        logger.info(
            f"Created frame queue for client {client_id} (max={DEFAULT_QUEUE_SIZE})"
        )

        # Add client to active_streams BEFORE creating frame sender task
        # This ensures the frame sender worker can find the client in active_streams
        self.active_streams[client_id] = {
            "streaming_task": None,  # Will be set later
            "frame_sender_task": None,  # Will be set later
            "start_time": time.time(),
            "frame_count": 0,
            "prefer_binary": prefer_binary,
        }
        logger.info(f"Added client {client_id} to active_streams")

        # Start frame sender task
        frame_sender_task = asyncio.create_task(self.frame_sender_worker(client_id))
        logger.info(f"Created frame sender task for client {client_id}")

        # Add done callback for frame sender task
        def frame_sender_done_callback(task):
            if task.exception():
                logger.error(
                    f"❌ Frame sender task failed for {client_id}: {task.exception()}"
                )
            else:
                logger.info(
                    f"✅ Frame sender task completed successfully for {client_id}"
                )

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
                    logger.error(
                        f"Streaming task failed for {client_id}: {task.exception()}"
                    )
                else:
                    logger.info(
                        f"Streaming task completed successfully for {client_id}"
                    )

            streaming_task.add_done_callback(task_done_callback)

        except Exception as e:
            logger.error(f"Error creating streaming task for {client_id}: {e}")
            # Clean up active_streams if task creation failed
            if client_id in self.active_streams:
                del self.active_streams[client_id]
            await self.send_message(
                client_id,
                {
                    "type": "error",
                    "message": f"Failed to create streaming task: {str(e)}",
                },
            )
            return

        # Update active_streams with the actual task objects
        self.active_streams[client_id]["streaming_task"] = streaming_task
        self.active_streams[client_id]["frame_sender_task"] = frame_sender_task
        logger.info(f"Updated active_streams with task objects for {client_id}")

    async def frame_sender_worker(self, client_id: str):
        """Async worker that forwards queued frames to the WebSocket client."""
        logger.info(f"🚀 STARTING frame_sender_worker for {client_id}")
        frames_sent = 0
        frame_queue = self.frame_queues.get(client_id)

        if frame_queue is None:
            logger.error(f"❌ Frame queue missing for {client_id}")
            return

        try:
            while True:
                try:
                    message = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                    self._set_queue_depth(client_id, frame_queue.qsize())
                except asyncio.TimeoutError:
                    if client_id not in self.active_streams:
                        break
                    continue

                if message is None:
                    logger.debug(f"Poison pill received for {client_id}")
                    break

                msg_type = message.get("type")
                if msg_type == "frame" and "frame_bytes" in message:
                    prefer_binary = self.active_streams.get(client_id, {}).get(
                        "prefer_binary", True
                    )
                    if prefer_binary:
                        await self.send_binary_frame(client_id, message)
                    else:
                        await self.send_legacy_frame(client_id, message)
                    frames_sent += 1
                    if frames_sent % 50 == 0:
                        logger.info(
                            f"📈 Sent {frames_sent} frames to {client_id} (queue={frame_queue.qsize()})"
                        )
                else:
                    await self.send_message(client_id, message)
        except Exception as exc:
            import traceback

            logger.error(f"❌ Frame sender worker error for {client_id}: {exc}")
            logger.error(f"❌ Worker traceback: {traceback.format_exc()}")
        finally:
            self._set_queue_depth(client_id, 0)
            logger.info(
                f"🏁 EXITING frame_sender_worker for {client_id}, frames sent={frames_sent}"
            )

    async def send_legacy_frame(self, client_id: str, message: Dict[str, Any]):
        if client_id not in self.active_connections:
            return
        try:
            b64 = base64.b64encode(message["frame_bytes"]).decode("utf-8")
            legacy_msg = {
                "type": "frame",
                "frame_id": message.get("frame_id"),
                "frame_data": b64,
                "timestamp": message.get("timestamp"),
                "mime": "image/webp",
            }
            await self.send_message(client_id, legacy_msg)
        except Exception as exc:
            logger.error(f"❌ Error sending legacy frame to {client_id}: {exc}")

    async def send_binary_frame(self, client_id: str, message: Dict[str, Any]):
        """Send a single frame as binary WebSocket message.
        Format: frame_id(uint32 BE) + timestamp(double BE) + data_len(uint32 BE) + WebP bytes
        """
        if client_id not in self.active_connections:
            return
        try:
            frame_id = int(message.get("frame_id", 0))
            ts = float(message.get("timestamp", time.time()))
            frame_bytes = message.get("frame_bytes", b"")
            if not isinstance(frame_bytes, (bytes, bytearray)):
                frame_bytes = bytes(frame_bytes)
            payload = build_binary_frame_payload(frame_id, ts, frame_bytes)
            await self.active_connections[client_id].send_bytes(payload)
        except Exception as e:
            logger.error(f"❌ Error sending binary frame to {client_id}: {e}")
            self.cleanup_client(client_id)

    async def run_streaming_pipeline(
        self, client_id: str, audio_path: str, source_path: str, config: Dict[str, Any]
    ):
        try:
            logger.info(f"🚀 ENTERING run_streaming_pipeline for client {client_id}")
            logger.info(f"Config received: {config}")
            logger.info(f"Audio path: {audio_path}, Source path: {source_path}")

            # Immediate test to verify we're executing
            await asyncio.sleep(0.001)
            logger.info(f"✅ Async execution confirmed for client {client_id}")

        except Exception as immediate_error:
            logger.error(
                f"❌ Immediate error in run_streaming_pipeline for {client_id}: {immediate_error}"
            )
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
            setup_kwargs = dict(config.get("setup_kwargs", {}))
            sampling_timesteps = clamp_sampling_timesteps(
                setup_kwargs.get("sampling_timesteps"),
                default=self.default_sampling_steps,
            )
            setup_kwargs["sampling_timesteps"] = sampling_timesteps
            # Use a temporary output path (won't be used since we replace the writer)
            temp_output_path = f"/tmp/streaming_output_{client_id}.mp4"

            logger.info(f"About to call SDK.setup() for client {client_id}")
            logger.info(f"Source path: {source_path}")
            logger.info(f"Temp output path: {temp_output_path}")
            logger.info(f"Setup kwargs: {setup_kwargs}")

            # This is where it might be getting stuck - let's run it in executor
            loop = asyncio.get_event_loop()
            setup_task = partial(
                SDK.setup, source_path, temp_output_path, **setup_kwargs
            )
            await loop.run_in_executor(None, setup_task)

            logger.info(f"SDK.setup() completed successfully for client {client_id}")

            # Replace the writer with our WebSocket writer
            logger.info(
                f"Replacing writer with WebSocket writer for client {client_id}"
            )
            SDK.writer = websocket_writer
            logger.info(f"Writer replaced successfully for client {client_id}")

            # Load and process audio
            logger.info(f"About to load audio file: {audio_path}")
            import math

            import librosa

            audio, sr = librosa.core.load(audio_path, sr=16000)
            num_f = math.ceil(len(audio) / 16000 * 25)
            logger.info(
                f"Audio loaded: {len(audio)} samples, {len(audio) / sr:.2f}s, {num_f} frames"
            )

            # Setup audio processing
            run_kwargs = dict(config.get("run_kwargs", {}))
            fade_in = run_kwargs.get("fade_in", -1)
            fade_out = run_kwargs.get("fade_out", -1)
            ctrl_info = run_kwargs.get("ctrl_info", {})
            chunksize_tuple = parse_chunk_config(
                run_kwargs.get("chunksize", self.default_chunk_config),
                fallback=self.default_chunk_config,
            )
            run_kwargs["chunksize"] = chunksize_tuple
            # Ensure run_kwargs carries the same sampling value so SDK.run_chunk can refer to it if needed
            run_kwargs["sampling_timesteps"] = sampling_timesteps
            chunk_sleep_s = run_kwargs.get("chunk_sleep_s", self.default_chunk_sleep_s)
            chunk_sleep_s = max(0.0, float(chunk_sleep_s))

            logger.info(f"About to call SDK.setup_Nd() for client {client_id}")
            SDK.setup_Nd(
                N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info
            )
            logger.info(f"SDK.setup_Nd() completed for client {client_id}")

            # Send metadata to client
            logger.info(f"Sending metadata to client {client_id}")
            await self.send_message(
                client_id,
                {
                    "type": "metadata",
                    "audio_duration": len(audio) / sr,
                    "expected_frames": num_f,
                    "timestamp": time.time(),
                },
            )
            logger.info(f"Metadata sent to client {client_id}")

            # Process audio chunks
            logger.info(
                f"Starting audio processing for client {client_id}, online_mode: {SDK.online_mode}"
            )

            if SDK.online_mode:
                chunksize = chunksize_tuple
                audio = np.concatenate(
                    [np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0
                )
                split_len = int(sum(chunksize) * 0.04 * 16000) + 80

                logger.info(
                    f"Processing {len(range(0, len(audio), chunksize[1] * 640))} audio chunks"
                )

                for i, chunk_start in enumerate(
                    range(0, len(audio), chunksize[1] * 640)
                ):
                    if client_id not in self.active_streams:
                        logger.info(
                            f"Client {client_id} disconnected, stopping processing"
                        )
                        break  # Client disconnected

                    audio_chunk = audio[chunk_start : chunk_start + split_len]
                    if len(audio_chunk) < split_len:
                        audio_chunk = np.pad(
                            audio_chunk,
                            (0, split_len - len(audio_chunk)),
                            mode="constant",
                        )

                    logger.info(f"Processing chunk {i + 1} for client {client_id}")

                    # Run the chunk processing in a thread executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, SDK.run_chunk, audio_chunk, chunksize
                    )

                    if chunk_sleep_s:
                        await asyncio.sleep(chunk_sleep_s)
            else:
                # Offline mode
                logger.info(f"Running offline mode for client {client_id}")
                aud_feat = SDK.wav2feat.wav2feat(audio)
                SDK.audio2motion_queue.put(aud_feat)

            # Close pipeline
            logger.info(f"Closing pipeline for client {client_id}")
            SDK.close()

            # Send completion message
            await self.send_message(
                client_id, {"type": "streaming_completed", "timestamp": time.time()}
            )

        except Exception as e:
            logger.error(f"Streaming error for client {client_id}: {e}")
            await self.send_message(
                client_id, {"type": "error", "message": f"Streaming error: {str(e)}"}
            )
        finally:
            # Send poison pill to frame sender
            if client_id in self.frame_queues:
                self.enqueue_frame(client_id, None)

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
                self.enqueue_frame(client_id, None)

            await self.send_message(
                client_id, {"type": "streaming_stopped", "timestamp": time.time()}
            )
            logger.info(f"Stopped streaming for client {client_id}")

    async def send_message(self, client_id: str, message: Dict[str, Any]):
        if client_id in self.active_connections:
            try:
                message_type = message.get("type", "unknown")
                await self.active_connections[client_id].send_text(json.dumps(message))
                if message_type in {
                    "streaming_started",
                    "metadata",
                    "streaming_completed",
                    "streaming_stopped",
                }:
                    logger.info(f"Sent {message_type} to {client_id}")
            except Exception as e:
                logger.error(f"❌ Error sending message to {client_id}: {e}")
                logger.error(f"❌ Message type was: {message.get('type', 'unknown')}")
                import traceback

                logger.error(f"❌ Send traceback: {traceback.format_exc()}")
                self.cleanup_client(client_id)
        else:
            logger.error(
                f"❌ Client {client_id} not in active_connections when sending {message.get('type', 'unknown')}"
            )

    def cleanup_client(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.active_streams:
            stream_info = self.active_streams[client_id]
            stream_info["streaming_task"].cancel()
            stream_info["frame_sender_task"].cancel()
            del self.active_streams[client_id]
        if client_id in self.frame_queues:
            self.enqueue_frame(client_id, None)
            del self.frame_queues[client_id]
            self._set_queue_depth(client_id, 0)


class WebSocketFrameWriter:
    """Custom writer that sends frames via WebSocket instead of writing to file"""

    def __init__(self, server: StreamingServer, client_id: str):
        self.server = server
        self.client_id = client_id
        self.frame_count = 0

    def __call__(self, frame_rgb: np.ndarray, fmt: str = "rgb"):
        # Convert to WebP bytes for binary WebSocket transmission
        if fmt == "rgb":
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_rgb

        queue_depth = self.server.get_queue_depth(self.client_id)
        if queue_depth > DEFAULT_QUEUE_SIZE * 0.8:
            quality = 60
        elif queue_depth > DEFAULT_QUEUE_SIZE * 0.5:
            quality = 75
        else:
            quality = 85

        # Encode frame as WebP (adaptive品質)
        _, buffer = cv2.imencode(
            ".webp", frame_bgr, [cv2.IMWRITE_WEBP_QUALITY, quality]
        )
        image_bytes = buffer.tobytes()

        # Send frame via WebSocket using queue
        message = {
            "type": "frame",
            "frame_id": self.frame_count,
            "frame_bytes": image_bytes,
            "timestamp": time.time(),
        }

        self.server.enqueue_frame(self.client_id, message)
        if self.frame_count % 100 == 0:
            logger.info(
                f"Enqueued frame {self.frame_count} for {self.client_id} "
                f"(WebP quality={quality}%, queue≈{queue_depth})"
            )

        self.frame_count += 1

        # Update stream info
        if self.client_id in self.server.active_streams:
            self.server.active_streams[self.client_id]["frame_count"] = self.frame_count

    def close(self):
        # Send completion message
        message = {
            "type": "writer_closed",
            "total_frames": self.frame_count,
            "timestamp": time.time(),
        }

        self.server.enqueue_frame(self.client_id, message)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ditto Streaming Server")
    parser.add_argument(
        "--data_root",
        type=str,
        default="../checkpoints/ditto_trt_Ampere_Plus/",
        help="Path to TRT data_root",
    )
    parser.add_argument(
        "--cfg_pkl",
        type=str,
        default="../checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
        help="Path to cfg_pkl",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--online-sampling-steps",
        type=int,
        default=30,
        help="Sampling timesteps for online diffusion (lower = faster).",
    )
    parser.add_argument(
        "--online-chunk-config",
        type=str,
        default="3,5,2",
        help="Default chunk config pre,main,post for streaming mode.",
    )
    parser.add_argument(
        "--chunk-sleep-ms",
        type=float,
        default=0.0,
        help="Sleep in milliseconds between chunk submissions (safety valve).",
    )

    args = parser.parse_args()

    # Initialize server
    chunk_tuple = parse_chunk_config(args.online_chunk_config)
    sampling_steps = clamp_sampling_timesteps(args.online_sampling_steps)
    chunk_sleep_s = max(0.0, args.chunk_sleep_ms / 1000.0)

    server = StreamingServer(
        args.cfg_pkl,
        args.data_root,
        default_chunk_config=chunk_tuple,
        default_sampling_steps=sampling_steps,
        chunk_sleep_s=chunk_sleep_s,
    )

    # Run server
    logger.info(f"Starting Ditto Streaming Server on {args.host}:{args.port}")
    logger.info(f"Using model: {args.data_root}")
    logger.info(f"Using config: {args.cfg_pkl}")

    uvicorn.run(server.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
