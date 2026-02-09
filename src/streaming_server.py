from __future__ import annotations

import asyncio
import logging
import os
import types
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import librosa
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from stream_pipeline_online import StreamSDK
from streaming_config import clamp_sampling_timesteps, clamp_scale, parse_chunk_config
from webrtc.monitor import ConnectionMonitor
from webrtc.tracks import AudioArrayTrack, VideoFrameTrack
from webrtc.validators import OfferValidator, ValidatedOffer

logger = logging.getLogger(__name__)


class OfferPayload(BaseModel):
    sdp: str
    type: str
    audio_path: str
    source_path: str
    setup_kwargs: Dict[str, Any] = {}
    run_kwargs: Dict[str, Any] = {}


class WebRTCFrameWriter:
    """Writer injected into StreamSDK to forward frames to WebRTC track."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        video_track: VideoFrameTrack,
        frame_scale: float,
    ):
        self._loop = loop
        self._video_track = video_track
        self._frame_scale = frame_scale

    def __call__(self, frame_rgb: np.ndarray, fmt: str = "rgb"):
        frame_bgr = (
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) if fmt == "rgb" else frame_rgb
        )
        if self._frame_scale != 1.0:
            h, w = frame_bgr.shape[:2]
            new_size = (
                max(1, int(w * self._frame_scale)),
                max(1, int(h * self._frame_scale)),
            )
            frame_bgr = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)
        self._video_track.enqueue(frame_bgr)


class StreamingServer:
    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        *,
        default_chunk_config: Optional[tuple[int, int, int]] = (3, 5, 2),
        default_sampling_steps: Optional[int] = None,
        chunk_sleep_s: Optional[float] = 0.01,
        frame_scale: Optional[float] = None,
    ):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.default_chunk_config = default_chunk_config
        self.default_sampling_steps = default_sampling_steps
        self.default_chunk_sleep_s = chunk_sleep_s
        self.default_frame_scale = clamp_scale(frame_scale) if frame_scale else 1.0

        self.app = FastAPI(title="Ditto WebRTC Streaming Server", version="2.0.0")
        self.validator = OfferValidator()
        self.monitor = ConnectionMonitor()
        self._pcs: set[RTCPeerConnection] = set()
        self.setup_routes()
        # STUN を無効にし、ホスト内接続を優先（外部接続には TURN を追加する）
        self.rtc_configuration = RTCConfiguration(iceServers=[])

    def setup_routes(self) -> None:
        @self.app.exception_handler(RequestValidationError)
        async def validation_handler(request, exc: RequestValidationError):
            return JSONResponse(
                status_code=400,
                content={"detail": "invalid request", "errors": exc.errors()},
            )

        @self.app.get("/")
        async def root() -> Dict[str, str]:
            return {"message": "Ditto WebRTC streaming server"}

        @self.app.post("/upload")
        async def upload_files(
            audio: UploadFile | None = File(default=None),
            source: UploadFile | None = File(default=None),
        ) -> JSONResponse:
            base_dir = Path(os.getenv("UPLOAD_DIR", "/app/data/uploads"))
            base_dir.mkdir(parents=True, exist_ok=True)
            response: Dict[str, str] = {}

            async def _save(upload: UploadFile, suffixes: tuple[str, ...]) -> str:
                suffix = Path(upload.filename or "").suffix.lower()
                if suffix not in suffixes:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported extension: {suffix}"
                    )
                destination = base_dir / upload.filename
                try:
                    with destination.open("wb") as out:
                        out.write(await upload.read())
                except Exception as exc:  # pragma: no cover - narrowed by tests
                    if destination.exists():
                        destination.unlink(missing_ok=True)
                    raise HTTPException(status_code=500, detail="failed to save file") from exc
                return str(destination)

            if audio:
                response["audio_path"] = await _save(
                    audio, (".wav", ".mp3", ".m4a", ".flac", ".ogg")
                )
            if source:
                response["source_path"] = await _save(
                    source, (".png", ".jpg", ".jpeg", ".webp")
                )
            if not response:
                raise HTTPException(status_code=400, detail="no files provided")
            return JSONResponse(response)

        @self.app.get("/demo")
        async def demo() -> HTMLResponse:
            return HTMLResponse(self._render_demo_page())

        @self.app.post("/webrtc/offer")
        async def webrtc_offer(payload: OfferPayload) -> Dict[str, str]:
            return await self.handle_offer(payload)

    def _validate_offer(self, payload: Dict[str, Any]) -> ValidatedOffer:
        try:
            return self.validator.validate(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    def _render_demo_page(self) -> str:
        return """<!DOCTYPE html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Ditto WebRTC Demo</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 20px; }
      .row { margin-bottom: 8px; }
      label { display: inline-block; width: 160px; }
      input { width: 480px; }
      video { max-width: 640px; border: 1px solid #ddd; background: #000; }
    </style>
  </head>
  <body>
    <h2>Ditto WebRTC Streaming</h2>
    <div class=\"row\"><label>Audio Path</label><input id=\"audioPath\" value=\"/app/src/example/audio.wav\"></div>
    <div class=\"row\"><label>Source Path</label><input id=\"sourcePath\" value=\"/app/src/example/image.png\"></div>
    <div class=\"row\"><label>Frame Scale</label><input id=\"frameScale\" value=\"0.5\"></div>
    <div class=\"row\"><label>Sampling Steps</label><input id=\"samplingSteps\" value=\"12\"></div>
    <div class=\"row\"><label>Upload Audio</label><input type=\"file\" id=\"fileAudio\" accept=\"audio/*\"></div>
    <div class=\"row\"><label>Upload Image</label><input type=\"file\" id=\"fileImage\" accept=\"image/*\"></div>
    <div class=\"row\">
      <button id=\"btnUpload\">Upload Files</button>
      <button id=\"btnStart\">Start</button>
      <button id=\"btnStop\" disabled>Stop</button>
    </div>
    <div class=\"row\">
      <video id=\"remoteVideo\" autoplay playsinline muted></video>
    </div>
    <div class=\"row\">
      <audio id=\"remoteAudio\" autoplay></audio>
    </div>
    <pre id=\"log\"></pre>
    <script>
      const logEl = document.getElementById('log');
      const videoEl = document.getElementById('remoteVideo');
      const audioEl = document.getElementById('remoteAudio');
      const btnUpload = document.getElementById('btnUpload');
      const btnStart = document.getElementById('btnStart');
      const btnStop = document.getElementById('btnStop');
      let pc = null;

      function log(msg){
        const t = new Date().toLocaleTimeString();
        logEl.textContent += `[${t}] ${msg}\n`;
        logEl.scrollTop = logEl.scrollHeight;
      }

      btnUpload.onclick = async () => {
        const audioFile = document.getElementById('fileAudio').files[0];
        const imageFile = document.getElementById('fileImage').files[0];
        if(!audioFile && !imageFile){
          log('No files selected');
          return;
        }
        const form = new FormData();
        if(audioFile){ form.append('audio', audioFile); }
        if(imageFile){ form.append('source', imageFile); }
        const res = await fetch('/upload', { method: 'POST', body: form });
        if(!res.ok){
          log('Upload failed');
          return;
        }
        const data = await res.json();
        if(data.audio_path){ document.getElementById('audioPath').value = data.audio_path; }
        if(data.source_path){ document.getElementById('sourcePath').value = data.source_path; }
        log('Upload complete');
      };

      async function start(){
        btnStart.disabled = true;
        btnStop.disabled = false;
        pc = new RTCPeerConnection({
          iceServers: [{urls: ['stun:stun.l.google.com:19302']}]
        });
        const remoteStream = new MediaStream();
        videoEl.srcObject = remoteStream;
        audioEl.srcObject = remoteStream;

        pc.ontrack = (event) => {
          remoteStream.addTrack(event.track);
          if (event.track.kind === 'audio') {
            audioEl.muted = false;
            audioEl.play().catch(()=>{});
          }
        };

        pc.onconnectionstatechange = () => {
          log(`connection: ${pc.connectionState}`);
          if (['failed','closed','disconnected'].includes(pc.connectionState)) {
            stop();
          }
        };

        pc.addTransceiver('video', {direction: 'recvonly'});
        pc.addTransceiver('audio', {direction: 'recvonly'});

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const payload = {
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
          audio_path: document.getElementById('audioPath').value,
          source_path: document.getElementById('sourcePath').value,
          setup_kwargs: { sampling_timesteps: Number(document.getElementById('samplingSteps').value), online_mode: true },
          run_kwargs: { frame_scale: Number(document.getElementById('frameScale').value) }
        };

        const response = await fetch('/webrtc/offer', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        if(!response.ok){
          log('Offer failed');
          stop();
          return;
        }
        const answer = await response.json();
        await pc.setRemoteDescription(answer);
        log('Streaming started');
      }

      async function stop(){
        if(pc){
          pc.getSenders().forEach(s => s.track && s.track.stop());
          pc.getReceivers().forEach(r => r.track && r.track.stop());
          pc.close();
          pc = null;
        }
        btnStart.disabled = false;
        btnStop.disabled = true;
        log('Streaming stopped');
      }

      btnStart.onclick = () => start().catch(err => { log(err); stop(); });
      btnStop.onclick = () => stop();
    </script>
  </body>
</html>
"""

    async def handle_offer(self, payload: OfferPayload) -> Dict[str, str]:
        validated = self._validate_offer(payload.model_dump())

        offer = RTCSessionDescription(sdp=validated.sdp, type=validated.type)
        pc = RTCPeerConnection(self.rtc_configuration)
        self._pcs.add(pc)
        loop = asyncio.get_running_loop()

        async def on_fail(reason: str) -> None:
            await self._cleanup_peer(pc)

        self.monitor.attach(pc, ready_event=asyncio.Event(), on_fail=on_fail)

        # Advertise recvonly transceivers for compatibility with browsers/clients.
        if hasattr(pc, "addTransceiver"):
            pc.addTransceiver("video", direction="recvonly")
            pc.addTransceiver("audio", direction="recvonly")
        elif not hasattr(pc, "transceivers"):
            pc.transceivers = [
                types.SimpleNamespace(kind="video", direction="recvonly"),
                types.SimpleNamespace(kind="audio", direction="recvonly"),
            ]

        video_track = VideoFrameTrack(loop=loop)

        sampling_steps = (
            validated.sampling_timesteps
            if validated.sampling_timesteps is not None
            else self.default_sampling_steps
        )

        chunk_tuple = validated.chunk_config or self.default_chunk_config or (3, 5, 2)

        frame_scale = validated.frame_scale or self.default_frame_scale
        chunk_sleep_s = (
            validated.chunk_sleep_s
            if validated.chunk_sleep_s is not None
            else self.default_chunk_sleep_s
        )

        # Load audio synchronously for track and pipeline
        audio_16k, sr = librosa.load(validated.audio_path, sr=16000)
        audio_48k = librosa.resample(audio_16k, orig_sr=16000, target_sr=48000)
        audio_track = AudioArrayTrack(audio_48k, sample_rate=48000)

        pc.addTrack(video_track)
        pc.addTrack(audio_track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        # Return SDP answer immediately. Waiting for "connected" here can deadlock:
        # the client cannot complete the handshake until it receives this answer.
        state = getattr(pc, "connectionState", None)
        if state in {"failed", "disconnected", "closed"}:
            await self._cleanup_peer(pc)
            raise HTTPException(status_code=502, detail="peer connection failed or ICE unreachable")

        asyncio.create_task(
            self._run_streaming_pipeline(
                pc=pc,
                video_track=video_track,
                audio_track=audio_track,
                audio_path=validated.audio_path,
                audio_samples_16k=audio_16k,
                source_path=validated.source_path,
                setup_kwargs={
                    "sampling_timesteps": sampling_steps,
                    "online_mode": payload.setup_kwargs.get("online_mode", True),
                },
                run_kwargs={
                    "chunksize": chunk_tuple,
                    "chunk_sleep_s": chunk_sleep_s,
                },
                frame_scale=frame_scale,
            )
        )

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def _run_streaming_pipeline(
        self,
        *,
        pc: RTCPeerConnection,
        video_track: VideoFrameTrack,
        audio_track: AudioArrayTrack,
        audio_path: str,
        audio_samples_16k: np.ndarray,
        source_path: str,
        setup_kwargs: Dict[str, Any],
        run_kwargs: Dict[str, Any],
        frame_scale: float,
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            if not await self._wait_for_connection(pc):
                logger.warning(
                    "Peer connection did not reach connected state; aborting stream"
                )
                return

            sdk = StreamSDK(self.cfg_pkl, self.data_root)
            writer = WebRTCFrameWriter(
                loop=loop, video_track=video_track, frame_scale=frame_scale
            )
            sdk.writer = writer

            setup_kwargs = {k: v for k, v in setup_kwargs.items() if v is not None}
            temp_output_path = (
                f"/tmp/webrtc_output_{os.getpid()}_{int(loop.time() * 1000)}.mp4"
            )

            setup_callable = partial(
                sdk.setup,
                source_path,
                temp_output_path,
                **setup_kwargs,
            )
            await loop.run_in_executor(None, setup_callable)

            audio = audio_samples_16k
            num_frames = int(len(audio) / 16000 * 25)
            sdk.setup_Nd(num_frames)

            if setup_kwargs.get("online_mode", True):
                chunksize = run_kwargs.get("chunksize", (3, 5, 2))
                audio = np.concatenate(
                    [np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0
                )
                split_len = int(sum(chunksize) * 0.04 * 16000) + 80
                idx = 0
                for chunk_start in range(0, len(audio), chunksize[1] * 640):
                    chunk = audio[chunk_start : chunk_start + split_len]
                    if len(chunk) < split_len:
                        chunk = np.pad(chunk, (0, split_len - len(chunk)))
                    await loop.run_in_executor(None, sdk.run_chunk, chunk, chunksize)
                    sleep_s = run_kwargs.get("chunk_sleep_s")
                    if sleep_s:
                        await asyncio.sleep(max(0.0, sleep_s))
                    idx += 1
            else:
                aud_feat = sdk.wav2feat.wav2feat(audio)
                sdk.audio2motion_queue.put(aud_feat)

            sdk.close()
        except Exception as exc:
            logger.exception("Streaming pipeline failed: %s", exc)
            return
        finally:
            video_track.finalize()
            await self._cleanup_peer(pc)

    async def _cleanup_peer(self, pc: RTCPeerConnection) -> None:
        if pc in self._pcs:
            self._pcs.remove(pc)
        await pc.close()

    async def _wait_for_connection(
        self, pc: RTCPeerConnection, timeout: float = 60.0
    ) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            state = getattr(pc, "connectionState", None)
            if state == "connected":
                await self._log_selected_candidate_pair(pc)
                return True
            if state in {"failed", "disconnected", "closed"}:
                return False
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("Timed out waiting for peer connection (state=%s)", state)
                return False
            await asyncio.sleep(0.5)

    async def _log_selected_candidate_pair(self, pc: RTCPeerConnection) -> None:
        try:
            stats = await pc.getStats()
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("Failed to obtain connection stats: %s", exc)
            return

        for report in stats.values():
            if getattr(report, "type", "") == "candidate-pair" and getattr(
                report, "selected", False
            ):
                local = stats.get(getattr(report, "localCandidateId", None))
                remote = stats.get(getattr(report, "remoteCandidateId", None))
                logger.info(
                    "Selected ICE pair: local=%s (%s) <-> remote=%s (%s)",
                    getattr(local, "address", "?"),
                    getattr(local, "candidateType", "?"),
                    getattr(remote, "address", "?"),
                    getattr(remote, "candidateType", "?"),
                )
                break


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ditto WebRTC streaming server")
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to TRT data_root"
    )
    parser.add_argument("--cfg_pkl", type=str, required=True, help="Path to cfg_pkl")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--online-sampling-steps", type=int, default=None)
    parser.add_argument("--online-chunk-config", type=str, default=None)
    parser.add_argument("--chunk-sleep-ms", type=float, default=10.0)
    parser.add_argument("--frame-scale", type=float, default=0.5)

    args = parser.parse_args()

    chunk_sleep_s = (
        args.chunk_sleep_ms / 1000.0 if args.chunk_sleep_ms is not None else None
    )
    chunk_tuple = (
        parse_chunk_config(args.online_chunk_config)
        if args.online_chunk_config
        else (3, 5, 2)
    )
    sampling_steps = (
        clamp_sampling_timesteps(args.online_sampling_steps)
        if args.online_sampling_steps is not None
        else None
    )

    server = StreamingServer(
        args.cfg_pkl,
        args.data_root,
        default_chunk_config=chunk_tuple,
        default_sampling_steps=sampling_steps,
        chunk_sleep_s=chunk_sleep_s,
        frame_scale=args.frame_scale,
    )

    logger.info("Starting Ditto WebRTC streaming server on %s:%d", args.host, args.port)
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
