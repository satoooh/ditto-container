from __future__ import annotations

import asyncio
import fractions
import array
from typing import Optional

import av
import numpy as np
from aiortc.mediastreams import MediaStreamTrack


class VideoFrameTrack(MediaStreamTrack):
    """MediaStreamTrack implementation that sends frames pushed from the Ditto pipeline."""

    kind = "video"

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        time_base: fractions.Fraction = fractions.Fraction(1, 90000),
        metrics=None,
    ):
        super().__init__()
        self._loop = loop
        self._queue: asyncio.Queue[Optional[av.VideoFrame]] = asyncio.Queue()
        self._time_base = time_base
        self._pts = 0
        self._metrics = metrics

    async def recv(self) -> av.VideoFrame:
        frame = await self._queue.get()
        if frame is None:
            raise asyncio.CancelledError("video track ended")
        return frame

    def enqueue(self, frame_bgr: np.ndarray) -> None:
        """Convert BGR frame to VideoFrame and schedule for send."""

        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        frame.pts = self._pts
        frame.time_base = self._time_base
        if self._metrics:
            self._metrics.record_video(self._pts, float(self._time_base))
        self._pts += 1
        asyncio.run_coroutine_threadsafe(self._queue.put(frame), self._loop)

    def finalize(self) -> None:
        asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)


class AudioArrayTrack(MediaStreamTrack):
    """MediaStreamTrack that streams a mono audio array (float32 -1..1)."""

    kind = "audio"

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        frame_duration: float = 0.02,
        metrics=None,
    ):
        super().__init__()
        if hasattr(samples, "astype"):
            self._samples = samples.astype(getattr(np, "float32", float))
        elif hasattr(np, "array"):
            self._samples = np.array(samples, dtype=getattr(np, "float32", float))
        else:
            self._samples = [float(x) for x in samples]
        self._sample_rate = sample_rate
        self._frame_size = int(sample_rate * frame_duration)
        self._cursor = 0
        self._time_base = fractions.Fraction(1, sample_rate)
        self._pts = 0
        self._finished = False
        self._metrics = metrics

    async def recv(self) -> av.AudioFrame:
        if self._cursor >= len(self._samples):
            if self._finished:
                raise asyncio.CancelledError("audio track finished")
            self._finished = True
            await asyncio.sleep(self._frame_size / self._sample_rate)
            frame = av.AudioFrame(format="s16", layout="mono", samples=self._frame_size)
            frame.sample_rate = self._sample_rate
            frame.pts = self._pts
            frame.time_base = self._time_base
            frame.planes[0].update(b"\x00" * (self._frame_size * 2))
            return frame

        chunk = self._samples[self._cursor : self._cursor + self._frame_size]
        self._cursor += self._frame_size
        await asyncio.sleep(len(chunk) / self._sample_rate)

        if hasattr(np, "clip"):
            pcm16 = np.clip(chunk, -1.0, 1.0)
        else:
            pcm16 = [max(-1.0, min(1.0, float(x))) for x in chunk]

        if hasattr(pcm16, "astype"):
            pcm16 = (pcm16 * 32767).astype(getattr(np, "int16", int))
        else:
            pcm16 = [int(x * 32767) for x in pcm16]

        frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm16))
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += len(pcm16)
        if self._metrics:
            self._metrics.record_audio(frame.pts, float(frame.time_base))
        if hasattr(pcm16, "tobytes"):
            payload = pcm16.tobytes()
        else:
            payload = array.array("h", pcm16).tobytes()
        frame.planes[0].update(payload)
        return frame
