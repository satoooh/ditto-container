from __future__ import annotations

import asyncio
import fractions
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
    ):
        super().__init__()
        self._loop = loop
        self._queue: asyncio.Queue[Optional[av.VideoFrame]] = asyncio.Queue()
        self._time_base = time_base
        self._pts = 0

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
    ):
        super().__init__()
        self._samples = samples.astype(np.float32)
        self._sample_rate = sample_rate
        self._frame_size = int(sample_rate * frame_duration)
        self._cursor = 0
        self._time_base = fractions.Fraction(1, sample_rate)
        self._pts = 0
        self._finished = False

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

        pcm16 = np.clip(chunk, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)

        frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm16))
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += len(pcm16)
        frame.planes[0].update(pcm16.tobytes())
        return frame
