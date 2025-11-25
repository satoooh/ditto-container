# Where: src/tests/test_video_frame_track_pts.py
# What: Ensure VideoFrameTrack enqueues frames with monotonic PTS (task 4.1).

import asyncio
import sys
import types

# Stub av.VideoFrame to avoid heavy dependency
class _StubVideoFrame:
    def __init__(self):
        self.pts = None
        self.time_base = None

    @staticmethod
    def from_ndarray(arr, format="bgr24"):
        return _StubVideoFrame()


sys.modules["av"] = types.SimpleNamespace(VideoFrame=_StubVideoFrame)
sys.modules["numpy"] = types.SimpleNamespace(ndarray=object)
sys.modules["aiortc.mediastreams"] = types.SimpleNamespace(MediaStreamTrack=type("MST", (), {}))

# Ensure real module reload (not previously stubbed)
sys.modules.pop("webrtc.tracks", None)
from webrtc.tracks import VideoFrameTrack


def test_video_frame_pts_monotonic():
    loop = asyncio.new_event_loop()
    track = VideoFrameTrack(loop=loop)

    track.enqueue([[0]],)
    track.enqueue([[1]],)
    track.enqueue([[2]],)

    f1 = loop.run_until_complete(track._queue.get())
    f2 = loop.run_until_complete(track._queue.get())
    f3 = loop.run_until_complete(track._queue.get())

    assert f1.pts == 0
    assert f2.pts == 1
    assert f3.pts == 2

    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
