"""Connection monitoring helpers for WebRTC peers."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class ConnectionMonitor:
    """Watch RTCPeerConnection state changes and trigger cleanup on failure."""

    def attach(
        self,
        pc,
        *,
        ready_event: asyncio.Event,
        on_fail: Callable[[str], Awaitable[None]],
    ) -> None:
        """Attach callbacks to a peer connection.

        ready_event: set when connection reaches connected/failed/disconnected/closed.
        on_fail: called once on failure-like states with the terminal state name.
        """

        has_failed = False

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            nonlocal has_failed
            state = getattr(pc, "connectionState", None)
            logger.info("Peer connection state: %s", state)
            # Signal readiness only on terminal or connected states to avoid false early wake-ups.
            if state in {"connected", "failed", "disconnected", "closed"} and not ready_event.is_set():
                ready_event.set()
            if state in {"failed", "disconnected", "closed"} and not has_failed:
                has_failed = True
                await on_fail(state or "unknown")

        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change() -> None:
            state = getattr(pc, "iceConnectionState", None)
            logger.info("ICE connection state: %s", state)
            if state in {"failed", "disconnected", "closed"} and not ready_event.is_set():
                ready_event.set()
