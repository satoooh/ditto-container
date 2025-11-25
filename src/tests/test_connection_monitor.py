# Where: src/tests/test_connection_monitor.py
# What: Unit tests for ConnectionMonitor (task 3.1).
# Why: Verify failure callbacks and ready event behavior without aiortc heavy deps.

import asyncio

import pytest

from webrtc.monitor import ConnectionMonitor


class _StubPC:
    def __init__(self):
        self.handlers = {}
        self.connectionState = "new"
        self.iceConnectionState = "new"

    def on(self, event):
        def decorator(func):
            self.handlers[event] = func
            return func

        return decorator


def test_monitor_sets_ready_on_connected():
    pc = _StubPC()
    monitor = ConnectionMonitor()
    ready = asyncio.Event()
    called = []

    async def on_fail(reason):
        called.append(reason)

    monitor.attach(pc, ready_event=ready, on_fail=on_fail)

    pc.connectionState = "connected"
    asyncio.run(pc.handlers["connectionstatechange"]())

    assert ready.is_set()
    assert called == []


def test_monitor_triggers_on_fail_once():
    pc = _StubPC()
    monitor = ConnectionMonitor()
    ready = asyncio.Event()
    called = []

    async def on_fail(reason):
        called.append(reason)

    monitor.attach(pc, ready_event=ready, on_fail=on_fail)

    pc.connectionState = "failed"
    asyncio.run(pc.handlers["connectionstatechange"]())
    pc.connectionState = "closed"
    asyncio.run(pc.handlers["connectionstatechange"]())

    assert ready.is_set()
    assert called == ["failed"]
