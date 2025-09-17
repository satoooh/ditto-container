"""Shared helpers for Ditto streaming binary protocol."""
from __future__ import annotations

import struct
from typing import Tuple

FRAME_HEADER_STRUCT = struct.Struct('!IdI')


def build_binary_frame_payload(frame_id: int, timestamp: float, jpeg_bytes: bytes) -> bytes:
    """Pack a binary frame payload with metadata header."""
    header = FRAME_HEADER_STRUCT.pack(int(frame_id), float(timestamp), len(jpeg_bytes))
    return header + jpeg_bytes


def parse_binary_frame(payload: bytes) -> Tuple[int, float, bytes]:
    """Parse a binary WebSocket payload into metadata and JPEG bytes."""
    if len(payload) < FRAME_HEADER_STRUCT.size:
        raise ValueError("Binary payload too small for frame header")

    frame_id, timestamp, length = FRAME_HEADER_STRUCT.unpack(payload[:FRAME_HEADER_STRUCT.size])
    jpeg_bytes = payload[FRAME_HEADER_STRUCT.size:FRAME_HEADER_STRUCT.size + length]

    if len(jpeg_bytes) != length:
        raise ValueError(
            f"Binary payload truncated: expected {length} bytes, got {len(jpeg_bytes)}"
        )

    return frame_id, timestamp, jpeg_bytes
