import importlib
import math
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_streaming_protocol = importlib.import_module("streaming_protocol")
FRAME_HEADER_STRUCT = _streaming_protocol.FRAME_HEADER_STRUCT
build_binary_frame_payload = _streaming_protocol.build_binary_frame_payload
parse_binary_frame = _streaming_protocol.parse_binary_frame


def test_build_binary_frame_payload_roundtrip() -> None:
    frame_id = 42
    timestamp = 123.456
    jpeg_bytes = b"\x00\x01\xff\xd8\xff"

    payload = build_binary_frame_payload(frame_id, timestamp, jpeg_bytes)
    header_size = FRAME_HEADER_STRUCT.size

    assert len(payload) == header_size + len(jpeg_bytes)

    parsed_id, parsed_ts, parsed_len = FRAME_HEADER_STRUCT.unpack(payload[:header_size])

    assert parsed_id == frame_id
    assert math.isclose(parsed_ts, timestamp, rel_tol=1e-9)
    assert parsed_len == len(jpeg_bytes)
    assert payload[header_size:] == jpeg_bytes


def test_parse_binary_frame_roundtrip() -> None:
    frame_id = 7
    timestamp = 1.234
    jpeg_bytes = b"ABCDEF"

    payload = build_binary_frame_payload(frame_id, timestamp, jpeg_bytes)
    parsed_id, parsed_ts, parsed_bytes = parse_binary_frame(payload)

    assert parsed_id == frame_id
    assert math.isclose(parsed_ts, timestamp, rel_tol=1e-9)
    assert parsed_bytes == jpeg_bytes


def test_parse_binary_frame_truncated() -> None:
    payload = build_partial_payload()
    with pytest.raises(ValueError):
        parse_binary_frame(payload)


def build_partial_payload() -> bytes:
    header = FRAME_HEADER_STRUCT.pack(1, 1.0, 4)
    return header + b"12"  # shorter than declared length
