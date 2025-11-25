# Where: src/tests/test_offer_validator.py
# What: TDD coverage for OfferValidator payload checks and normalization.
# Why: Ensure /webrtc/offer inputs are validated before pipeline startup.

import tempfile
from pathlib import Path

import pytest


def _make_temp_file(suffix: str) -> Path:
    fd, path_str = tempfile.mkstemp(suffix=suffix)
    Path(path_str).write_bytes(b"test")
    Path(path_str).chmod(0o644)
    return Path(path_str)


def test_offer_validator_happy_path():
    from webrtc.validators import OfferValidator

    audio = _make_temp_file(".wav")
    image = _make_temp_file(".png")
    validator = OfferValidator()

    payload = {
        "sdp": "v=0",
        "type": "offer",
        "audio_path": str(audio),
        "source_path": str(image),
        "setup_kwargs": {"sampling_timesteps": 12},
        "run_kwargs": {
            "frame_scale": 0.5,
            "chunksize": [3, 5, 2],
            "chunk_sleep_ms": 10,
        },
    }

    result = validator.validate(payload)

    assert result.sdp == "v=0"
    assert result.type == "offer"
    assert result.audio_path == str(audio)
    assert result.source_path == str(image)
    assert result.sampling_timesteps == 12
    assert result.frame_scale == 0.5
    assert result.chunk_config == (3, 5, 2)
    assert result.chunk_sleep_s == 0.01
    assert result.warnings == []


def test_offer_validator_rejects_invalid_extension():
    from webrtc.validators import OfferValidator

    audio = _make_temp_file(".txt")
    image = _make_temp_file(".png")
    validator = OfferValidator()

    payload = {
        "sdp": "v=0",
        "type": "offer",
        "audio_path": str(audio),
        "source_path": str(image),
    }

    with pytest.raises(ValueError):
        validator.validate(payload)


def test_offer_validator_falls_back_with_warning():
    from webrtc.validators import OfferValidator

    audio = _make_temp_file(".wav")
    image = _make_temp_file(".png")
    validator = OfferValidator()

    payload = {
        "sdp": "v=0",
        "type": "offer",
        "audio_path": str(audio),
        "source_path": str(image),
        "setup_kwargs": {"sampling_timesteps": "not-a-number"},
        "run_kwargs": {"frame_scale": "abc", "chunksize": "bad"},
    }

    result = validator.validate(payload)

    assert result.sampling_timesteps == 30  # default clamp
    assert result.frame_scale == 1.0  # default clamp
    assert result.chunk_config == (3, 5, 2)  # fallback
    assert result.warnings  # warnings should be present


def test_offer_validator_missing_required_field():
    from webrtc.validators import OfferValidator

    image = _make_temp_file(".png")
    validator = OfferValidator()

    payload = {
        "sdp": "v=0",
        "type": "offer",
        "source_path": str(image),
    }

    with pytest.raises(ValueError):
        validator.validate(payload)

