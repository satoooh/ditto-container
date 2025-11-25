# Where: src/tests/test_parameter_normalizer.py
# What: TDD coverage for ParameterNormalizer (task 1.2).
# Why: Ensure frame/audio params are clamped and defaults applied with warnings.

import pytest

from webrtc.parameters import ParameterNormalizer


def test_normalizer_applies_defaults_when_missing():
    norm = ParameterNormalizer()
    result = norm.normalize({})

    assert result.frame_scale == 1.0
    assert result.sampling_timesteps == 30
    assert result.chunk_config == (3, 5, 2)
    assert result.chunk_sleep_s is None
    assert result.warnings == []


def test_normalizer_clamps_and_warns_on_invalid_values():
    norm = ParameterNormalizer()
    result = norm.normalize(
        {
            "frame_scale": "abc",
            "sampling_timesteps": "nan",
            "chunk_config": "bad",
            "chunk_sleep_ms": "oops",
        }
    )

    assert result.frame_scale == 1.0
    assert result.sampling_timesteps == 30
    assert result.chunk_config == (3, 5, 2)
    assert result.chunk_sleep_s is None
    assert len(result.warnings) >= 3


def test_normalizer_clamps_bounds_and_converts_ms():
    norm = ParameterNormalizer()
    result = norm.normalize(
        {
            "frame_scale": 2.0,
            "sampling_timesteps": 2,
            "chunk_config": "0,4,3",
            "chunk_sleep_ms": 50,
        }
    )

    assert result.frame_scale == 1.0  # upper clamp
    assert result.sampling_timesteps == 5  # lower clamp
    assert result.chunk_config == (1, 4, 3)  # zeros become 1
    assert result.chunk_sleep_s == 0.05
    assert result.warnings == []


def test_normalizer_accepts_seconds_directly():
    norm = ParameterNormalizer()
    result = norm.normalize(
        {
            "frame_scale": 0.25,
            "sampling_timesteps": 80,
            "chunk_config": [2, 3, 2],
            "chunk_sleep_s": 0.2,
        }
    )

    assert result.frame_scale == 0.25
    assert result.sampling_timesteps == 80
    assert result.chunk_config == (2, 3, 2)
    assert result.chunk_sleep_s == 0.2
    assert result.warnings == []

