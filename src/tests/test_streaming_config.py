from streaming_config import (
    clamp_quality,
    clamp_sampling_timesteps,
    clamp_scale,
    parse_chunk_config,
    to_chunk_list,
)


def test_parse_chunk_config_from_string():
    assert parse_chunk_config("2,4,3") == (2, 4, 3)


def test_parse_chunk_config_from_iterable():
    assert parse_chunk_config([1, 2, 3]) == (1, 2, 3)


def test_parse_chunk_config_invalid_length():
    assert parse_chunk_config("5,6", fallback=(3, 5, 2)) == (3, 5, 2)


def test_clamp_sampling_timesteps():
    assert clamp_sampling_timesteps(200, default=30, maximum=120) == 120
    assert clamp_sampling_timesteps(2, default=30, minimum=5) == 5
    assert clamp_sampling_timesteps(None, default=30) == 30


def test_to_chunk_list_roundtrip():
    chunks = (3, 5, 2)
    assert to_chunk_list(chunks) == [3, 5, 2]


def test_clamp_quality():
    assert clamp_quality(150) == 100
    assert clamp_quality(5) == 10
    assert clamp_quality(None, default=80) == 80


def test_clamp_scale():
    assert clamp_scale(0.05) == 0.1
    assert clamp_scale(1.5) == 1.0
    assert clamp_scale(None, default=0.5) == 0.5
