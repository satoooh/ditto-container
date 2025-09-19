# File: src/tests/test_data_root_resolution.py
# Purpose: Validate automatic TensorRT engine directory selection logic for mixed GPU deployments.
# Why: Ensures both legacy Ampere and new Blackwell engine layouts remain compatible across machines.

import os
from typing import Iterable, Optional, Tuple

import pytest

from server_paths import resolve_data_root


@pytest.fixture(autouse=True)
def configure_defaults(monkeypatch: pytest.MonkeyPatch) -> Iterable[Tuple[str, Optional[int], Optional[int]]]:
    shadow = (
        ("/first/candidate", 12, None),
        ("/second/candidate", 8, 11),
    )
    monkeypatch.setattr("server_paths.DEFAULT_DATA_ROOT_CANDIDATES", shadow)
    return shadow


@pytest.fixture(autouse=True)
def fake_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("server_paths._current_cc", lambda: (12, 0))


def test_uses_requested_path_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path):
    existing = tmp_path / "custom"
    existing.mkdir()
    monkeypatch.setattr(
        "server_paths.os.path.isdir", lambda path: os.path.abspath(path) == str(existing)
    )

    resolved = resolve_data_root(str(existing))

    assert resolved == str(existing)


def test_picks_candidate_matching_cc(monkeypatch: pytest.MonkeyPatch, configure_defaults):
    target = configure_defaults[0][0]

    def fake_isdir(path: str) -> bool:
        return os.path.abspath(path) == target

    monkeypatch.setattr("server_paths.os.path.isdir", fake_isdir)

    resolved = resolve_data_root(None)

    assert resolved == target


def test_fallback_when_no_candidate_matches(monkeypatch: pytest.MonkeyPatch, configure_defaults):
    # Force all candidates to miss by reporting different compute capability
    monkeypatch.setattr("server_paths._current_cc", lambda: (7, 0))

    available = {path: True for path, *_ in configure_defaults}

    def fake_isdir(path: str) -> bool:
        return available.get(os.path.abspath(path), False)

    monkeypatch.setattr("server_paths.os.path.isdir", fake_isdir)

    resolved = resolve_data_root(None)

    assert resolved == configure_defaults[0][0]
