# File: src/tests/test_data_root_resolution.py
# Purpose: Validate automatic TensorRT engine directory selection logic for mixed GPU deployments.
# Why: Ensures both legacy Ampere and new universal engine layouts remain compatible across machines.

import os
from typing import Iterable

import pytest

from server_paths import resolve_data_root


@pytest.fixture(autouse=True)
def reset_candidates(monkeypatch: pytest.MonkeyPatch) -> Iterable[str]:
    # Keep tests isolated by shadowing the default tuple with deterministic values.
    shadow = (
        "/first/candidate",
        "/second/candidate",
    )
    monkeypatch.setattr("server_paths.DEFAULT_DATA_ROOT_CANDIDATES", shadow)
    return shadow


def test_uses_requested_path_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path):
    existing = tmp_path / "custom"
    existing.mkdir()
    monkeypatch.setattr(
        "server_paths.os.path.isdir",
        lambda path: os.path.abspath(path) == str(existing),
    )

    resolved = resolve_data_root(str(existing))

    assert resolved == str(existing)


def test_falls_back_to_default_candidate(
    monkeypatch: pytest.MonkeyPatch, reset_candidates
):
    target = reset_candidates[1]

    def fake_isdir(path: str) -> bool:
        return os.path.abspath(path) == target

    monkeypatch.setattr("server_paths.os.path.isdir", fake_isdir)

    resolved = resolve_data_root(None)

    assert resolved == target


def test_raises_when_no_candidate_exists(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("server_paths.os.path.isdir", lambda path: False)

    with pytest.raises(FileNotFoundError):
        resolve_data_root(None)
