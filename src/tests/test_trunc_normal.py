"""Checks for the truncated normal initializer utilities."""

import pytest

pytest.importorskip("torch")

from core.models.modules.util import trunc_normal_
import torch


def test_trunc_normal_values_within_bounds() -> None:
    tensor = torch.empty(256)
    trunc_normal_(tensor, mean=0.0, std=1.0, a=-0.5, b=0.5)

    assert torch.all(tensor >= -0.5)
    assert torch.all(tensor <= 0.5)
