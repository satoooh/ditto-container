# File: src/tests/test_trt_hardware_compat.py
# Purpose: Ensure hardware compatibility flag resolution works when TensorRT exposes Blackwell enums.
# Why: Guards against regressions after upgrading to TensorRT 10.9 Blackwell builds.

import sys
from types import SimpleNamespace

import pytest

import src.scripts.cvt_onnx_to_trt as converter


@pytest.fixture(autouse=True)
def clean_fake_trt():
    original = sys.modules.pop("tensorrt", None)
    try:
        yield
    finally:
        if original is not None:
            sys.modules["tensorrt"] = original
        else:
            sys.modules.pop("tensorrt", None)


def test_blackwell_enum_is_detected(monkeypatch: pytest.MonkeyPatch):
    class FakeEnum(dict):
        __members__ = {
            "HOPPER": SimpleNamespace(value=1),
            "BLACKWELL_PLUS": SimpleNamespace(value=2),
        }

    fake_trt = SimpleNamespace(HardwareCompatibilityLevel=FakeEnum())
    sys.modules["tensorrt"] = fake_trt

    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (12, 0))

    level, flag = converter._resolve_hardware_compatibility()

    assert level is FakeEnum.__members__["BLACKWELL_PLUS"]
    assert flag == "--hardware-compatibility-level=Blackwell_Plus"
