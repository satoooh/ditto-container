# File: src/tests/test_trt_hardware_compat.py
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

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (12, 0))

    level, flag = converter._resolve_hardware_compatibility()

    assert level is FakeEnum.__members__["BLACKWELL_PLUS"]
    assert flag == "--hardware-compatibility-level=Blackwell_Plus"

    specs = converter._resolve_requested_hardware_levels(["auto"])
    assert len(specs) == 1
    assert specs[0].key == "blackwell_plus"
    assert specs[0].suffix == ""


def test_same_cc_falls_back_to_cli(monkeypatch: pytest.MonkeyPatch):
    class FakeEnum(dict):
        __members__ = {
            "BLACKWELL_PLUS": SimpleNamespace(value=2),
        }

    fake_trt = SimpleNamespace(HardwareCompatibilityLevel=FakeEnum())
    sys.modules["tensorrt"] = fake_trt

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (12, 0))

    specs = converter._resolve_requested_hardware_levels(["same_cc"])

    assert len(specs) == 1
    spec = specs[0]
    assert spec.key == "same_compute_capability"
    assert spec.cli_flag == "--hardware-compatibility-level=Same_Compute_Capability"
    assert spec.enum_value is None
    assert spec.suffix == "_same_compute_capability"


def test_duplicate_targets_are_deduplicated(monkeypatch: pytest.MonkeyPatch):
    class FakeEnum(dict):
        __members__ = {
            "BLACKWELL_PLUS": SimpleNamespace(value=2),
            "SAME_COMPUTE_CAPABILITY": SimpleNamespace(value=3),
        }

    fake_trt = SimpleNamespace(HardwareCompatibilityLevel=FakeEnum())
    sys.modules["tensorrt"] = fake_trt

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (12, 0))

    specs = converter._resolve_requested_hardware_levels(["auto", "same_cc", "same_cc"])

    assert [spec.key for spec in specs] == ["blackwell_plus", "same_compute_capability"]
    assert [spec.suffix for spec in specs] == ["", "_same_compute_capability"]

