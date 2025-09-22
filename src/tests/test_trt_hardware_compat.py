# File: src/tests/test_trt_hardware_compat.py
# File: src/tests/test_trt_hardware_compat.py
# Purpose: Ensure hardware compatibility flag resolution works when TensorRT exposes Blackwell enums.
# Why: Guards against regressions after upgrading to TensorRT 10.9 Blackwell builds.

import sys
from types import SimpleNamespace

import pytest

import src.scripts.cvt_onnx_to_trt as converter
import src.scripts.hardware_compat as hardware_compat


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

    monkeypatch.setattr(hardware_compat.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware_compat.torch.cuda, "get_device_capability", lambda: (12, 0))

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

    monkeypatch.setattr(hardware_compat.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware_compat.torch.cuda, "get_device_capability", lambda: (12, 0))

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

    monkeypatch.setattr(hardware_compat.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware_compat.torch.cuda, "get_device_capability", lambda: (12, 0))

    specs = converter._resolve_requested_hardware_levels(["auto", "same_cc", "same_cc"])

    assert [spec.key for spec in specs] == ["blackwell_plus", "same_compute_capability"]
    assert [spec.suffix for spec in specs] == ["_blackwell_plus", "_same_compute_capability"]



def test_polygraphy_command_builds_expected_flags(tmp_path):
    spec = converter.HardwareSpec(
        key="blackwell_plus",
        enum_value=None,
        cli_flag="--hardware-compatibility-level=Blackwell_Plus",
        origin="explicit",
    )
    profile_shapes = {"audio": ((1, 1000), (1, 320080), (1, 320080))}
    plugin_file = tmp_path / "libgrid_sample_3d_plugin.so"
    plugin_file.write_text("stub")

    cmd = converter._polygraphy_command(
        "model.onnx",
        "model.engine",
        True,
        spec,
        profile_shapes=profile_shapes,
        plugin_file=str(plugin_file),
    )

    assert "--fp16" in cmd
    assert "--hardware-compatibility-level=Blackwell_Plus" in cmd
    assert "--trt-min-shapes" in cmd
    assert "audio:[1,1000]" in cmd
    assert "--plugins" in cmd
    assert str(plugin_file) in cmd


def test_polygraphy_command_skips_optional_sections():
    spec = converter.HardwareSpec(
        key="auto",
        enum_value=None,
        cli_flag=None,
        origin="auto",
    )

    cmd = converter._polygraphy_command(
        "model.onnx",
        "model.engine",
        False,
        spec,
        profile_shapes=None,
        plugin_file=None,
    )

    assert "--fp16" not in cmd
    assert "--plugins" not in cmd
    assert not any(token.startswith("--trt-") for token in cmd if ":" in token)
