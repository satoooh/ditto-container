# File: src/scripts/hardware_compat.py
# Purpose: Centralise TensorRT hardware compatibility resolution for conversion scripts.
# Why: Keep compatibility heuristics reusable while staying within per-file line limits.

import importlib
import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - fallback for environments without torch
    class _FallbackCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def get_device_capability() -> Tuple[int, int]:
            raise RuntimeError("CUDA capability unavailable: torch not installed")

    class _FallbackTorch:
        cuda = _FallbackCuda()

    torch = _FallbackTorch()  # type: ignore


@dataclass(frozen=True)
class HardwareSpec:
    key: str
    enum_value: Optional[object]
    cli_flag: Optional[str]
    origin: str
    suffix: str = ""


_RULES: Tuple[Tuple[str, int, Optional[int], Optional[int], Optional[int], str], ...] = (
    ("BLACKWELL", 12, None, None, None, "BLACKWELL_PLUS"),
    ("HOPPER", 9, 11, None, None, "HOPPER_PLUS"),
    ("ADA", 8, 8, 9, None, "ADA"),
    ("AMPERE", 8, 8, None, 8, "AMPERE"),
)

_ALIASES: Dict[str, str] = {
    "ampere_plus": "AMPERE_PLUS",
    "ampere": "AMPERE",
    "hopper_plus": "HOPPER_PLUS",
    "blackwell_plus": "BLACKWELL_PLUS",
    "blackwell": "BLACKWELL",
    "ada": "ADA",
    "same_cc": "SAME_COMPUTE_CAPABILITY",
    "same_compute_capability": "SAME_COMPUTE_CAPABILITY",
}


def _sanitize(name: str) -> str:
    tokens = [token for token in re.split(r"[^0-9A-Za-z]+", name) if token]
    return "_".join(token.lower() for token in tokens) or "auto"


def _cli_flag(name: str) -> str:
    tokens = [token for token in name.replace("-", "_").split("_") if token]
    return "--hardware-compatibility-level=" + "_".join(token.title() for token in tokens)


def _enum_members(enum_obj) -> Dict[str, object]:
    if hasattr(enum_obj, "__members__") and enum_obj.__members__:
        return dict(enum_obj.__members__)

    members: Dict[str, object] = {}
    for attr in dir(enum_obj):
        if not attr or not attr[0].isupper():
            continue
        try:
            members[attr] = getattr(enum_obj, attr)
        except AttributeError:
            continue
    return members


def _load_members() -> Dict[str, object]:
    try:
        import tensorrt as trt  # type: ignore
    except ImportError:
        return {}

    enum_obj = getattr(trt, "HardwareCompatibilityLevel", None)
    if enum_obj is None:
        try:
            enum_obj = getattr(importlib.import_module("tensorrt.tensorrt"), "HardwareCompatibilityLevel", None)
        except (ImportError, AttributeError):
            enum_obj = None
    return _enum_members(enum_obj) if enum_obj is not None else {}


def _current_cc() -> Optional[Tuple[int, int]]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
    except Exception:
        return None
    return None


def _auto_spec(members: Dict[str, object]) -> HardwareSpec:
    cc = _current_cc()
    if not cc:
        return HardwareSpec(key="auto", enum_value=None, cli_flag=None, origin="auto")

    major, minor = cc
    for prefix, min_major, max_major, min_minor, max_minor, fallback in _RULES:
        if major < min_major or (max_major is not None and major > max_major):
            continue
        if (min_minor is not None and minor < min_minor) or (max_minor is not None and minor > max_minor):
            continue
        if members:
            candidates = [name for name in members if prefix in name]
            if candidates:
                candidates.sort(key=lambda name: (0 if name.endswith("PLUS") else 1, len(name)))
                chosen = candidates[0]
                return HardwareSpec(
                    key=_sanitize(chosen),
                    enum_value=members[chosen],
                    cli_flag=_cli_flag(chosen),
                    origin="auto",
                )
        return HardwareSpec(
            key=_sanitize(fallback),
            enum_value=None,
            cli_flag=_cli_flag(fallback),
            origin="auto",
        )
    return HardwareSpec(key="auto", enum_value=None, cli_flag=None, origin="auto")


def _explicit_spec(option: str, members: Dict[str, object]) -> HardwareSpec:
    normalized = option.replace("-", "_").lower()
    if normalized == "auto":
        return _auto_spec(members)
    if normalized == "none":
        return HardwareSpec(key="none", enum_value=None, cli_flag=None, origin="explicit")

    canonical = _ALIASES.get(normalized, normalized.upper())
    if members and canonical in members:
        return HardwareSpec(
            key=_sanitize(canonical),
            enum_value=members[canonical],
            cli_flag=_cli_flag(canonical),
            origin="explicit",
        )
    return HardwareSpec(
        key=_sanitize(canonical),
        enum_value=None,
        cli_flag=_cli_flag(canonical),
        origin="explicit",
    )


def resolve_requested_hardware_levels(requested: Sequence[str]) -> List[HardwareSpec]:
    members = _load_members()
    options = list(requested) if requested else ["auto"]

    specs: List[HardwareSpec] = []
    seen: set[str] = set()
    for option in options:
        spec = _explicit_spec(option, members)
        if spec.key in seen:
            continue
        seen.add(spec.key)
        specs.append(spec)

    if not specs:
        specs.append(_auto_spec(members))

    total = len(specs)
    return [replace(spec, suffix="" if total == 1 and spec.origin == "auto" else f"_{spec.key}") for spec in specs]


def resolve_hardware_compatibility() -> Tuple[Optional[object], Optional[str]]:
    spec = _auto_spec(_load_members())
    return spec.enum_value, spec.cli_flag
