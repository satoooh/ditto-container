# File: src/scripts/cvt_onnx_to_trt.py
# Purpose: Convert Ditto ONNX graphs into TensorRT engines with hardware-aware settings.
# Why: Build engines for multiple GPU generations while keeping GridSample fallbacks contained.

import argparse
import os
from typing import Dict, List, Optional, Tuple

try:
    from .hardware_compat import (
        HardwareSpec,
        resolve_hardware_compatibility as _resolve_hardware_compatibility,
        resolve_requested_hardware_levels as _resolve_requested_hardware_levels,
    )
except ImportError:  # pragma: no cover - fallback when executed as a script
    from hardware_compat import (
        HardwareSpec,
        resolve_hardware_compatibility as _resolve_hardware_compatibility,
        resolve_requested_hardware_levels as _resolve_requested_hardware_levels,
    )

resolve_hardware_compatibility = _resolve_hardware_compatibility
resolve_requested_hardware_levels = _resolve_requested_hardware_levels


def _ensure_tensorrt_symbols() -> None:
    try:
        import tensorrt as trt  # type: ignore
    except ImportError:
        return

    if hasattr(trt, "OnnxParserFlag") and hasattr(trt, "BuilderFlag"):
        return

    try:
        bindings = __import__("tensorrt.tensorrt", fromlist=["__name__"])
    except Exception:
        return

    for attr in ("OnnxParserFlag", "BuilderFlag", "PreviewFeature", "Logger"):
        if hasattr(bindings, attr) and not hasattr(trt, attr):
            setattr(trt, attr, getattr(bindings, attr))


_ensure_tensorrt_symbols()

_FP32_MODELS = {"motion_extractor", "hubert", "wavlm"}
_DYNAMIC_PROFILES: Dict[str, Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = {
    "wavlm": {
        "audio": ((1, 1000), (1, 320080), (1, 320080)),
    },
    "hubert": {
        "input_values": ((1, 3240), (1, 6480), (1, 12960)),
    },
}
_GRID_SAMPLE_EXCLUDE = {
    "SHAPE",
    "ASSERTION",
    "SHUFFLE",
    "IDENTITY",
    "CONSTANT",
    "CONCATENATION",
    "GATHER",
    "SLICE",
    "CONDITION",
    "CONDITIONAL_INPUT",
    "CONDITIONAL_OUTPUT",
    "FILL",
    "NON_ZERO",
    "ONE_HOT",
}
_PLUGIN_MODEL = "warp_network"
_SKIP_SUFFIX = "warp_network_ori"


def _build_engine(
    onnx_file: str,
    trt_file: str,
    fp16: bool,
    spec: HardwareSpec,
    profile_shapes: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None,
    plugin_file: Optional[str] = None,
) -> None:
    import tensorrt as trt  # local import to defer heavy dependency

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    if plugin_file:
        if not os.path.isfile(plugin_file):
            raise FileNotFoundError(f"GridSample plugin not found: {plugin_file}")
        builder.get_plugin_registry().load_library(plugin_file)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        parser.clear_errors()
        raise RuntimeError(
            f"TensorRT parser failed for {os.path.basename(onnx_file)}: {' | '.join(errors)}"
        )

    config = builder.create_builder_config()
    config.builder_optimization_level = 5
    if spec.enum_value is not None:
        config.hardware_compatibility_level = spec.enum_value
    elif spec.cli_flag:
        print(f"[warn] Python bindings cannot apply {spec.cli_flag}; using default level")

    try:
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
    except AttributeError:
        pass

    if fp16:
        try:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        except AttributeError:
            pass

    try:
        config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)
    except AttributeError:
        pass

    if plugin_file:
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if str(layer.type)[10:] in _GRID_SAMPLE_EXCLUDE:
                continue
            if "GridSample" in layer.name:
                print(f"set {layer.name} to float32")
                layer.precision = trt.float32

    if profile_shapes:
        profile = builder.create_optimization_profile()
        for tensor_name, (min_shape, opt_shape, max_shape) in profile_shapes.items():
            profile.set_shape(
                tensor_name,
                tuple(min_shape),
                tuple(opt_shape),
                tuple(max_shape),
            )
        config.add_optimization_profile(profile)
    else:
        dynamic_inputs = [
            network.get_input(i)
            for i in range(network.num_inputs)
            if not network.get_input(i).is_shape_tensor
            and any(dim < 0 for dim in network.get_input(i).shape)
        ]
        if dynamic_inputs:
            names = ", ".join(inp.name for inp in dynamic_inputs)
            raise RuntimeError(
                f"Dynamic inputs require profile definitions but none provided: {names}"
            )

    engine_blob = builder.build_serialized_network(network, config)
    if engine_blob is None:
        raise RuntimeError(
            f"TensorRT failed to build engine for {os.path.basename(onnx_file)}"
        )

    with open(trt_file, "wb") as handle:
        handle.write(engine_blob)


def main(
    onnx_dir: str,
    trt_dir: str,
    plugin_path: Optional[str],
    compat_options: Optional[List[str]],
    force: bool,
) -> None:
    names = sorted(i[:-5] for i in os.listdir(onnx_dir) if i.endswith(".onnx"))
    specs = _resolve_requested_hardware_levels(compat_options or [])
    plugin_available = plugin_path and os.path.isfile(plugin_path)

    for spec in specs:
        print("=" * 10, f"hardware target: {spec.key}", "=" * 10)
        for name in names:
            if name == _SKIP_SUFFIX:
                continue

            fp16 = name not in _FP32_MODELS and not name.startswith("lmdm")
            profile_shapes = _DYNAMIC_PROFILES.get(name)
            onnx_file = os.path.join(onnx_dir, f"{name}.onnx")
            trt_file = os.path.join(
                trt_dir, f"{name}{spec.suffix}_fp{16 if fp16 else 32}.engine"
            )

            if not force and os.path.isfile(trt_file):
                print(f"[skip] {os.path.basename(trt_file)} already present")
                continue

            print(f"[build] {name} -> {os.path.basename(trt_file)} ({spec.key})")
            try:
                _build_engine(
                    onnx_file,
                    trt_file,
                    fp16,
                    spec,
                    profile_shapes=profile_shapes,
                )
            except RuntimeError as exc:
                if name == _PLUGIN_MODEL and plugin_available:
                    print("[fallback] retrying with GridSample plugin")
                    _build_engine(
                        onnx_file,
                        trt_file,
                        fp16,
                        spec,
                        profile_shapes=profile_shapes,
                        plugin_file=plugin_path,
                    )
                else:
                    raise RuntimeError(f"{name}: {exc}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, required=True, help="input onnx dir")
    parser.add_argument("--trt_dir", type=str, required=True, help="output trt dir")
    parser.add_argument(
        "--hardware-compatibility",
        action="append",
        choices=(
            "auto",
            "none",
            "ampere_plus",
            "blackwell_plus",
            "hopper_plus",
            "ada",
            "same_cc",
            "same_compute_capability",
        ),
        help="Repeat to emit additional hardware compatibility targets (default: auto)",
    )
    parser.add_argument(
        "--grid-sample-plugin",
        type=str,
        default=None,
        help="Path to libgrid_sample_3d_plugin.so for fallback builds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild engines even if output files already exist",
    )

    args = parser.parse_args()
    if not os.path.isdir(args.onnx_dir):
        raise FileNotFoundError(f"ONNX directory not found: {args.onnx_dir}")
    os.makedirs(args.trt_dir, exist_ok=True)

    plugin_path = args.grid_sample_plugin or os.path.join(
        args.onnx_dir, "libgrid_sample_3d_plugin.so"
    )

    main(
        args.onnx_dir,
        args.trt_dir,
        plugin_path,
        args.hardware_compatibility,
        args.force,
    )
