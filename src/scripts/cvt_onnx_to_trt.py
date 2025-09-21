# File: src/scripts/cvt_onnx_to_trt.py
# Purpose: Convert Ditto ONNX graphs into TensorRT engines with hardware-aware settings.
# Why: Build engines for multiple GPU generations while keeping GridSample fallbacks contained.

import argparse
import os
import subprocess
from typing import Iterable, List, Optional

from hardware_compat import (
    HardwareSpec,
    resolve_hardware_compatibility as _resolve_hardware_compatibility,
    resolve_requested_hardware_levels as _resolve_requested_hardware_levels,
)

resolve_hardware_compatibility = _resolve_hardware_compatibility
resolve_requested_hardware_levels = _resolve_requested_hardware_levels

_FP32_MODELS = {"motion_extractor", "hubert", "wavlm"}
_DYNAMIC_SHAPES = {
    "wavlm": [
        "--trt-min-shapes",
        "audio:[1,1000]",
        "--trt-max-shapes",
        "audio:[1,320080]",
        "--trt-opt-shapes",
        "audio:[1,320080]",
    ],
    "hubert": [
        "--trt-min-shapes",
        "input_values:[1,3240]",
        "--trt-max-shapes",
        "input_values:[1,12960]",
        "--trt-opt-shapes",
        "input_values:[1,6480]",
    ],
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


def onnx_to_trt(
    onnx_file: str,
    trt_file: str,
    fp16: bool,
    spec: HardwareSpec,
    more_cmd: Optional[Iterable[str]] = None,
) -> None:
    cmd: List[str] = ["polygraphy", "convert", onnx_file, "-o", trt_file]
    if fp16:
        cmd.append("--fp16")
    if spec.cli_flag:
        cmd.append(spec.cli_flag)
    cmd.extend(
        [
            "--version-compatible",
            "--onnx-flags",
            "NATIVE_INSTANCENORM",
            "--builder-optimization-level=5",
        ]
    )
    if more_cmd:
        cmd.extend(more_cmd)

    printable = " ".join(cmd)
    print(printable)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"polygraphy convert failed (exit code {result.returncode}) for {os.path.basename(onnx_file)}"
        )


def onnx_to_trt_for_gridsample(
    onnx_file: str,
    trt_file: str,
    fp16: bool,
    spec: HardwareSpec,
    plugin_file: str,
) -> None:
    if not os.path.isfile(plugin_file):
        raise FileNotFoundError(f"GridSample plugin not found: {plugin_file}")

    import tensorrt as trt  # local import to defer heavy dependency

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
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
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    try:
        config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)
    except AttributeError:
        pass

    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if str(layer.type)[10:] in _GRID_SAMPLE_EXCLUDE:
            continue
        if "GridSample" in layer.name:
            print(f"set {layer.name} to float32")
            layer.precision = trt.float32

    engine_blob = builder.build_serialized_network(network, config)
    if engine_blob is None:
        raise RuntimeError(
            f"TensorRT failed to build engine for {os.path.basename(onnx_file)} using plugin fallback"
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
            more_cmd = _DYNAMIC_SHAPES.get(name)
            onnx_file = os.path.join(onnx_dir, f"{name}.onnx")
            trt_file = os.path.join(
                trt_dir, f"{name}{spec.suffix}_fp{16 if fp16 else 32}.engine"
            )

            if not force and os.path.isfile(trt_file):
                print(f"[skip] {os.path.basename(trt_file)} already present")
                continue

            print(f"[build] {name} -> {os.path.basename(trt_file)} ({spec.key})")
            try:
                onnx_to_trt(onnx_file, trt_file, fp16, spec, more_cmd=more_cmd)
            except RuntimeError as exc:
                if name == _PLUGIN_MODEL and plugin_available:
                    print("[fallback] retrying with GridSample plugin")
                    onnx_to_trt_for_gridsample(
                        onnx_file,
                        trt_file,
                        fp16,
                        spec,
                        plugin_path or "",
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
