import os
import argparse
import torch


def _enum_members(enum_obj):
    if hasattr(enum_obj, "__members__") and enum_obj.__members__:
        return dict(enum_obj.__members__)

    members = {}
    for attr in dir(enum_obj):
        if not attr or not attr[0].isupper():
            continue
        try:
            members[attr] = getattr(enum_obj, attr)
        except AttributeError:
            continue
    return members


def _resolve_hardware_compatibility():
    try:
        import tensorrt as trt  # type: ignore
    except ImportError:
        return None, None

    major, _ = torch.cuda.get_device_capability()
    members = _enum_members(trt.HardwareCompatibilityLevel)
    if not members:
        return None, None

    ordered_prefixes = [
        ("BLACKWELL", 12, None),
        ("HOPPER", 9, 11),
        ("ADA", 8, 8),
        ("AMPERE", 8, 8),
    ]

    def _cli_name(enum_name: str) -> str:
        return "--hardware-compatibility-level=" + "_".join(
            part.capitalize() for part in enum_name.split("_")
        )

    for prefix, min_major, max_major in ordered_prefixes:
        if major < min_major:
            continue
        if max_major is not None and major > max_major:
            continue
        candidates = [name for name in members if prefix in name]
        if not candidates:
            continue
        candidates.sort(key=lambda name: (0 if name.endswith("PLUS") else 1, len(name)))
        chosen = candidates[0]
        return members[chosen], _cli_name(chosen)
    return None, None

    major, _ = torch.cuda.get_device_capability()
    available = {level.name: level for level in trt.HardwareCompatibilityLevel}
    ordered_prefixes = [
        ("BLACKWELL", 12, None),
        ("HOPPER", 9, 11),
        ("ADA", 8, 8),
        ("AMPERE", 8, 8),
    ]

    def _cli_name(enum_name: str) -> str:
        return "--hardware-compatibility-level=" + "_".join(
            part.capitalize() for part in enum_name.split("_")
        )

    for prefix, min_major, max_major in ordered_prefixes:
        if major < min_major:
            continue
        if max_major is not None and major > max_major:
            continue
        candidates = [name for name in available if prefix in name]
        if not candidates:
            continue
        candidates.sort(key=lambda name: (0 if name.endswith("PLUS") else 1, len(name)))
        chosen = candidates[0]
        return available[chosen], _cli_name(chosen)
    return None, None


def onnx_to_trt(onnx_file, trt_file, fp16=False, more_cmd=None):
    _, compat_flag = _resolve_hardware_compatibility()
    cmd = [
        "polygraphy",
        "convert",
        onnx_file,
        "-o",
        trt_file,
        "--fp16" if fp16 else "",
        "--version-compatible",
        "--onnx-flags",
        "NATIVE_INSTANCENORM",
        "--builder-optimization-level=5",
    ]
    if compat_flag:
        cmd.insert(5, compat_flag)
    if more_cmd:
        cmd.extend(more_cmd)
    cmd = [arg for arg in cmd if arg]
    print(" ".join(cmd))
    os.system(" ".join(cmd))


def onnx_to_trt_for_gridsample(
    onnx_file, trt_file, fp16=False, plugin_file="./libgrid_sample_3d_plugin.so"
):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    plugin_libs = [plugin_file]

    onnx_path = onnx_file
    engine_path = trt_file

    builder = trt.Builder(logger)
    for pluginlib in plugin_libs:
        builder.get_plugin_registry().load_library(pluginlib)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, logger)
    res = parser.parse_from_file(onnx_path)  # parse from file
    if not res:
        print(f"Fail parsing {onnx_path}")
        for i in range(parser.num_errors):  # Get error information
            error = parser.get_error(i)
            print(error)  # Print error information
            print(
                f"{error.code() = }\n{error.file() = }\n{error.func() = }\n{error.line() = }\n{error.local_function_stack_size() = }"
            )
            print(
                f"{error.local_function_stack() = }\n{error.node_name() = }\n{error.node_operator() = }\n{error.node() = }"
            )
        parser.clear_errors()
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    config.builder_optimization_level = 5
    # Set the flag of hardware compatibility, handled dynamically based on GPU capability
    level_enum, _ = _resolve_hardware_compatibility()
    if level_enum is not None:
        config.hardware_compatibility_level = level_enum

    try:
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
    except AttributeError:
        pass

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)
    exclude_list = [
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
    ]
    for i in range(0, network.num_layers):
        layer = network.get_layer(i)
        if str(layer.type)[10:] in exclude_list:
            continue
        if "GridSample" in layer.name:
            print(f"set {layer.name} to float32")
            layer.precision = trt.float32
    config.plugins_to_serialize = plugin_libs
    engineString = builder.build_serialized_network(network, config)
    if engineString is not None:
        with open(engine_path, "wb") as f:
            f.write(engineString)


def main(onnx_dir, trt_dir, grid_sample_plugin_file=""):
    names = [i[:-5] for i in os.listdir(onnx_dir) if i.endswith(".onnx")]
    for name in names:
        if name == "warp_network_ori":
            continue

        print("=" * 20, f"{name} start", "=" * 20)

        fp16 = (
            False
            if name in {"motion_extractor", "hubert", "wavlm"}
            or name.startswith("lmdm")
            else True
        )

        more_cmd = None
        if name == "wavlm":
            more_cmd = [
                "--trt-min-shapes audio:[1,1000]",
                "--trt-max-shapes audio:[1,320080]",
                "--trt-opt-shapes audio:[1,320080]",
            ]
        elif name == "hubert":
            more_cmd = [
                "--trt-min-shapes input_values:[1,3240]",
                "--trt-max-shapes input_values:[1,12960]",
                "--trt-opt-shapes input_values:[1,6480]",
            ]

        onnx_file = f"{onnx_dir}/{name}.onnx"
        trt_file = f"{trt_dir}/{name}_fp{16 if fp16 else 32}.engine"

        if os.path.isfile(trt_file):
            print("=" * 20, f"{name} skip", "=" * 20)
            continue

        if name == "warp_network":
            onnx_to_trt_for_gridsample(
                onnx_file, trt_file, fp16, plugin_file=grid_sample_plugin_file
            )
        else:
            onnx_to_trt(onnx_file, trt_file, fp16, more_cmd=more_cmd)

        print("=" * 20, f"{name} done", "=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, help="input onnx dir")
    parser.add_argument("--trt_dir", type=str, help="output trt dir")
    args = parser.parse_args()

    onnx_dir = args.onnx_dir
    trt_dir = args.trt_dir

    assert os.path.isdir(onnx_dir)
    os.makedirs(trt_dir, exist_ok=True)

    grid_sample_plugin_file = os.path.join(onnx_dir, "libgrid_sample_3d_plugin.so")
    main(onnx_dir, trt_dir, grid_sample_plugin_file)
