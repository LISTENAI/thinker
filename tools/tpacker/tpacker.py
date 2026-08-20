#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

import traceback

from .argument_parser import parse_arguments, parse_parameters, export_configuration
from .load_device import load_device_info
from .resource_packer import serialize_model
from .load_model import load_and_convert_onnx_model
from .graph_optimizer import optimize_graph
from .enum_defines import MemType, Colors, MemoryConfig
from .graph_analysis.combine import adapt_graph_to_hardware
from .generate_report import generate_memory_report, clean_invalid_files
from .flops_report import statistical_calculation_amount

def main():
    try:
        args, parameter_comments = parse_arguments()
        tpacker(args, parameter_comments)
    except Exception as e:
        error_info = f"{Colors.RED}Error occurred: {str(e)}{Colors.RESET}\n"
        error_info += f"Traceback:\n{traceback.format_exc()}"
        print(error_info)
        raise SystemExit(1)


def _format_dynamic_shape(dynamic_shape):
    if isinstance(dynamic_shape, str):
        return dynamic_shape
    return ",".join(
        f"{name}={':'.join(str(value) for value in shape)}"
        for name, shape in dynamic_shape.items()
    )


def _format_csv(value):
    if isinstance(value, str):
        return value
    return ",".join(value)


def _format_memory(memory):
    if isinstance(memory, str):
        return memory
    return ",".join(f"{name}:{location}" for name, location in memory.items())


def pack_model(
    graph_path=None,
    output_path=None,
    *,
    dump=None,
    inputs=None,
    outputs=None,
    dynamic_shape=None,
    strategy=None,
    isstream=None,
    platform=None,
    ramsize=None,
    psramsize=None,
    dma_prefetch=None,
    memory=None,
    threshold1=None,
    threshold2=None,
    threshold3=None,
    threshold4=None,
    config_file=None,
    export_config=None,
) -> bytes:
    """Pack an ONNX model and return the serialized resource bytes.

    Explicit Python arguments override values from ``config_file``. Parameters
    left as ``None`` use the configuration file value or the CLI default.
    """
    argv = []
    values = {
        "--graph_path": graph_path,
        "--output_path": output_path,
        "--dump": dump,
        "--inputs": _format_csv(inputs) if inputs is not None else None,
        "--outputs": _format_csv(outputs) if outputs is not None else None,
        "--dynamic_shape": (
            _format_dynamic_shape(dynamic_shape)
            if dynamic_shape is not None else None
        ),
        "--strategy": _format_csv(strategy) if strategy is not None else None,
        "--isstream": isstream,
        "--platform": platform,
        "--ramsize": ramsize,
        "--psramsize": psramsize,
        "--dma_prefetch": dma_prefetch,
        "--memory": _format_memory(memory) if memory is not None else None,
        "--threshold1": threshold1,
        "--threshold2": threshold2,
        "--threshold3": threshold3,
        "--threshold4": threshold4,
        "--config_file": config_file,
        "--export_config": export_config,
    }
    for option, value in values.items():
        if value is not None:
            argv.extend((option, str(value)))

    try:
        args, parameter_comments = parse_arguments(argv)
    except SystemExit as error:
        raise ValueError("Invalid tpacker arguments") from error
    return tpacker(args, parameter_comments)

def tpacker(args, parameter_comments) -> bytes:
    BANNER = "=" * 83
    clean_invalid_files()
    try:
        # Parse parameters
        model_config, device_config, memory_config = parse_parameters(args)

        # Load model and convert ONNX graph to internal IR
        print(f"{Colors.BLUE}\n2. Load model and convert to custom IR{Colors.RESET}")
        graph = load_and_convert_onnx_model(args.graph_path, model_config, args.dump)

        # Optimize the graph
        print(f"{Colors.BLUE}\n3. Graph Optimization{Colors.RESET}")
        graph = optimize_graph(graph, model_config, args.dump)

        # Load target platform information
        print(f"{Colors.BLUE}\n4. Retrieve hardware platform information{Colors.RESET}")
        device = load_device_info(graph.platform, device_config)

        # Export the effective device capacities before model packing can fail.
        if args.export_config:
            device_config.ramsize = device.sram_size
            device_config.psramsize = device.psram_size
            if not export_configuration(
                args,
                model_config,
                device_config,
                memory_config,
                parameter_comments,
                args.export_config,
            ):
                raise RuntimeError(
                    f"Failed to export configuration file: {args.export_config}"
                )
            print(f"{Colors.BLUE}4.3 Config File Export Success{Colors.RESET}")

        # Hardware-aware graph adaptation
        print(f"{Colors.BLUE}\n5. Hardware-Aware Computational Graph{Colors.RESET}")
        model, memory_plan = adapt_graph_to_hardware(graph, device, memory_config, args.dump)

        # Generate memory analysis report
        if args.dump:
            print(f"{Colors.BLUE}\n6. Generate Memory Analysis Report{Colors.RESET}")
            generate_memory_report(args.graph_path, memory_plan)

        # Statistical calculation amount
        print(f"{Colors.BLUE}\n7. Operation Count Analysis{Colors.RESET}")
        statistical_calculation_amount(graph)

        # Serialize the model
        print(f"{Colors.BLUE}\n8. Serialization of Computation Graph{Colors.RESET}")
        packed_model = serialize_model(model, memory_plan, device)
        packed_bytes = packed_model.to_bytes()

        # Save the model
        print(f"{Colors.BLUE}\n9. Save Resource File{Colors.RESET}")
        with open(args.output_path, "wb") as f:
            f.write(packed_bytes)

        return packed_bytes
    finally:
        print(BANNER)

if __name__ == "__main__":
    main()

__all__ = ['pack_model', 'tpacker']
