#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

import os
import json
import argparse
import importlib
import pkgutil
from typing import Dict, List, Tuple, Optional
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
    # Parse command line arguments and call tpacker function
    args, parameter_comments = parse_arguments()  # Get just the args object
    tpacker(args, parameter_comments)

def tpacker(args, parameter_comments):
    BANNER = "=" * 83
    print(BANNER)
    clean_invalid_files()
    try:
        # Parse parameters
        print(f"{Colors.BLUE}1. Parse input information{Colors.RESET}")
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

        # Save the model
        print(f"{Colors.BLUE}\n9. Save Resource File{Colors.RESET}")
        with open(args.output_path, "wb") as f:
            f.write(packed_model.to_bytes())

        # Export configuration file
        if args.export_config:
            export_configuration(args, model_config, device_config, memory_config, parameter_comments, args.export_config)
            print(f"{Colors.BLUE}\n10. Config File Export Success{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}\n10. Config File Export Skipped{Colors.RESET}")

        print(BANNER)

    except Exception as e:
        error_info = f"{Colors.RED}Error occurred: {str(e)}{Colors.RESET}\n"
        error_info += f"Traceback:\n{traceback.format_exc()}"
        print(error_info)
        print(BANNER)
        exit(1)

if __name__ == "__main__":
    main()

__all__ = ['tpacker']