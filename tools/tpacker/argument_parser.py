#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2025.09.31

import os
import json
import argparse
from typing import Dict, Optional, Tuple, List
from .enum_defines import MemType, Colors, MemoryConfig, ModelConfig, DeviceConfig

def parse_boolean(value: str) -> bool:
    """Parse a string to a boolean value."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_platform_list() -> List[str]:
    """获取所有支持的平台列表."""
    return ['venus', 'mars', 'arcs', 'venusa', 'VENUS', 'MARS', 'ARCS', 'VENUSA']

def parse_shape_config(s: str) -> Dict[str, Tuple[int]]:
    """Parse dynamic shapes configuration string into a dictionary."""
    dynamic_shape = {}
    if not s:
        return dynamic_shape
    for item in s.split(","):
        key, shape_str = item.split("=")
        shape = tuple(int(x) for x in shape_str.split(":"))
        if len(shape) == 1:
            shape = (1, shape[0], 1)
        elif len(shape) == 2:
            shape = (shape[0], shape[1], 1)
        dynamic_shape[key] = shape
    return dynamic_shape

def split_csv_string(s: str) -> List[str]:
    """Split a comma-separated string into a list of strings."""
    return s.split(",") if s else []

def _parse_memory(s:str) -> Dict[str, Tuple[int]]:
    dynamic_memory = dict()
    if s == "":
        return dynamic_memory
    s1 = s.split(",")
    for i in range(len(s1)):
      s2 = s1[i].split(":")
      if len(s2)==1:
        assert s1[0].lower() in {'psram', 'share-mem', 'flash'}, f"{s1[i]}"
      assert s2[1].lower() in {'psram', 'share-mem', 'flash'}
      dynamic_memory[s2[0]] = s2[1]
    return dynamic_memory
    
def read_config_file(args, parameter_source, parameter_comments):
    """读取配置文件并更新参数."""
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
            
            # 更新参数
            for section in config:
                for key, value in config[section].items():
                    if parameter_source[key] == "default":
                        if isinstance(value, dict) and 'value' in value:
                            setattr(args, key, value['value'])
                        else:
                            setattr(args, key, value)
                        parameter_source[key] = "config_file"
            return True
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Configuration file not found{Colors.RESET}")
        return False
    except json.JSONDecodeError:
        print(f"{Colors.RED}Error: Invalid JSON format in configuration file{Colors.RESET}")
        return False

def export_configuration(args, model_config, device_config, memory_config, parameter_comments, config_file):
    """保存当前参数到配置文件."""
    try:
        config = {
            "Basic Configuration": {
                "graph_path": {
                    "value": args.graph_path,
                    "comment": parameter_comments["graph_path"]
                },
                "output_path": {
                    "value": args.output_path,
                    "comment": parameter_comments["output_path"]
                },
                "dump": {
                    "value": args.dump,
                    "comment": parameter_comments["dump"]
                }
            },
            "Model Configuration": {
                "inputs": {
                    "value": ",".join(model_config.inputs) if model_config.inputs else "",
                    "comment": parameter_comments["inputs"]
                },
                "outputs": {
                    "value": ",".join(model_config.outputs) if model_config.outputs else "",
                    "comment": parameter_comments["outputs"]
                },
                "strategy": {
                    "value": ",".join(model_config.strategy) if model_config.strategy else "",
                    "comment": parameter_comments["strategy"]
                },
                "dynamic_shape": {
                    "value": ",".join(model_config.dynamic_shape) if model_config.dynamic_shape else "",
                    "comment": parameter_comments["dynamic_shape"]
                },
                "isstream": {
                    "value": model_config.isstream,
                    "comment": parameter_comments["isstream"]
                }
            },
            "Device Configuration": {
                "platform": {
                    "value": device_config.platform,
                    "comment": parameter_comments["platform"]
                },
                "ramsize": {
                    "value": device_config.ramsize,
                    "comment": parameter_comments["ramsize"]
                },
                "psramsize": {
                    "value": device_config.psramsize,
                    "comment": parameter_comments["psramsize"]
                }
            },
            "Memory Preallocate Configuration": {
                "dma_prefetch": {
                    "value": memory_config.dma_prefetch,
                    "comment": parameter_comments["dma_prefetch"]
                },
                "memory": {
                    "value": ",".join(f"{k}:{v}" for k, v in memory_config.storage_location.items()),
                    "comment": parameter_comments["memory"]
                },
                "threshold1": {
                    "value": memory_config.threshold1,
                    "comment": parameter_comments["threshold1"]
                },
                "threshold2": {
                    "value": memory_config.threshold2,
                    "comment": parameter_comments["threshold2"]
                },
                "threshold3": {
                    "value": memory_config.threshold3,
                    "comment": parameter_comments["threshold3"]
                },
                "threshold4": {
                    "value": memory_config.threshold4,
                    "comment": parameter_comments["threshold4"]
                }
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"{Colors.RED}Error saving configuration file: {str(e)}{Colors.RESET}")
        return False

def parse_arguments() -> Tuple[argparse.Namespace, Dict, Dict]:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="Welcome to use tpacker",
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Basic Configuration
    parser.add_argument(
        "-g", "--graph_path", required=False, type=str, help="Path of ONNX graph file"
    )
    parser.add_argument(
        "-o", "--output_path", default="model.pkg", type=str, help="Output file path"
    )
    parser.add_argument(
        "-d", "--dump", default='True', type=parse_boolean,
        help="Switch for print intermediate information"
    )

    # Model Configuration
    model_config = parser.add_argument_group("Model Configuration")
    model_config.add_argument(
        "--inputs", default='', type=str,
        help="Comma-separated list of input node names"
    )
    model_config.add_argument(
        "--outputs", default='', type=str,
        help="Comma-separated list of output node names"
    )
    model_config.add_argument(
        "-c", "--dynamic_shape", default="", type=str,
        help="Dynamic shape configuration (e.g. name1=min:max:factor)"
    )
    model_config.add_argument(
        "-s", "--strategy", default=None, type=str, 
        help="ignore methods of op fusion", choices=["Remove_QuantDequant"]
    )
    model_config.add_argument(
        "--isstream", default=None, type=str, choices=[None, "split_h", "split_w"],
        help="Enable stream processing"
    )

    # Device Configuration
    device_config = parser.add_argument_group("Device Configuration")
    device_config.add_argument(
        "-p", "--platform", type=str, choices=get_platform_list(),
        help="Target platform"
    )
    device_config.add_argument(
        "-r", "--ramsize", default=655360, type=int,
        help="Maximum valid Share-Memory size"
    )
    device_config.add_argument(
        "--psramsize", default=8388608, type=int,
        help="Maximum valid PSRAM size"
    )

    # Memory Preallocate Configuration
    memory_config = parser.add_argument_group("Memory Preallocate Configuration")
    memory_config.add_argument(
        "--dma_prefetch", default='True', type=parse_boolean,
        help="DMA prefetching function switch"
    )
    memory_config.add_argument(
        "-m", "--memory", default='', type=str,
        help="Specify the storage location of node data as either PSRAM or shared memory.(e.g. inputs[0]:share-memory)"
    )
    memory_config.add_argument(
        "--threshold1", default=655360, type=int, 
        help="Set the maximum weight size for convolutional operators."
    )
    memory_config.add_argument(
        "--threshold2", default=655360, type=int, 
        help="Set the maximum output size for convolutional operators."
    )
    memory_config.add_argument(
        "--threshold3", default=655360, type=int, 
        help="Set the maximum output size for linearint operators."
    )
    memory_config.add_argument(
        "--threshold4", default=655360, type=int,
        help="Set the maximum size for nodes in shared memory."
    )

    # 配置文件相关参数
    parser.add_argument(
        "--config_file", type=str,
        help="Path to configuration file (JSON format)"
    )
    parser.add_argument(
        "--export_config", type=str,
        help="Export current configuration to a JSON file"
    )

    args = parser.parse_args()

    # 确保至少提供了--graph或--config-file
    if not args.graph_path and not args.config_file:
        parser.error("Either --graph_path or --config-file must be provided.")

    # 记录每个参数的来源
    parameter_source = {}

    # 创建参数注释字典
    parameter_comments = {}

    for action in parser._actions:
        if action.dest != 'help':
            param_name = action.dest
            # 获取参数的注释（help信息）
            comment = action.help if action.help else "No comment provided"
            parameter_comments[param_name] = comment

            if getattr(args, param_name) is not None:
                if action.default == getattr(args, param_name):
                    parameter_source[param_name] = "default"
                else:
                    parameter_source[param_name] = "command_line"
            else:
                parameter_source[param_name] = "default"
    print(f"{Colors.GREEN}1.1 Parse command line input passed{Colors.RESET}")

    # 读取配置文件
    if args.config_file:
        read_config_file(args, parameter_source, parameter_comments)
        print(f"{Colors.GREEN}1.2 Parse config file input passed{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}1.2 Parse config file input skipped{Colors.RESET}")
    return args, parameter_comments

def parse_parameters(args) -> Tuple[ModelConfig, DeviceConfig, MemoryConfig]:
    """解析参数并返回配置对象."""
    model_config = ModelConfig(
        inputs=split_csv_string(args.inputs),
        outputs=split_csv_string(args.outputs),
        strategy=split_csv_string(args.strategy),
        dynamic_shape=parse_shape_config(args.dynamic_shape),
        isstream=args.isstream
    )

    device_config = DeviceConfig(
        platform=args.platform,
        ramsize=args.ramsize,
        psramsize=args.psramsize
    )

    memory_config = MemoryConfig(
        dma_prefetch=args.dma_prefetch,
        memory=_parse_memory(args.memory),
        threshold1=args.threshold1,
        threshold2=args.threshold2,
        threshold3=args.threshold3,
        threshold4=args.threshold4
    )

    print(f"{Colors.GREEN}1.3 Parse parameter all passed{Colors.RESET}")
    return model_config, device_config, memory_config

__all__ = ["parse_arguments", "parse_parameters", "export_configuration"]