# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2025.09.31

import onnx
import struct
import numpy as np
from typing import Dict, List, Tuple

from .devices import *
from .enum_defines import Colors, DeviceConfig

def load_device_info(graph_platform: str, device_config: DeviceConfig) -> Device:
    """Load device information and adjust memory settings based on configuration
    
    Args:
        graph_platform (str): Target platform name
        device_config (DeviceConfig): Device configuration
        
    Returns:
        Device: Configured device object
    """
    # Check and set platform
    if device_config.platform:
        assert graph_platform.upper() == device_config.platform.upper(), "Platform mismatch"
    else:
        device_config.platform = graph_platform
    
    # Create device instance
    device = create_device_from_name(graph_platform)
    
    # Adjust SRAM size
    if device_config.ramsize < device.sram_size:
        print(f'{Colors.YELLOW}Warning: The set sram size({device_config.ramsize}) is less than the default value({device.sram_size}) and it works{Colors.RESET}')
        device.sram_size = device_config.ramsize
    elif device_config.ramsize > device.sram_size:
        print(f'{Colors.RED}The set sram size({device_config.ramsize}) is greater than the default value, invalidte the set value({device.sram_size}){Colors.RESET}')
        device_config.ramsize = device.sram_size
    else:
        print(f"{Colors.GREEN}4.1 set sram size passed{Colors.RESET}")

    # Adjust PSRAM size
    if device_config.psramsize < device.psram_size:
        print(f'{Colors.YELLOW}Warning: The set psram size({device_config.psramsize}) is less than the default value({device.psram_size}) and it works{Colors.RESET}')
        device.psram_size = device_config.psramsize
    elif device_config.psramsize > device.psram_size:
        print(f'{Colors.RED}The set psram size({device_config.psramsize}) is greater than the default value, invalidte the set value({device.psram_size}){Colors.RESET}')
        device_config.psramsize = device.psram_size
    else:
        print(f"{Colors.GREEN}4.2 set psram size passed{Colors.RESET}")

    return device

__all__ = ["load_device_info"]