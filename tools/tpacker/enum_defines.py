# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

from enum import Enum
from typing import List, Tuple, Optional, Dict

# Alignment functions
def ALIGN(size: int, alignment: int) -> int:
    """Align the given size to the specified alignment boundary."""
    return (size + alignment - 1) // alignment * alignment

ALIGN2 = lambda x: ALIGN(x, 2)
ALIGN4 = lambda x: ALIGN(x, 4)
ALIGN8 = lambda x: ALIGN(x, 8)
ALIGN16 = lambda x: ALIGN(x, 16)
ALIGN32 = lambda x: ALIGN(x, 32)

# Color constants for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    RESET = '\033[0m'

# Base class for enums with string conversion
class EnumBase(Enum):
    @staticmethod
    def from_str(cls: type, value: str) -> Enum:
        """Convert a string to the corresponding enum value."""
        value = value.upper().replace('-', '_')
        for member in cls:
            if member.name == value:
                return member
        raise ValueError(f"Invalid value '{value}' for enum '{cls.__name__}'")

# Device types
class DevType(EnumBase):
    HIFI = 0
    LUNA = 1
    RISC_V = 2

# Memory types
class MemType(EnumBase):
    FLASH = 0
    PSRAM = 1
    SHARE_MEM = 2

# Tensor layouts
class Layout(EnumBase):
    NCHW = 0
    NHWC = 1
    NCWH = 2
    NHWC8 = 3
    NWHC8 = 4
    NHWC16 = 5
    WHCN = 6

# Tensor types
class TensorType(EnumBase):
    Normal = 0
    Input = 2
    Output = 3
    Empty = 4

# Model configuration
class ModelConfig:
    """Configuration parameters for the model."""
    def __init__(self, inputs: List[str], outputs: List[str], strategy: List[str], dynamic_shape: Dict[str, Tuple[int]], isstream: Optional[str]):
        """
        Initialize the model configuration.
        
        Args:
            inputs: List of input node names.
            outputs: List of output node names.
            dynamic_shape: Dictionary mapping node names to their dynamic shapes.
            isstream: Stream processing mode, optional.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.strategy = strategy
        self.dynamic_shape = dynamic_shape
        self.isstream = isstream

# Device configuration
class DeviceConfig:
    """Configuration parameters for the target device."""
    def __init__(self, platform: str, ramsize: int, psramsize: int):
        """
        Initialize the device configuration.
        
        Args:
            platform: Target platform identifier.
            ramsize: Maximum size of shared memory.
            psramsize: Maximum size of PSRAM.
        """
        self.platform = platform
        self.ramsize = ramsize
        self.psramsize = psramsize

# Memory allocation configuration
class MemoryConfig:
    """Configuration parameters for memory pre-allocation."""
    def __init__(self, dma_prefetch: bool, memory: Dict[str, Tuple[int]], 
                 threshold1: int, threshold2: int, threshold3: int, threshold4: int):
        """
        Initialize the memory configuration.
        
        Args:
            dma_prefetch: Enable DMA prefetch functionality.
            storage_location: Dictionary specifying memory locations for data nodes.
            threshold1: Maximum size for convolution operator weights.
            threshold2: Maximum size for convolution operator outputs.
            threshold3: Maximum size for linear operator outputs.
            threshold4: Maximum size for nodes in shared memory.
        """
        self.dma_prefetch = dma_prefetch
        self.storage_location = memory
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.threshold3 = threshold3
        self.threshold4 = threshold4

__all__ = [
    "ALIGN2", "ALIGN4", "ALIGN8", "ALIGN16", "ALIGN32",
    "Colors", "DevType", "MemType", "Layout", "TensorType",
    "ModelConfig", "DeviceConfig", "MemoryConfig"
]