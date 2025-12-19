import math
import sympy
from enum import Enum
import numpy as np

class XEnum(Enum):
    """Base enum class with additional utility methods."""
    @classmethod
    def from_str(cls, name, default=None):
        """Convert a string to an enum value.
        
        Args:
            name (str or Enum): The name of the enum value.
            default (optional): Default value if the name is not found.
            
        Returns:
            Enum: The corresponding enum value.
            
        Raises:
            LookupError: If the name is not found and no default is provided.
        """
        name_ = name.name if isinstance(name, Enum) else str(name)
        if name_.upper() in cls.__dict__:
            return cls.__dict__[name_.upper()]
        if default is not None:
            return default
        raise LookupError(f"Unknown enum name: {name} for {cls.__name__}.")

    @classmethod
    def from_int(cls, value):
        """Convert an integer to an enum value.
        
        Args:
            value (int or Enum): The integer value.
            
        Returns:
            Enum: The corresponding enum value.
        """
        value_ = value.value if isinstance(value, Enum) else int(value)
        return cls(value_)

class RoundMethod(XEnum):
    """Enum for rounding methods."""
    FLOOR = 0
    FLOOR_ADD = 1
    ROUND = 2
    CEIL = 3

class QuantType(XEnum):
    """Enum for quantization types."""
    QSHIFT_QUANT = 0
    QMAX_QUANT = 1
    TF_QUANT = 2
    NORMAL_QUANT = 3
    LUNA_QUANT = 4

class CeilMode(XEnum):
    """Enum for ceiling modes."""
    NO = 0
    CEIL = 1

class AutoPad(XEnum):
    """Enum for auto-padding modes."""
    NOTSET = 0
    SAME_UPPER = 1
    SAME_LOWER = 2
    VALID = 3

class PadMode(XEnum):
    """Enum for padding modes."""
    CONSTANT = 0
    REFLECT = 1
    EDGE = 2

def attr2tuple(value, default, dim=None):
    """Convert a value to a tuple with specified dimensions.
    
    Args:
        value (int, str, list, tuple, or None): The input value.
        default (int or list): Default value to use if the input is None.
        dim (int, optional): The target dimension. If None, infer from default.
        
    Returns:
        tuple: The converted tuple.
        
    Raises:
        ValueError: If the input is invalid.
    """
    # Infer dimension if not provided
    if dim is None:
        dim = 1 if isinstance(default, int) else len(default)
    
    # Use default value if input is None
    if value is None:
        return tuple(int(default) for _ in range(dim)) if isinstance(default, int) else tuple(int(default[i]) for i in range(dim))
    
    # Convert string to integer
    if isinstance(value, str):
        value = eval(value)
    
    # Handle different types
    if isinstance(value, int):
        return tuple(int(value) for _ in range(dim))
    elif isinstance(value, (list, tuple)):
        length = len(value)
        if length == 0:
            return tuple(int(default) for _ in range(dim))
        elif length == 1:
            return tuple(int(value[0]) for _ in range(dim))
        elif length == 2:
            if dim == 2:
                return tuple(int(value[i]) for i in range(dim))
            elif dim == 4:
                return tuple(int(value[i // 2]) for i in range(dim))
            else:
                raise ValueError(f"Invalid attribute value: {value}")
        elif length == 4:
            return tuple(int(value[i]) for i in range(dim))
        elif length == 3:
            return tuple(int(value[i]) for i in range(dim))
        elif length == 6:
            return tuple(int(value[i]) for i in range(dim))
        else:
            raise ValueError(f"Invalid attribute value: {value}")
    else:
        raise ValueError(f"Invalid attribute type: {type(value)}")

# Built-in functions for math operations
_builtin_max = max
_builtin_min = min
_builtin_floor = math.floor
_builtin_ceil = math.ceil

def max(*args):
    """Return the maximum value, supporting sympy symbols."""
    for arg in args:
        if isinstance(arg, sympy.Basic):
            return sympy.Max(*args)
    return _builtin_max(*args)

def min(*args):
    """Return the minimum value, supporting sympy symbols."""
    for arg in args:
        if isinstance(arg, sympy.Basic):
            return sympy.Min(*args)
    return _builtin_min(*args)

def floor(*args):
    """Return the floor value, supporting sympy symbols."""
    for arg in args:
        if isinstance(arg, sympy.Basic):
            return sympy.floor(*args)
    return _builtin_floor(*args)

def ceil(*args):
    """Return the ceiling value, supporting sympy symbols."""
    for arg in args:
        if isinstance(arg, sympy.Basic):
            return sympy.ceiling(*args)
    return _builtin_ceil(*args)

def clip(v, min_v, max_v):
    """Clip the value between min and max."""
    return min(max(v, min_v), max_v)

def combine4bit_8bit(x):
    """Combine 4-bit integers into 8-bit integers."""
    if not (-8 <= x.min() and x.max() < 8):
        raise ValueError("Input must be 4-bit integers (-8 to 7)")
    
    x = x.squeeze()
    new_x = x.reshape(-1, x.shape[-1])
    shape = list(new_x.shape)
    shape[-1] = (shape[-1] + 1) // 2  # Ceiling division
    
    combined = np.zeros(shape, dtype=np.int8)
    
    # Handle odd length by padding with zero
    if new_x.shape[-1] % 2 != 0:
        new_x = np.concatenate([new_x, np.zeros((new_x.shape[0], 1), dtype=np.int8)], axis=1)
    
    for i in range(shape[-1]):
        combined[:, i] = (new_x[:, 2 * i + 1] << 4) | (new_x[:, 2 * i] & 0x0F)
    
    return combined