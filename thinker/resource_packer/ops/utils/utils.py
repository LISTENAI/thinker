import math
import sympy
from enum import Enum


class XEnum(Enum):
    @classmethod
    def from_str(cls, name, default=None):
        name_ = name.name if isinstance(name, Enum) else str(name)
        if name_.upper() in cls.__dict__:
            return cls.__dict__[name_.upper()]
        if default is not None:
            return default
        raise LookupError(f"Unknown enum name: {name} for {cls.__name__}.")

    @classmethod
    def from_int(cls, value):
        value_ = value.value if isinstance(value, Enum) else int(value)
        return cls(value_)


class QuantType(XEnum):
    QSHIFT_QUANT = 0
    QMAX_QUANT = 1
    TF_QUANT = 2
    NORMAL_QUANT = 3
    LUNA_QUANT = 4


class CeilMode(XEnum):
    NO = 0
    CEIL = 1


class AutoPad(XEnum):
    NOTSET = 0
    SAME_UPPER = 1
    SAME_LOWER = 2
    VALID = 3


class PadMode(XEnum):
    CONSTANT = 0
    REFLECT = 1
    EDGE = 2


def attr2tuple(value, default, dim=None):
    # get dim
    if dim == None:
        if isinstance(default, int):
            dim = 1
        else:
            dim = len(default)

    if value is None:
        if isinstance(default, int):
            return tuple(int(default) for i in range(dim))
        return tuple(int(default[i]) for i in range(dim))
    # str2int
    if isinstance(value, str):
        value = eval(value)

    if isinstance(value, int):
        return tuple(int(value) for i in range(dim))
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return tuple(int(default) for i in range(dim))
        elif len(value) == 1:
            return tuple(int(value[0]) for i in range(dim))
        elif len(value) == 2:
            if dim == 2:
                return tuple(int(value[i]) for i in range(dim))
            elif dim == 4:
                return tuple(int(value[i // 2]) for i in range(dim))
            else:
                raise ValueError(f"Invalid attr {value}")
        elif len(value) == 4:
            return tuple(int(value[i]) for i in range(dim))
        elif len(value) == 3:
            return tuple(int(value[i]) for i in range(dim))
        elif len(value) == 6:
            return tuple(int(value[i]) for i in range(dim))
        else:
            raise ValueError(f"Invalid attr {value}")
    else:
        raise ValueError(f"Invalid attr {value}")


__all__ = ["XEnum", "QuantType", "CeilMode", "AutoPad", "PadMode", "attr2tuple"]
