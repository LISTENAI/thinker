import math
import numpy as np

from .utils import QuantType
from .._type._ctype import tffi
from ...enum_defines import DevType
from typing import Any, Dict, List, Optional
from .base import Operator, OperatorAttrs, register_op


class RequantAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "scale_x" in self.attrs
        assert "scale_o" in self.attrs
        assert "data_bits" in self.attrs
        assert "o_bist" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("RequantAttrs *")

        attrs.scale_x = self.attrs["scale_x"]
        attrs.scale_o = self.attrs["scale_o"]
        attrs.o_bits = self.attrs["o_bits"]
        attrs.data_bits = self.attrs["data_bits"]
        attrs.quant_type = self.attrs["quant_type"].value

        return bytes(tffi.buffer(attrs))


@register_op
class Requant(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        X = self.inputs[0]

        scale_x = self.attrs.get("scale_x")
        temp1 = math.log(scale_x, 2)
        assert temp1 == int(temp1)
        assert self.inputs[0].scale == int(temp1)

        scale_o = self.attrs.get("scale_o")
        temp1 = math.log(scale_o, 2)
        assert temp1 == int(temp1)

        if 32 == self.attrs["o_bits"]:
            data_type = np.dtype("i4")
        elif 16 == self.attrs["o_bits"]:
            data_type = np.dtype("i2")
        elif 8 == self.attrs["o_bits"]:
            data_type = np.dtype("i1")
        else:
            raise ZeroDivisionError(f"o_bits is not equal data_bits.")

        Y = X.clone(dtype=data_type, scale=int(temp1))
        self.outputs = [Y]


__all__ = ["Requant"]
