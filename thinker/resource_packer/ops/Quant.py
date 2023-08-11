import math
import numpy as np

from .utils import QuantType
from .._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class QuantAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "scale_x" in self.attrs
        assert "data_bits" in self.attrs
        assert self.attrs['data_bits'] in (8, 16, 32)

    def serialize(self) -> bytes:
        attrs = tffi.new("QuantAttrs *")
        attrs.data_bits = self.attrs["data_bits"]
        quant_type = QuantType.from_str(self.attrs.get("platform_quant", "qmax_quant"))
        attrs.quant_type = quant_type.value
        return bytes(tffi.buffer(attrs))


@register_op
class Quant(Operator):
    def __init__(self, attrs={}):
        self.attrs = QuantAttrs(attrs)

    def infer_tensor(self):
        attrs = self.attrs
        X = self.inputs[0]
        data_bits = int(attrs.get("data_bits", "8"))
        assert (
            X.dtype == np.float32 and data_bits in (8, 16, 32)
        ), "Quant just support switch float32 to int8"

        scale_o = attrs["scale_x"]
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001
        if(data_bits == 8):
            Y = X.clone(dtype=np.dtype("i1"), scale=int(temp), zero=0)
        elif(data_bits == 16):
            Y = X.clone(dtype=np.dtype("i2"), scale=int(temp), zero=0)
        else:
            Y = X.clone(dtype=np.dtype("i4"), scale=int(temp), zero=0)

        self.outputs = [Y]

  
__all__ = ["Quant"]
