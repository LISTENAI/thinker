import numpy as np
import math
from .utils import QuantType
from .._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class iqTanhOperatorAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "scale_x" in self.attrs
        assert "scale_o" in self.attrs
        assert "platform_quant" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("iqBinaryAttrs *")

        quant_type = self.attrs.get("platform_quant")
        attrs.quant_type = QuantType.from_str(quant_type).value

        return bytes(tffi.buffer(attrs))


@register_op
class iqTanh(Operator):
    def __init__(self, attrs={}):
        self.attrs = iqTanhOperatorAttrs(attrs)

    def infer_tensor(self):
        assert len(self.inputs) == 1

        X = self.inputs[0]
        assert X.dtype == np.int16, "input of iqtanh must be int16"

        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert X.scale == int(
            temp
        ), "scale of tensor must be same with scale_x in attribute"
        assert X.scale == 11, "scale of iqtanh input must be Q11"

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        Y = X.clone(scale=int(temp))
        self.outputs = [Y]
        if all([x.has_data() for x in self.inputs]):
            self.forward()


__all__ = ["iqTanh"]
