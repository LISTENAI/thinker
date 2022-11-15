import math

from ...enum_defines import DevType
from .base import iqBinaryOperator, register_op


@register_op
class iqMul(iqBinaryOperator):
    def infer_tensor(self):
        X = self.inputs[0]
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert self.inputs[0].scale == temp

        scale_y = self.attrs.get("scale_y")
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert self.inputs[1].scale == temp

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        Y = X.clone(scale=int(temp))


__all__ = ["iqMul"]
