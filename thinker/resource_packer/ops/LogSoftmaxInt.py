import math
import numpy as np

from ...graph import Tensor
from .._type._ctype import tffi
from ...enum_defines import DevType
from .base import Operator, OperatorAttrs, iqUnaryOperator, iqUnaryOperatorAttrs, register_op


class LogSoftmaxIntAttrs(iqUnaryOperatorAttrs):
    def checkparams(self) -> None:
        assert "dim" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("LogSoftmaxIntAttrs *")
        attrs.axis = self.attrs["dim"]
        return bytes(tffi.buffer(attrs))


@register_op
class LogSoftmaxInt(iqUnaryOperator):
    def __init__(self, attrs={}):
        self.attrs = LogSoftmaxIntAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1
        X = inputs[0]

        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert X.scale == int(temp), "scale of inputs in LogSoftmaxInt must be 2^Q"
        if X.scale != -1 :
            assert X.scale == int(temp), "scale of tensor must be same with scale_x in attribute"
        else:
            self.inputs[0].scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        Y = X.clone(scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self, dev_type: DevType):
        axis = self.attrs.get("dim", -1)
        workspace_bytes = self.inputs[0].shape[axis] * 2
        max_workspace = Tensor.from_shape([workspace_bytes], np.int32, dev_type)
        return [max_workspace]

  
__all__ = ["LogSoftmaxInt"]
