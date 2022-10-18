import math
import numpy as np

from ...graph import Tensor
from .._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class LayerNormIntAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        attrs = tffi.new("LayerNormIntAttrs *")

        attrs.axis = 1
        attrs.keepdims = 1
        attrs.eps = 0.000001

        return bytes(tffi.buffer(attrs))


@register_op
class LayerNormInt(Operator):
    def __init__(self, attrs={}):
        self.attrs = LayerNormIntAttrs(attrs)

    def infer_tensor(self):
        X = self.inputs[0]

        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "scale of inputs must be 2^Q"
        assert X.scale == int(temp)

        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w, 2)
        assert abs(temp - int(temp)) < 0.000001, "scale of weight must be 2^Q"
        self.inputs[1].scale = int(temp)

        if len(self.inputs) == 3:
            self.inputs[2].scale = self.inputs[0].scale + self.inputs[1].scale
        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "scale of output must be 2^Q"

        Y = X.clone(scale=int(temp))

        self.outputs = [Y]
        return

    def get_workspace(self, dev_type):
        x = self.inputs[0]
        w = self.inputs[1]
        w_size = 1
        for i in w.shape:
            w_size *= i
        workspace_size = w_size * 4 * self.inputs[-1].dtype.itemsize
        max_workspace = Tensor.from_shape([workspace_size], np.int8, dev_type)
        return [max_workspace]


__all__ = ["LayerNormInt"]
