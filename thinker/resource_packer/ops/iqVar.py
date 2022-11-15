import math
import numpy as np

from ...graph import Tensor
from ...enum_defines import DevType
from .base import Operator, register_op


@register_op
class iqVar(Operator):
    def infer_tensor(self):
        inputs = self.inputs
        X = inputs[0]
        x_shape = list(X.shape)
        x_shape[2] = 1

        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert self.inputs[0].scale == temp

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001
        self.outputs[0].scale = temp

        Y = X.clone(shape=tuple(x_shape), dtype=X.dtype, scale=int(temp))
        self.outputs[0] = Y

    def get_workspace(self, dev_type: DevType):
        workspace_size = 0
        workspace_size += (
            self.inputs[0].nbytes
            * self.inputs[0].dtype.itemsize
            / self.inputs[0].shape[2]
            * 3
        )
        max_workspace = Tensor.from_shape([workspace_size], np.int32, dev_type)
        return [max_workspace]


__all__ = ["iqVar"]
