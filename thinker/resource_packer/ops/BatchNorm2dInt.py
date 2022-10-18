import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType
from .base import Operator, register_op


@register_op
class BatchNorm2dInt(Operator):
    def infer_tensor(self):
        assert "scale_x" in self.attrs.attrs
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert (
            abs(temp - int(temp)) < 0.000001
        ), "scale of inputs in BatchNorm2dInt must be 2^Q"
        assert self.inputs[0].scale == int(temp)

        assert "scale_w" in self.attrs.attrs
        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w, 2)
        assert (
            abs(temp - int(temp)) < 0.000001
        ), "scale of weight in BatchNorm2dInt must be 2^Q"
        self.inputs[1].scale = int(temp)

        self.inputs[2].scale = self.inputs[0].scale + self.inputs[1].scale

        assert "scale_o" in self.attrs.attrs
        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert (
            abs(temp - int(temp)) < 0.000001
        ), "scale of output in BatchNorm2dInt must be 2^Q"

        Y = self.inputs[0].clone(scale=int(temp))

        self.outputs = [Y]
        return

    def get_workspace(self, dev_type: DevType) -> List[Tensor]:
        X = self.inputs[0]
        workspace_bytes = X.shape[2] * X.shape[3] * 4
        max_workspace = Tensor.from_shape([workspace_bytes], np.int8, dev_type)
        return [max_workspace]


__all__ = ["BatchNorm2dInt"]
