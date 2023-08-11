import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from .base import Operator, register_op


@register_op
class BatchNorm2dInt(Operator):
    def infer_tensor(self):
        assert len(self.inputs[0].shape) == 4, "Just support 4D data yet"

        assert "scale_x" in self.attrs.attrs
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert (
            abs(temp - int(temp)) < 0.000001
        ), "scale of inputs in BatchNorm2dInt must be 2^Q"
        if self.inputs[0].scale != -1:
            assert self.inputs[0].scale == int(temp)
        else:
            self.inputs[0].scale = int(temp)

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
        Y = self.outputs[0]
        workspace_bytes = X.shape[2] * X.shape[3] * 4
        if X.mem_type != MemType.SHARE_MEM or Y.mem_type != MemType.SHARE_MEM:
            workspace_bytes += X.nbytes
        max_workspace = Tensor.from_shape([workspace_bytes], np.int8, dev_type)
        return [max_workspace]


__all__ = ["BatchNorm2dInt"]
