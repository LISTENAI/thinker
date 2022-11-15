import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType
from .base import iqBinaryOperator, register_op, BaseLayout


@register_op
class iqAdd(iqBinaryOperator, BaseLayout):
    def infer_tensor(self):
        X = self.inputs[0]
        scale_x = self.attrs.get("scale_x", 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert self.inputs[0].scale == temp

        scale_y = self.attrs.get("scale_y", 1.0)
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert self.inputs[1].scale == temp

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        Y = X.clone(scale=int(temp))
        self.outputs[0] = Y

    def get_workspace(self, dev_type: DevType) -> List[Tensor]:
        x1 = self.inputs[0]
        x2 = self.inputs[1]

        scale_x = self.attrs["scale_x"]
        scale_y = self.attrs["scale_y"]

        workspace_size = 0
        if scale_x != scale_y:
            workspace_size = x1.nbytes
            max_workspace = Tensor.from_shape([workspace_size], np.int8, x1.mem_type)
            return [max_workspace]
        else:
            return []


__all__ = ["iqAdd"]
