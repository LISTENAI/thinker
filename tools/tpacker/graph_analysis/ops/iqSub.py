import math
import numpy as np
from typing import List
from ...graph import Tensor
from ...enum_defines import DevType, MemType, ALIGN4
from .base import iqBinaryOperator, register_op

@register_op
class iqSub(iqBinaryOperator):
    def infer_tensor(self, dynamic_shape):
        super().infer_tensor(dynamic_shape)
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            x1, x2 = self.inputs
            assert tuple(x1.shape) == tuple(x2.shape), "iqSub on venus does not support broadcasting"
            assert x1.dtype == x2.dtype == np.int8, "iqSub on venus only supports int8"
            assert x1.scale >= self.outputs[0].scale and x2.scale >= self.outputs[0].scale, \
                "iqSub on venus requires input scales not less than the output scale"
            assert x1.scale - self.outputs[0].scale <= 63 and x2.scale - self.outputs[0].scale <= 63, \
                "iqSub on venus exceeds q7 scalar shift range"
        elif platform == "venusA":
            x1, x2 = self.inputs
            assert tuple(x1.shape) == tuple(x2.shape), "iqSub on venusA runtime does not support broadcasting"
            assert x1.dtype == np.int8 and x2.dtype == np.int8, "iqSub on venusA runtime only supports int8"
            qx = int(math.log(self.attrs.get("scale_x", 1.0), 2))
            qy = int(math.log(self.attrs.get("scale_y", 1.0), 2))
            qo = int(math.log(self.attrs.get("scale_o", 1.0), 2))
            assert qo <= qx <= qo + 63, "iqSub on venusA requires scale_x shift in [0, 63]"
            assert qo <= qy <= qo + 63, "iqSub on venusA requires scale_y shift in [0, 63]"

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqSub operation."""
        x1 = self.inputs[0]
        x2 = self.inputs[1]
        size = x1.nbytes
        Y = self.outputs[0]

        scale_x = self.attrs["scale_x"]
        scale_y = self.attrs["scale_y"]
        scale_o = self.attrs["scale_o"]

        workspace_size = 0
        if (scale_x != scale_o) or x1.mem_type != MemType.SHARE_MEM:
            workspace_size += ALIGN4(size)
        if (scale_y != scale_o) or x2.mem_type != MemType.SHARE_MEM:
            workspace_size += ALIGN4(size)
        if Y.mem_type != MemType.SHARE_MEM:
            workspace_size += size

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

__all__ = ["iqSub"]
