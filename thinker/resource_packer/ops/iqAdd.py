import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from .base import iqBinaryOperator, register_op, BaseLayout


@register_op
class iqAdd(iqBinaryOperator, BaseLayout):
    def get_workspace(self, dev_type: DevType) -> List[Tensor]:
        x1 = self.inputs[0]
        x2 = self.inputs[1]
        size = x1.nbytes
        Y = self.outputs[0]

        scale_x = self.attrs["scale_x"]
        scale_y = self.attrs["scale_y"]
        scale_o = self.attrs["scale_o"]

        workspace_size = 0
        if (scale_x != scale_o) or x1.mem_type != MemType.SHARE_MEM:
            workspace_size += size
        if (scale_y != scale_o) or x2.mem_type != MemType.SHARE_MEM:
            workspace_size += size
        if Y.mem_type != MemType.SHARE_MEM:
            workspace_size = max(workspace_size, size)

        max_workspace = Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)
        return [max_workspace]

__all__ = ["iqAdd"]
