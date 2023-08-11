import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from ...enum_defines import ALIGN16, ALIGN32
from .base import iqBinaryOperator, register_op


@register_op
class BmmInt(iqBinaryOperator):
    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 2, "num of inputs in BmmInt must be 2"

        X = inputs[0]
        W = inputs[1]
        assert X.dtype == W.dtype
        assert len(X.shape) >= 1
        assert len(W.shape) > 1
        x_shape = list(X.shape)
        w_shape = list(W.shape)
        assert X.shape[-1] == W.shape[-2]

        scale_x = self.attrs.get("scale_x", 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        if self.inputs[0].scale != -1:
            assert self.inputs[0].scale == int(temp)
        else:
            self.inputs[0].scale = int(temp)

        scale_y = self.attrs.get("scale_y", 1.0)
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 0.000001
        if self.inputs[1].scale != -1:
            assert self.inputs[1].scale == int(temp)
        else:
            self.inputs[1].scale = int(temp)

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        x_h = x_shape[0]
        x_w = x_shape[1]

        M = ALIGN16(x_h)
        N = ALIGN32(x_w)
        if X.dtype == np.int8:
            assert M * N < 65536, "left matmul of linearint must less 64KB"
        elif X.dtype == np.int16:
            assert M * N < 32768, "left matmul of linearint must less 64KB"
        elif X.dtype == np.int32:
            assert M * N < 16384, "left matmul of linearint must less 64KB"
        else:
            raise (f"BmmInt Not supported this data type.")

        shape = list(x_shape)
        shape[-1] = w_shape[-1]

        Y = Tensor(shape=shape, dtype=X.dtype, scale=int(temp))
        self.outputs = [Y]
        
    def get_workspace(self, dev_type: DevType) -> List[Tensor]:
        X = self.inputs[0]
        Y = self.outputs[0]
        workspace_bytes = 0
        if X.mem_type != MemType.SHARE_MEM:
            workspace_bytes = X.nbytes
        if Y.mem_type != MemType.SHARE_MEM:
            workspace_bytes = max(workspace_bytes, Y.nbytes)
        if workspace_bytes != 0:
            return [Tensor.from_shape([workspace_bytes], np.int8, dev_type)]
        else:
            return []

__all__ = ["BmmInt"]
