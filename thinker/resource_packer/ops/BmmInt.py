import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from ...enum_defines import ALIGN4, ALIGN8
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

        shape = list(x_shape)
        shape[-1] = w_shape[-1]

        Y = Tensor(shape=shape, dtype=X.dtype, scale=int(temp))
        self.outputs = [Y]
        
    def get_workspace(self, dev_type: DevType) -> List[Tensor]:
        workspace_bytes = 0
        M = self.inputs[0].shape[-2]
        N = self.inputs[0].shape[-1]
        L = self.inputs[1].shape[-1]
        int8_condition_l = ALIGN4(M) * ALIGN8(N)
        int8_condition_r = ALIGN8(N) * ALIGN4(L)
        split_M = M
        if int8_condition_l > 65536:
            split_num = 2
            split_M = math.ceil(M / split_num)
            int8_condition_l_split = ALIGN4(split_M) * ALIGN8(N)
            while int8_condition_l_split > 65536:
                split_num += 1
                split_M = math.ceil(M / split_num)
                int8_condition_l_split = ALIGN4(split_M) * ALIGN8(N)

        if self.inputs[0].mem_type != MemType.SHARE_MEM and self.outputs[0].mem_type != MemType.SHARE_MEM and dev_type == DevType.LUNA:
            workspace_bytes = split_M * max(N, L) + split_M * L * 4
        elif self.inputs[0].mem_type != MemType.SHARE_MEM and dev_type == DevType.LUNA:
            workspace_bytes = split_M * N
        elif self.outputs[0].mem_type != MemType.SHARE_MEM and dev_type == DevType.LUNA:
            workspace_bytes = split_M * L

        if workspace_bytes:
            return [Tensor.from_shape([workspace_bytes], np.int8, dev_type)]
        else:
            return []

__all__ = ["BmmInt"]
