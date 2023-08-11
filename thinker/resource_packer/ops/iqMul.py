import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType
from .base import iqBinaryOperator, register_op


@register_op
class iqMul(iqBinaryOperator):
    def get_workspace(self, dev_type: DevType) -> List[Tensor]:
        input1_data = self.inputs[0]
        input2_data = self.inputs[1]
        workspace_size = 0
        if len(input1_data.shape) == 4 and len(input2_data.shape) == 4 and \
            input1_data.shape[0] == input2_data.shape[0] and input1_data.shape[1] == input2_data.shape[1] and \
            (input1_data.shape[2]  > 1 or input1_data.shape[3] > 1) and \
            input2_data.shape[2] == 1 and input2_data.shape[3] == 1:
            workspace_size = input1_data.shape[2]*input1_data.shape[3] 
            workspace_size += input1_data.shape[1]*input1_data.shape[2]*input1_data.shape[3]

        max_workspace = Tensor.from_shape(
            [workspace_size], np.int8, self.inputs[0].mem_type
        )
        return [max_workspace]

__all__ = ["iqMul"]
