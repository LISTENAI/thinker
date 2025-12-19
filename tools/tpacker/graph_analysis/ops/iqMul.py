import math
import numpy as np
from typing import List
from ...graph import Tensor
from ...enum_defines import DevType
from .base import iqBinaryOperator, register_op

@register_op
class iqMul(iqBinaryOperator):
    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqMul operation."""
        input1 = self.inputs[0]
        input2 = self.inputs[1]
        workspace_size = 0

        # Check if inputs meet specific shape conditions
        if (len(input1.shape) == 4 and 
            len(input2.shape) == 4 and 
            input1.shape[0] == input2.shape[0] and 
            input1.shape[1] == input2.shape[1] and 
            (input1.shape[2] > 1 or input1.shape[3] > 1) and 
            input2.shape[2] == 1 and 
            input2.shape[3] == 1):
            
            # Calculate workspace size based on input dimensions
            workspace_size = input1.shape[2] * input1.shape[3]
            workspace_size += input1.shape[1] * input1.shape[2] * input1.shape[3]

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, input1.mem_type)]
        return []

__all__ = ["iqMul"]