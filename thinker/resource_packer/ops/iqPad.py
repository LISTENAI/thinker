import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType
from .base import iqBinaryOperator, register_op


@register_op
class iqPad(iqBinaryOperator):
    def infer_tensor(self):
        inputs = self.inputs
        num_input = len(inputs)
        assert num_input == 3
        X = inputs[0]
        pads = inputs[1]
        assert len(X.shape) >= len(pads.shape) and len(X.shape) <= 4, "shape of input must <= 4"
        shape = list(X.shape)
        assert len(pads.shape) == 1
        if pads.shape[0] == 2:
            shape[-1] = shape[-1] + pads.data[1] * 2
            shape[-2] = shape[-2] + pads.data[0] * 2
        elif pads.shape[0] == 4:
            shape[-1] = shape[-1] + pads.data[1] + pads.data[3]
            shape[-2] = shape[-2] + pads.data[0] + pads.data[2]            
        elif pads.shape[0] == 6:
            shape[-1] = shape[-1] + pads.data[2] + pads.data[5]
            shape[-2] = shape[-2] + pads.data[1] + pads.data[4]   
            shape[-3] = shape[-3] + pads.data[0] + pads.data[3]   
        elif pads.shape[0] == 8:
            shape[-1] = shape[-1] + pads.data[3] + pads.data[7]
            shape[-2] = shape[-2] + pads.data[2] + pads.data[6]   
            shape[-3] = shape[-3] + pads.data[1] + pads.data[5] 
            shape[-4] = shape[-4] + pads.data[0] + pads.data[4] 

        Y = Tensor.clone(X, shape=tuple(shape))
        self.outputs = [Y]

__all__ = ["iqPad"]
