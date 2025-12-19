import numpy as np
from typing import List
from ...graph import Tensor
from .base import iqBinaryOperator, register_op

@register_op
class Pad(iqBinaryOperator):
    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 3, "Pad operator must have exactly 3 inputs"

        X = inputs[0]
        pads = inputs[1]

        shape = list(X.shape)
        assert len(pads.shape) == 1, "pads must be a 1D tensor"

        pads_data = pads.data
        if len(pads_data) == 2:
            # Pad last dimension on both sides
            shape[-1] += pads_data[1] * 2
            # Pad second last dimension on both sides
            shape[-2] += pads_data[0] * 2
        elif len(pads_data) == 4:
            # Pad last dimension: left and right
            shape[-1] += pads_data[1] + pads_data[3]
            # Pad second last dimension: top and bottom
            shape[-2] += pads_data[0] + pads_data[2]

        # Create output tensor with the new shape
        Y = X.clone(shape=tuple(shape))
        self.outputs = [Y]

__all__ = ["Pad"]