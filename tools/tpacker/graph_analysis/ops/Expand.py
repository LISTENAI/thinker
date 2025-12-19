import numpy as np
from ...xsympy import is_sympy
from .base import Operator, OperatorAttrs, register_op

@register_op
class Expand(Operator):
    def __init__(self, attrs={}):
        """Initialize the Expand operator with given attributes."""
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor's shape and data based on input tensors."""
        inputs = self.inputs
        X = inputs[0]
        shape2 = list(inputs[1].data)

        # Ensure both shapes have the same length by padding with 1s
        shape1 = list(X.shape)
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2), "Shape lengths must be equal after padding"

        # Determine the output shape
        output_shape = []
        for s1, s2 in zip(shape1, shape2):
            if s1 == 1:
                output_shape.append(s2)
            elif s2 == 1:
                output_shape.append(s1)
            elif s1 == s2:
                output_shape.append(s1)
            else:
                raise ValueError("Incompatible dimensions for expansion")

        Y = X.clone(shape=tuple(output_shape))

        # Perform data expansion if applicable
        if X.has_data() and not is_sympy(output_shape):
            input_shape = list(X.data.shape)
            if len(input_shape) < len(output_shape):
                input_shape = [1] * (len(output_shape) - len(input_shape)) + input_shape
            tile_shape = np.array(output_shape) // np.array(input_shape)
            Y.data = np.tile(X.data, tuple(tile_shape))

        self.outputs = [Y]

__all__ = ['Expand']