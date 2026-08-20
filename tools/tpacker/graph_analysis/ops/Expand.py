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
        assert len(inputs) == 2, "Expand requires data and shape inputs"
        X = inputs[0]
        shape_input = inputs[1]
        assert shape_input.has_data(), "Expand shape input must be constant"
        assert shape_input.dtype in (np.dtype(np.int32), np.dtype(np.int64)), \
            "Expand shape input must be int32 or int64"
        assert len(shape_input.shape) == 1, "Expand shape input must be one-dimensional"
        shape_data = np.asarray(shape_input.data)
        assert shape_data.size == shape_input.shape[0], \
            "Expand shape input data does not match its tensor shape"
        shape2 = shape_data.reshape(-1).tolist()
        assert len(shape2) <= 7, "Expand runtime supports at most 7 dimensions"
        assert all(isinstance(dim, (int, np.integer)) and dim > 0 for dim in shape2), \
            "Expand dimensions must be positive integers"
        assert all(is_sympy(dim) or dim > 0 for dim in X.shape), \
            "Expand input dimensions must be positive"
        assert X.dtype.itemsize > 0 and float(X.bits) >= 1, \
            "Expand runtime only supports byte-addressable data types"

        # Ensure both shapes have the same length by padding with 1s
        shape1 = list(X.shape)
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) <= 7, "Expand runtime supports at most 7 dimensions"

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
            Y.data = np.broadcast_to(X.data, tuple(output_shape)).copy()

        self.outputs = [Y]

__all__ = ['Expand']
