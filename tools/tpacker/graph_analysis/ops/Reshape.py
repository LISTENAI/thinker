from functools import reduce
import numpy as np

from ...xsympy import is_sympy
from .base import Operator, OperatorAttrs, register_op


@register_op
class Reshape(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape based on input and reshape parameters"""
        assert len(self.inputs) == 2, "Reshape requires data and shape inputs"
        X = self.inputs[0]
        shape_input = self.inputs[1]
        assert shape_input.has_data(), "Reshape shape input must be constant"
        assert shape_input.dtype in (np.dtype(np.int32), np.dtype(np.int64)), \
            "Reshape shape input must be int32 or int64"
        assert len(shape_input.shape) == 1, "Reshape shape input must be one-dimensional"
        shape_data = np.asarray(shape_input.data)
        assert shape_data.size == shape_input.shape[0], \
            "Reshape shape input data does not match its tensor shape"
        assert shape_data.size <= 7, "Reshape runtime supports at most 7 dimensions"
        assert self.attrs.get("allowzero", 0) == 0, "Reshape only supports allowzero=0"
        input_shape = list(X.shape)
        output_shape = shape_data.reshape(-1).tolist()

        # Handle special cases: 0 and -1 in output shape
        dim_minus_one = None
        for i, s in enumerate(output_shape):
            if s == 0:
                assert i < len(input_shape), "Reshape zero dimension exceeds input rank"
                output_shape[i] = input_shape[i]
            elif s == -1:
                assert dim_minus_one is None, "Only one -1 is allowed in reshape"
                output_shape[i] = 1
                dim_minus_one = i
            else:
                assert is_sympy(s) or s > 0, "Reshape dimensions must be positive, zero, or -1"

        # Collect symbolic dimensions
        symbolic_input_dims = [s for s in input_shape if is_sympy(s)]
        symbolic_output_dims = [s for s in output_shape if is_sympy(s)]

        # Check consistency of symbolic dimensions
        same_dims = []
        if symbolic_input_dims and symbolic_output_dims:
            same_dims = symbolic_input_dims
            for dim in same_dims:
                assert input_shape.count(dim) >= 1, "Unsupported dimension type"
                assert output_shape.count(dim) >= 1, "Unsupported dimension type"

        # Calculate product of non-symbolic dimensions
        input_size = reduce(lambda x, y: x * y, 
                           [d for d in input_shape if d not in same_dims], 1)
        output_size = reduce(lambda x, y: x * y, 
                            [d for d in output_shape if d not in same_dims], 1)

        # Handle -1 dimension
        if dim_minus_one is not None:
            if is_sympy(input_size) or is_sympy(output_size):
                output_shape[dim_minus_one] = input_size // output_size
                if is_sympy(output_shape[dim_minus_one]) and output_shape[dim_minus_one].is_number:
                    output_shape[dim_minus_one] = int(output_shape[dim_minus_one])
            else:
                assert output_size != 0 and input_size % output_size == 0, \
                    "Reshape input size must be divisible by known output dimensions"
                output_shape[dim_minus_one] = int(input_size // output_size)

        final_output_size = reduce(lambda x, y: x * y,
                                   [d for d in output_shape if d not in same_dims], 1)
        assert is_sympy(input_size) or is_sympy(final_output_size) or input_size == final_output_size, \
            "Reshape input and output element counts must match"

        # Ensure all dimensions are integers
        output_shape = [int(s) if not is_sympy(s) else s for s in output_shape]

        # Create output tensor
        Y = X.clone(shape=tuple(output_shape), scale=X.scale)
        if X.has_data():
            Y.data = X.data.reshape(output_shape)
        self.outputs = [Y]

    def is_inplace(self) -> bool:
        """Check if the operation can be performed in-place"""
        return self.inputs[0].mem_type == self.outputs[0].mem_type


__all__ = ["Reshape"]
