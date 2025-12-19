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
        X = self.inputs[0]
        input_shape = list(X.shape)
        output_shape = list(self.inputs[1].data)

        # Handle special cases: 0 and -1 in output shape
        dim_minus_one = None
        for i, s in enumerate(output_shape):
            if s == 0:
                output_shape[i] = input_shape[i]
            elif s == -1:
                assert dim_minus_one is None, "Only one -1 is allowed in reshape"
                output_shape[i] = 1
                dim_minus_one = i

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
                output_shape[dim_minus_one] = int(input_size // output_size)

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