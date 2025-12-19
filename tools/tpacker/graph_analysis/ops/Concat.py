import numpy as np
from typing import Any, Dict, Optional

from ...xsympy import is_sympy
from ...graph import Tensor
from ...enum_defines import Layout
from .utils import calc_expr
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class ConcatAttrs(OperatorAttrs):
    """Attributes handler for Concat operator."""
    
    def serialize(self) -> bytes:
        """Serialize attributes to bytes."""
        attrs = tffi.new("iqCatAttrs *")
        attrs.axis = self.attrs["axis"]
        return bytes(tffi.buffer(attrs))


@register_op
class Concat(Operator):
    """Concatenate tensors along a specified axis."""
    
    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape and data."""
        inputs = self.inputs
        num_inputs = len(inputs)
        assert num_inputs > 1, "Concat requires at least two input tensors"

        axis = int(self.attrs["axis"])
        shape = list(inputs[0].shape)
        ndims = len(shape)
        assert -ndims <= axis < ndims, "Axis out of bounds"
        axis = axis + ndims if axis < 0 else axis

        for i in range(1, num_inputs):
            current_input = inputs[i]
            assert len(current_input.shape) == ndims, "All inputs must have the same number of dimensions"
            
            for dim in range(ndims):
                if dim != axis:
                    if is_sympy(inputs[0].shape[dim]) and is_sympy(current_input.shape[dim]):
                        assert calc_expr(str(inputs[0].shape[dim]), dynamic_shape) == calc_expr(str(current_input.shape[dim]), dynamic_shape), "Dimension mismatch"
                    else:
                        assert inputs[0].shape[dim] == current_input.shape[dim], "Dimension mismatch"
                else:
                    shape[dim] += current_input.shape[dim]

        Y = Tensor.clone(inputs[0], shape=tuple(shape))

        if all(tensor.has_data() for tensor in inputs):
            Y.data = np.concatenate([tensor.data for tensor in inputs], axis=axis)

        self.outputs = [Y]

    def sub_layout_convert(self):
        """Adjust axis for NHWC layout."""
        if self.inputs[0].layout == Layout.NHWC:
            axis = self.attrs['axis']
            if axis == 1:
                axis = 3
            elif axis == 2:
                axis = 1
            elif axis == 3:
                axis = 2
            self.attrs['axis'] = axis


__all__ = ['Concat']