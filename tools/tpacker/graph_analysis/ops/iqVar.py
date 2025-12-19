import math
import numpy as np
from typing import List

from ...graph import Tensor
from .utils import calc_expr
from ...xsympy import is_sympy
from ...enum_defines import DevType, MemType
from .base import Operator, register_op

@register_op
class iqVar(Operator):
    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "iqVar operator must have exactly one input"

        X = inputs[0]
        x_shape = list(X.shape)
        x_shape[2] = 1  # Set the third dimension to 1

        # Process input scale
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert X.scale == int(temp), "Input scale must match attribute scale_x"

        # Process output scale
        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        # Create output tensor
        Y = X.clone(shape=tuple(x_shape), scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqVar operation."""
        input_data = self.inputs[0]
        workspace_size = input_data.nbytes * input_data.dtype.itemsize / input_data.shape[2] * 3
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int32, MemType.SHARE_MEM)]
        return []

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the iqVar operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        xshape = list(X.shape)
        yshape = list(Y.shape)

        # Resolve symbolic expressions in shapes
        for i, s in enumerate(xshape):
            if is_sympy(s):
                xshape[i] = calc_expr(str(s), dynamic_shape)
        for i, s in enumerate(yshape):
            if is_sympy(s):
                yshape[i] = calc_expr(str(s), dynamic_shape)

        # Calculate FLOPs
        flops = int(np.prod(xshape)) + int(np.prod(yshape)) * 6
        return flops

__all__ = ["iqVar"]