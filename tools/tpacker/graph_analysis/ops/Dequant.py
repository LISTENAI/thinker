import math
import numpy as np
from typing import Any, Dict, Optional
from ...graph import Tensor
from ...xsympy import is_sympy
from ...enum_defines import DevType
from .utils import QuantType, calc_expr
from .base import UnaryOperator, OperatorAttrs, register_op

@register_op
class Dequant(UnaryOperator):
    def checkparams(self) -> None:
        """Check if the required parameters are present."""
        assert "scale_o" in self.attrs, "Missing required attribute: scale_o"

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor based on the input tensor and attributes."""
        inputs = self.inputs
        assert len(inputs) in (1, 2), "Number of inputs must be 1 or 2"

        X = inputs[0]

        # Calculate and validate scale
        scale_x = self.attrs["scale_o"]
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert X.scale == int(temp), "Input scale does not match expected value"

        # Create output tensor
        Y = X.clone(dtype=np.float32, bits=4, scale=1.0)
        self.outputs = [Y]

        # Perform forward computation if all inputs have data
        if all(x.has_data() for x in inputs):
            self.forward()

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs)."""
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

        # Calculate FLOPs based on output dimensions
        output_dims = yshape[1:]
        return int(np.prod(output_dims))

__all__ = ["Dequant"]