import math
import numpy as np
from typing import Any, Dict, Optional
from ...graph import Tensor
from ...xsympy import is_sympy
from ...enum_defines import DevType
from ...resource_packer._type._ctype import tffi
from .utils import QuantType, calc_expr
from .base import UnaryOperator, OperatorAttrs, register_op

class DequantAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "scale_o" in self.attrs, "Missing required attribute: scale_o"
        scale_o = self.attrs["scale_o"]
        assert isinstance(scale_o, (int, float)) and math.isfinite(scale_o) and scale_o > 0
        exponent = math.log2(scale_o)
        rounded_exponent = round(exponent)
        assert abs(exponent - rounded_exponent) < 1e-6, "Scale must be a power of 2"
        assert 0 <= rounded_exponent <= 30, "Scale exponent must be in [0, 30]"

    def serialize(self) -> bytes:
        attrs = tffi.new("DequantAttrs *")
        attrs.scale_o = round(math.log2(self.attrs["scale_o"]))
        return bytes(tffi.buffer(attrs))

@register_op
class Dequant(UnaryOperator):
    def __init__(self, attrs=None):
        self.attrs = DequantAttrs(attrs or {})

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor based on the input tensor and attributes."""
        inputs = self.inputs
        assert len(inputs) == 1, "Dequant requires exactly one input tensor"

        X = inputs[0]

        # Calculate and validate scale
        scale_o = self.attrs["scale_o"]
        temp = round(math.log2(scale_o))
        assert X.dtype in (np.dtype(np.int8), np.dtype(np.uint8), np.dtype(np.int32)), \
            "Dequant only supports int8, uint8, or int32 input"
        assert X.scale == temp, "Input scale does not match expected value"
        assert X.zero == 0, "Dequant only supports zero point 0"

        # Create output tensor
        Y = X.clone(dtype=np.float32, bits=4, scale=1.0, zero=0)
        self.outputs = [Y]

        # Perform forward computation if all inputs have data
        if X.has_data():
            self.forward()

    def forward(self):
        X = self.inputs[0]
        self.outputs[0].data = X.data.astype(np.float32) / float(2 ** X.scale)

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
