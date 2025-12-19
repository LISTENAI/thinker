import math
import numpy as np

from ...xsympy import is_sympy
from ...resource_packer._type._ctype import tffi
from .utils import QuantType, calc_expr
from .base import Operator, OperatorAttrs, register_op


class QuantAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check required parameters"""
        assert "scale_x" in self.attrs and "data_bits" in self.attrs
        assert self.attrs['data_bits'] in (8, 16, 32), "Data bits must be 8, 16 or 32"

    def serialize(self) -> bytes:
        """Serialize attributes to bytes"""
        attrs = tffi.new("QuantAttrs *")
        attrs.data_bits = self.attrs["data_bits"]
        quant_type = QuantType.from_str(self.attrs.get("platform_quant", "qmax_quant"))
        attrs.quant_type = quant_type.value
        return bytes(tffi.buffer(attrs))


@register_op
class Quant(Operator):
    def __init__(self, attrs={}):
        self.attrs = QuantAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor based on input and quantization parameters"""
        attrs = self.attrs
        X = self.inputs[0]
        data_bits = int(attrs.get("data_bits", 8))
        
        # Verify input type and bit depth
        assert (X.dtype == np.float32 and data_bits in (8, 16, 32)), "Quant only supports float32 to int8 conversion"

        # Calculate scale
        scale_o = attrs["scale_x"]
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 1e-6, "Scale must be power of 2"

        # Create output tensor based on bit depth
        dtype_map = {8: np.dtype("i1"), 16: np.dtype("i2"), 32: np.dtype("i4")}
        bits_map = {8: 1, 16: 2, 32: 4}
        Y = X.clone(dtype=dtype_map[data_bits], 
                    bits=bits_map[data_bits], 
                    scale=int(temp), zero=0)
        self.outputs = [Y]

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate floating point operations per second"""
        X = self.inputs[0]
        Y = self.outputs[0]
        
        # Process input shape
        input_shape = list(X.shape)
        for i, s in enumerate(input_shape):
            if is_sympy(s):
                input_shape[i] = calc_expr(str(s), dynamic_shape)

        # Process output shape
        output_shape = list(Y.shape)
        for i, s in enumerate(output_shape):
            if is_sympy(s):
                output_shape[i] = calc_expr(str(s), dynamic_shape)

        return int(np.prod(output_shape))


__all__ = ["Quant"]