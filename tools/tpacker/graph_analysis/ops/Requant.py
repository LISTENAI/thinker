import math
import numpy as np

from ...xsympy import is_sympy
from ...enum_defines import DevType
from .utils import QuantType, RoundMethod, calc_expr
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class RequantAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check required parameters for Requant operation"""
        assert "scale_x" in self.attrs and "scale_o" in self.attrs, "Missing scale parameters"
        assert "data_bits" in self.attrs and "o_bits" in self.attrs, "Missing bit depth parameters"
        assert self.attrs['data_bits'] in (8, 16, 32), "Data bits must be 8, 16, or 32"
        assert self.attrs['o_bits'] in (8, 16, 32), "Output bits must be 8, 16, or 32"
        platform = self.attrs.get("platform", "venus")
        quant_type = (
            RoundMethod.from_str(self.attrs.get("quant_mode"))
            if platform in ["arcs", "venusA"]
            else QuantType.from_str(self.attrs.get("platform_quant", "LUNA_QUANT"))
        )
        self.attrs["quant_type"] = quant_type
    def serialize(self) -> bytes:
        """Serialize Requant attributes to bytes"""
        attrs = tffi.new("RequantAttrs *")
        attrs.o_bits = self.attrs["o_bits"]
        attrs.data_bits = self.attrs["data_bits"]
        attrs.quant_type = self.attrs["quant_type"].value
        return bytes(tffi.buffer(attrs))


@register_op
class Requant(Operator):
    def __init__(self, attrs={}):
        self.attrs = RequantAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor based on input and quantization parameters"""
        X = self.inputs[0]
        scale_x = self.attrs.get("scale_x")
        scale_o = self.attrs.get("scale_o")
        
        # Verify scale values are powers of 2
        temp1 = math.log(scale_x, 2)
        assert temp1 == int(temp1), "Scale_x must be a power of 2"
        assert X.scale == int(temp1), "Input scale mismatch"
        
        temp2 = math.log(scale_o, 2)
        assert temp2 == int(temp2), "Scale_o must be a power of 2"

        # Determine output data type and bit depth
        bits_map = {8: (np.dtype("i1"), 1), 16: (np.dtype("i2"), 2), 32: (np.dtype("i4"), 4)}
        data_type, bits = bits_map[self.attrs["o_bits"]]
        
        Y = X.clone(dtype=data_type, bits=bits, scale=int(temp2))
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


__all__ = ["Requant"]