import math
import numpy as np
from typing import List

from ...xsympy import is_sympy
from ...enum_defines import DevType, MemType
from ...graph import Tensor
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
        assert platform in {"venus", "arcs", "mars", "venusA"}, \
            f"Unsupported Requant platform: {platform}"
        if "quant_mode" in self.attrs:
            quant_mode = self.attrs.get("quant_mode")
        elif "platform_quant" in self.attrs:
            quant_mode = self.attrs.get("platform_quant")
        else:
            quant_mode = "FLOOR_ADD"
        if quant_mode == "luna_quant":
            quant_mode = "FLOOR_ADD"
        assert RoundMethod.from_str(quant_mode) == RoundMethod.FLOOR_ADD, \
            "Requant runtime only supports FLOOR_ADD quantization"
        self.attrs['quant_mode'] = quant_mode

    def serialize(self) -> bytes:
        """Serialize Requant attributes to bytes"""
        attrs = tffi.new("RequantAttrs *")
        attrs.o_bits = self.attrs["o_bits"]
        attrs.data_bits = self.attrs["data_bits"]
        attrs.quant_type = RoundMethod.from_str(self.attrs["quant_mode"]).value
        return bytes(tffi.buffer(attrs))


@register_op
class Requant(Operator):
    def __init__(self, attrs={}):
        self.attrs = RequantAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor based on input and quantization parameters"""
        assert len(self.inputs) == 1, "Requant operator must have exactly one input"
        X = self.inputs[0]
        platform = self.attrs.get("platform", "venus")
        assert X.dtype == np.int8, f"Requant on {platform} runtime only supports int8 input"
        assert X.zero == 0, "Requant only supports symmetric input quantization"
        assert self.attrs["data_bits"] == 8, f"Requant on {platform} input bit metadata must be 8"
        if platform == "arcs":
            assert self.attrs["o_bits"] in (8, 32), "Requant on arcs only supports int8 or int32 output"
        scale_x = self.attrs.get("scale_x")
        scale_o = self.attrs.get("scale_o")
        
        # Verify scale values are powers of 2
        assert scale_x > 0 and scale_o > 0, "Requant scales must be positive"
        temp1 = math.log(scale_x, 2)
        assert abs(temp1 - int(temp1)) < 1e-6, "Scale_x must be a power of 2"
        assert X.scale == int(temp1), "Input scale mismatch"
        
        temp2 = math.log(scale_o, 2)
        assert abs(temp2 - int(temp2)) < 1e-6, "Scale_o must be a power of 2"
        q_delta = int(temp2) - int(temp1)
        if platform == "venus":
            max_shift = {8: 6, 16: 14, 32: 30}[self.attrs["o_bits"]]
            if self.attrs["o_bits"] > self.attrs["data_bits"]:
                assert 0 <= q_delta <= max_shift, "Requant widening on venus exceeds scalar shift range"
            else:
                assert -63 <= q_delta <= max_shift, "Requant on venus exceeds scalar shift range"
        elif platform in {"arcs", "mars"}:
            if self.attrs["o_bits"] > self.attrs["data_bits"]:
                assert self.attrs["o_bits"] == 32 and 0 <= q_delta <= 24, "Requant widening on arcs requires int32 output and a safe left shift"
            elif self.attrs["o_bits"] == self.attrs["data_bits"]:
                assert -63 <= q_delta <= 30, "Requant on arcs shift is unsupported"
            else:
                raise AssertionError("Requant narrowing is unsupported on arcs")
        elif platform == "venusA":
            assert -63 <= q_delta <= 30, "Requant on venusA exceeds scalar shift range"
            if self.attrs["o_bits"] > self.attrs["data_bits"]:
                assert q_delta >= 0, "Requant widening on venusA does not support right shift"

        # Determine output data type and bit depth
        bits_map = {8: (np.dtype("i1"), 1), 16: (np.dtype("i2"), 2), 32: (np.dtype("i4"), 4)}
        data_type, bits = bits_map[self.attrs["o_bits"]]
        
        Y = X.clone(dtype=data_type, bits=bits, scale=int(temp2))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        if self.attrs["o_bits"] == self.attrs["data_bits"]:
            assert self.inputs[0].mem_type == MemType.SHARE_MEM and \
                self.outputs[0].mem_type == MemType.SHARE_MEM, \
                "Same-width Requant requires SHARE_MEM input and output"
        return []

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
