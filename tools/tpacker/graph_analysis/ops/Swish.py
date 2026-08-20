import math
import numpy as np
from typing import List
from ...graph import Tensor
from ...xsympy import is_sympy
from .utils import calc_expr
from ...enum_defines import MemType
from ...enum_defines import ALIGN4
from .base import Operator, OperatorAttrs, register_op


class SwishOperatorAttrs(OperatorAttrs):
    """Attributes for Swish operator."""

    def checkparams(self) -> None:
        """Check if required parameters are present."""
        required_attrs = ["scale_x", "scale_o", "platform", "quant_mode"]
        for attr in required_attrs:
            assert attr in self.attrs, f"Missing required attribute: {attr}"
        quant_mode = self.attrs["quant_mode"]
        if isinstance(quant_mode, bytes):
            quant_mode = quant_mode.decode()
        quant_mode = quant_mode.upper()
        if quant_mode == "LUNA_QUANT":
            quant_mode = "FLOOR_ADD"
        assert quant_mode == "FLOOR_ADD", \
            "Swish runtime only supports floor_add quantization"
        self.attrs["quant_mode"] = quant_mode


@register_op
class QSwish(Operator):
    """Quantized Swish activation function (β=1).

    Swish(x) = x * sigmoid(x), also known as SiLU (Sigmoid-weighted Linear Unit).
    For quantized implementation, this is computed using integer arithmetic.

    Note: This operator only supports venusA platform.

    Attributes:
        o_bits: output bit width (default: 8)
        platform: target platform (only 'venusA' supported)
        quant_mode: quantization mode ('floor', 'floor_add', 'round', 'ceil')
        scale_o: output scale (power of 2)
        scale_x: input scale (power of 2)
        x_bits: input bit width (default: 8)
    """

    def __init__(self, attrs={}):
        """Initialize the Swish operator with given attributes."""
        self.attrs = SwishOperatorAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "Swish operator must have exactly one input"
        platform = self.attrs.get("platform", "venus")

        # Only venusA platform is supported
        assert platform == "venusA", "Swish operator only supports venusA platform"

        X = inputs[0]

        # Check input data type based on x_bits
        x_bits = self.attrs.get("data_bits", self.attrs.get("x_bits", 8))
        assert x_bits in {8, 16, 32}, "input bits must be 8, 16 or 32"

        if x_bits == 8:
            expected_dtype = np.int8
        elif x_bits == 16:
            expected_dtype = np.int16
        else:
            expected_dtype = np.int32

        assert X.dtype == np.dtype(expected_dtype), \
            f"input data type of Swish must be {expected_dtype} for x_bits={x_bits}"

        # Process input scale
        scale_x = self.attrs["scale_x"]
        assert np.isfinite(scale_x) and scale_x > 0, \
            "Swish scale_x must be finite and positive"
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        input_scale = int(temp)
        assert -3 <= input_scale <= 90, \
            "Swish input scale exponent must be in [-3, 90]"
        assert X.scale == input_scale, "Input scale must match attribute scale_x"
        assert X.zero == 0, "Swish only supports zero point 0"

        # Process output scale
        scale_o = self.attrs["scale_o"]
        assert np.isfinite(scale_o) and scale_o > 0, \
            "Swish scale_o must be finite and positive"
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        output_scale = int(temp)
        assert -36 <= output_scale <= 57, \
            "Swish output scale exponent must be in [-36, 57]"

        # Check output bits
        o_bits = self.attrs.get("o_bits", 8)
        assert o_bits == 8, "Swish on venusA runtime only supports int8 output"

        Y = X.clone(scale=output_scale, dtype=np.int8, bits=1, zero=0)
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the Swish operation."""
        X = self.inputs[0]
        data_size = np.prod(X.shape)
        assert not is_sympy(data_size), \
            "Swish workspace requires resolved input dimensions"
        data_size = int(data_size)
        assert 0 < data_size <= np.iinfo(np.uint32).max, \
            "Swish input is too large for the runtime"
        assert X.mem_type == MemType.SHARE_MEM, \
            "Swish on venusA requires SHARE_MEM input"
        assert self.outputs[0].mem_type == MemType.SHARE_MEM, \
            "Swish on venusA requires SHARE_MEM output"

        if X.dtype == np.int8:
            workspace_size = ALIGN4(data_size * 2) + data_size * 4
        elif X.dtype == np.int16:
            workspace_size = data_size * 4
        elif X.dtype == np.int32:
            workspace_size = data_size * 4
        else:
            raise AssertionError("Swish only supports int8, int16, or int32 input")
        return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the Swish operation."""
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

        # Swish = x * sigmoid(x)
        # sigmoid requires ~4 operations, multiply requires 1 operation
        # Estimated as ~6 operations per element
        flops = int(np.prod(yshape)) * 6
        return flops


__all__ = ["QSwish"]
