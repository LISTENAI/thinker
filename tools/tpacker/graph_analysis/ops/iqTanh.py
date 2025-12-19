import math
import numpy as np
from ...xsympy import is_sympy
from .utils import QuantType, calc_expr
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class iqTanhOperatorAttrs(OperatorAttrs):
    def __init__(self, attrs={}):
        """Initialize the iqTanh operator attributes."""
        self.attrs = attrs

    def checkparams(self) -> None:
        """Check if required parameters are present."""
        required_attrs = ["scale_x", "scale_o", "platform_quant"]
        for attr in required_attrs:
            assert attr in self.attrs, f"Missing required attribute: {attr}"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the iqTanh operation."""
        attrs = tffi.new("iqBinaryAttrs *")
        quant_type = self.attrs["platform_quant"]
        attrs.quant_type = QuantType.from_str(quant_type).value
        return bytes(tffi.buffer(attrs))

@register_op
class iqTanh(Operator):
    def __init__(self, attrs={}):
        """Initialize the iqTanh operator with given attributes."""
        self.attrs = iqTanhOperatorAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "iqTanh operator must have exactly one input"

        X = inputs[0]
        assert X.dtype == np.int16, "Input must be of type int16"

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
        Y = X.clone(scale=int(temp), dtype=np.int8, bits=1)
        self.outputs = [Y]

        # Perform forward computation if all inputs have data
        if all(x.has_data() for x in inputs):
            self.forward()

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the iqTanh operation."""
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
        flops = int(np.prod(yshape)) * 4
        return flops

__all__ = ["iqTanh"]