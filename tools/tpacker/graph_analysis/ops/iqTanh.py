import math
import numpy as np
from ...xsympy import is_sympy
from .utils import QuantType, calc_expr
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class iqTanhOperatorAttrs(OperatorAttrs):
    def __init__(self, attrs={}):
        """Initialize the iqTanh operator attributes."""
        super().__init__(attrs, "iqBinaryAttrs")

    def checkparams(self) -> None:
        """Check if required parameters are present."""
        required_attrs = ["scale_x", "scale_o", "platform_quant"]
        for attr in required_attrs:
            assert attr in self.attrs, f"Missing required attribute: {attr}"
        QuantType.from_str(self.attrs["platform_quant"])

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
        platform = self.attrs.get("platform")
        assert platform is None or platform in {"venus", "arcs", "venusA"}, \
            "iqTanh platform must be venus, arcs, or venusA"
        assert X.dtype in (np.int16, np.int32), \
            "iqTanh expects int16 or int32 input"
        assert X.zero == 0, "iqTanh only supports zero point 0"

        # Process input scale
        scale_x = self.attrs.get("scale_x")
        assert scale_x > 0, "iqTanh scale_x must be positive"
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert X.scale == int(temp), "Input scale must match attribute scale_x"
        if platform == "venus":
            assert X.dtype == np.int16 and int(temp) == 11, \
                "iqTanh on venus expects int16 Q11 input"
        elif platform in {"arcs", "venusA"}:
            assert X.dtype == np.int32, f"iqTanh on {platform} expects int32 Q4.27 input"
            assert int(temp) == 27, f"iqTanh on {platform} expects Q4.27 input"

        # Process output scale
        scale_o = self.attrs.get("scale_o")
        assert scale_o > 0, "iqTanh scale_o must be positive"
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert int(temp) == 7, "iqTanh int8 output scale must be Q7"
        out_bits = self.attrs.get("o_bits")
        assert out_bits is None or out_bits == 8, "iqTanh output type must be int8"

        # Create output tensor
        Y = X.clone(scale=int(temp), dtype=np.int8, bits=1)
        self.outputs = [Y]

        # Perform forward computation if all inputs have data
        if all(x.has_data() for x in inputs):
            self.forward()

    def get_workspace(self):
        platform = self.attrs.get("platform", "venus")
        assert platform in {"venus", "arcs", "venusA"}, \
            "iqTanh platform must be venus, arcs, or venusA"
        X = self.inputs[0]
        Y = self.outputs[0]
        if platform == "venus":
            assert X.dtype == np.int16 and X.scale == 11, \
                "iqTanh on venus expects int16 Q11 input"
        else:
            assert X.dtype == np.int32 and X.scale == 27, \
                f"iqTanh on {platform} expects int32 Q27 input"
        assert Y.dtype == np.int8 and Y.scale == 7 and Y.zero == 0
        from ...enum_defines import MemType
        assert X.mem_type == MemType.SHARE_MEM, f"iqTanh on {platform} requires SHARE_MEM input"
        assert Y.mem_type == MemType.SHARE_MEM, f"iqTanh on {platform} requires SHARE_MEM output"
        return []

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
