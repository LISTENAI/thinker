import math
import numpy as np
from ...xsympy import is_sympy
from ...graph import Tensor
from .utils import QuantType, calc_expr
from ...resource_packer._type._ctype import tffi
from ...enum_defines import DevType, MemType, Layout, ALIGN2, ALIGN8, ALIGN16
from .base import Operator, OperatorAttrs, register_op

class GluIntAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the GluInt operation."""
        attrs = tffi.new("GluIntAttrs *")
        attrs.axis = self.attrs["dim"]
        return bytes(tffi.buffer(attrs))

@register_op
class GluInt(Operator):
    def __init__(self, attrs={}):
        """Initialize the GluInt operator with given attributes."""
        self.attrs = GluIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "GluInt operator must have exactly one input"

        X = inputs[0]
        shape = list(X.shape)
        axis = self.attrs['dim']
        shape[axis] = shape[axis] // 2

        # Process scales
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x[0], 2) if isinstance(scale_x, tuple) else math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        if X.scale != -1:
            assert X.scale == int(temp), "Input scale must match attribute scale_x"
        else:
            self.inputs[0].scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o[0], 2) if isinstance(scale_o, tuple) else math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        # Determine output data type
        output_bits = self.attrs.get("o_bits")
        assert output_bits in (8, 16, 32), "Output bits must be 8, 16, or 32"
        dtype = np.int8 if output_bits == 8 else np.int16 if output_bits == 16 else np.int32

        Y = X.clone(shape=tuple(shape), scale=int(temp), dtype=dtype)
        self.outputs = [Y]

    def get_workspace(self):
        """Calculate the required workspace for the GluInt operation."""
        axis = self.attrs['dim']
        M = 1
        for i in range(axis):
            M *= self.inputs[0].shape[i]
        N = self.inputs[0].shape[axis]
        workspace_size = M * N * 7
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

__all__ = ["GluInt"]