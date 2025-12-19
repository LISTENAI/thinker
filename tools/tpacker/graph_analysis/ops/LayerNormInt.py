import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...xsympy import is_sympy
from .utils import calc_expr, combine4bit_8bit
from ...enum_defines import DevType, Layout, MemType
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class LayerNormIntAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the LayerNormInt operation."""
        attrs = tffi.new("LayerNormIntAttrs *")
        attrs.axis = 1
        attrs.keepdims = 1
        attrs.eps = 0.00001
        return bytes(tffi.buffer(attrs))

@register_op
class LayerNormInt(Operator):
    def __init__(self, attrs={}):
        """Initialize the LayerNormInt operator with given attributes."""
        self.attrs = LayerNormIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        X = self.inputs[0]
        W = self.inputs[1]

        # Check weight compatibility
        assert W.shape[0] in (X.shape[-1] * X.shape[-2], X.shape[-1]), "Layer norm not supported for this weight shape"

        # Process input scale
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Input scale must be a power of 2"

        # Process weight scale
        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w, 2)
        assert abs(temp - int(temp)) < 0.000001, "Weight scale must be a power of 2"
        W.scale = int(temp)

        # Process bias scale if present
        if len(self.inputs) == 3:
            self.inputs[2].scale = X.scale + W.scale

        # Process output scale
        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Output scale must be a power of 2"

        # Create output tensor
        Y = X.clone(scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the LayerNormInt operation."""
        x = self.inputs[0]
        w = self.inputs[1]
        w_size = np.prod(w.shape)
        workspace_size = max((w_size * 2 + 8), w_size * 4)
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def pack_params(self):
        """Pack the parameters for the LayerNormInt operation, handling weight quantization."""
        weight_bits = self.attrs.get("parameter_bits", 8)
        data = self.inputs[1].data
        shape = self.inputs[1].shape
        if weight_bits == 4:
            data = combine4bit_8bit(data)
        self.inputs[1].update(data=data, shape=shape, bits=np.float32(weight_bits / 8))

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the LayerNormInt operation."""
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

__all__ = ["LayerNormInt"]