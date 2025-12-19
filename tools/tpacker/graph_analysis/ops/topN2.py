import math
import numpy as np
from typing import Any, Dict, Optional

from ...graph import Tensor
from ...enum_defines import DevType, Layout, MemType
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op, BaseLayout

class topN2Attrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}) -> None:
        """Initialize topN2 attributes with given parameters."""
        super().__init__(attrs, "topNAttrs")

    def serialize(self) -> bytes:
        """Serialize topN2 attributes to bytes."""
        attrs = tffi.new("topNAttrs *")
        attrs.dim = self.attrs["dim"]
        attrs.max_num = self.attrs["max_num"]
        return bytes(tffi.buffer(attrs))

@register_op
class topN2(Operator, BaseLayout):
    def __init__(self, attrs={}):
        """Initialize topN2 operator with given attributes."""
        self.attrs = topN2Attrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape based on input tensor and topN2 parameters."""
        inputs = self.inputs
        assert len(inputs) == 1, "topN2 operation requires exactly one input"

        axis = int(self.attrs["dim"])
        N = int(self.attrs["max_num"])
        shape = list(inputs[0].shape)
        ndims = len(shape)
        assert axis < ndims and axis >= -ndims, "axis out of bounds"
        axis = axis + ndims if axis < 0 else axis

        shape[axis] = N
        shape[0] = 2

        # Handle scale_o
        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "scale_o must be a power of 2"

        # Create output tensor based on platform
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            Y = Tensor.clone(inputs[0], shape=tuple(shape), scale=int(temp), dtype=np.int16, bits=2)
        else:
            Y = Tensor.clone(inputs[0], shape=tuple(shape), scale=int(temp), dtype=np.int32, bits=4)

        self.outputs = [Y]

    def get_workspace(self):
        """Calculate the required workspace size for the operation."""
        max_num = int(self.attrs["max_num"])
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            workspace_size = max_num * 8
        else:
            workspace_size = max_num * 16
        max_workspace = Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)
        return [max_workspace]

__all__ = ["topN2"]