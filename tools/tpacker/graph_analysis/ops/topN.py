import math
import numpy as np
from typing import Any, Dict, Optional

from ...graph import Tensor
from ...resource_packer._type._ctype import tffi
from ...enum_defines import DevType, Layout, MemType
from .base import Operator, OperatorAttrs, register_op, BaseLayout

class topNAttrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}) -> None:
        """Initialize topN attributes with given parameters."""
        super().__init__(attrs, "topNAttrs")

    def serialize(self) -> bytes:
        """Serialize topN attributes to bytes."""
        attrs = tffi.new("topNAttrs *")
        attrs.dim = self.attrs["dim"]
        attrs.max_num = self.attrs["max_num"]
        return bytes(tffi.buffer(attrs))

@register_op
class topN(Operator, BaseLayout):
    def __init__(self, attrs={}):
        """Initialize topN operator with given attributes."""
        self.attrs = topNAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape based on input tensor and topN parameters."""
        inputs = self.inputs
        assert len(inputs) == 2, "topN operation requires exactly two inputs"

        axis = int(self.attrs["dim"])
        N = int(self.attrs["max_num"])
        shape = list(inputs[0].shape)
        ndims = len(shape)
        assert axis < ndims and axis >= -ndims, "axis out of bounds"
        if axis < 0:
            axis += ndims

        assert shape[0] == 1, "Only shape[0] == 1 is supported"
        shape[axis] = N
        shape[0] = 2

        platform = self.attrs.get("platform", "venus")
        if platform in ["arcs", "venusA"]:
            Y = Tensor.clone(inputs[0], shape=tuple(shape), dtype=np.int32, bits=4)
        else:
            Y = Tensor.clone(inputs[0], shape=tuple(shape), dtype=np.int16, bits=2)

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

__all__ = ["topN"]