import math
import numpy as np
from typing import Any, Dict, Optional

from ...graph import Tensor
from ...enum_defines import MemType
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op, BaseLayout


class ArgMaxAttrs(OperatorAttrs):
    """Attributes handler for ArgMax operator."""
    
    def __init__(self, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(attrs, "ArgMaxAttrs")

    def serialize(self) -> bytes:
        """Serialize attributes to bytes."""
        attrs = tffi.new("ArgMaxAttrs *")
        attrs.axis = self.attrs["axis"]
        return bytes(tffi.buffer(attrs))


@register_op
class ArgMax(Operator, BaseLayout):
    """ArgMax operator implementation."""
    
    def __init__(self, attrs: Dict = {}):
        super().__init__()
        self.attrs = ArgMaxAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer tensor shapes and create output tensor."""
        inputs = self.inputs
        assert len(inputs) == 1, "ArgMax expects exactly one input tensor"

        axis = int(self.attrs["axis"])
        shape = list(inputs[0].shape)
        ndims = len(shape)
        assert -ndims <= axis < ndims, "Axis out of bounds"

        axis = axis + ndims if axis < 0 else axis
        shape[axis] = 1
        shape[0] = 2

        Y = Tensor.clone(inputs[0], shape=tuple(shape), dtype=np.int32, bits=4)

        if all(x.has_data() for x in inputs):
            Y.data = np.concatenate([x.data for x in inputs], axis=axis)

        self.outputs = [Y]

    def get_workspace(self):
        """Calculate and return workspace tensor."""
        workspace_size = 8
        return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]


__all__ = ["ArgMax"]