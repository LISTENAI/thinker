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

    def checkparams(self) -> None:
        axis = int(self.attrs.get("axis", 0))
        assert -128 <= axis <= 127, "ArgMax axis cannot be serialized as int8"
        assert int(self.attrs.get("select_last_index", 0)) == 0, \
            "ArgMax runtime does not support select_last_index=1"
        self.attrs["axis"] = axis

    def serialize(self) -> bytes:
        """Serialize attributes to bytes."""
        attrs = tffi.new("ArgMaxAttrs *")
        attrs.axis = self.attrs["axis"]
        return bytes(tffi.buffer(attrs))


@register_op
class ArgMax(Operator, BaseLayout):
    """ArgMax operator implementation."""
    
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.attrs = ArgMaxAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer tensor shapes and create output tensor."""
        inputs = self.inputs
        assert len(inputs) == 1, "ArgMax expects exactly one input tensor"

        axis = int(self.attrs["axis"])
        shape = list(inputs[0].shape)
        ndims = len(shape)
        assert ndims >= 1, "ArgMax input must have at least one dimension"
        assert -ndims <= axis < ndims, "Axis out of bounds"

        axis = axis + ndims if axis < 0 else axis
        platform = self.attrs.get("platform", "venus")
        assert platform in ("venus", "arcs", "mars", "venusA", "venusa"), \
            f"Unsupported ArgMax platform: {platform}"
        supported_dtypes = {
            "venus": (np.int8, np.float32),
            "arcs": (np.int8, np.int32),
            "mars": (np.int8, np.int32),
            "venusA": (np.int8, np.int16, np.int32, np.float32),
            "venusa": (np.int8, np.int16, np.int32, np.float32),
        }
        assert inputs[0].dtype in supported_dtypes[platform], \
            f"ArgMax input dtype is not supported on {platform}"
        assert ndims == 1 or shape[0] == 1, \
            f"ArgMax on {platform} requires shape[0] == 1 for batched input"
        assert axis == ndims - 1, "ArgMax only supports the last axis"
        assert shape[axis] > 0, "ArgMax reduction dimension must be positive"
        shape[axis] = 1
        shape[0] = 2

        Y = Tensor.clone(inputs[0], shape=tuple(shape), dtype=np.int32, bits=4)

        self.outputs = [Y]

    def get_workspace(self):
        """Calculate and return workspace tensor."""
        workspace_size = 8
        return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]


__all__ = ["ArgMax"]
