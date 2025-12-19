import numpy as np

from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, BaseLayout, register_op


class ClipAttrs(OperatorAttrs):
    """Attributes handler for Clip operator."""
    
    def serialize(self) -> bytes:
        """Serialize attributes to bytes."""
        attrs = tffi.new("ClipAttrs")
        assert "min" in self.attrs, "Minimum value (min) not found in attributes"
        assert "max" in self.attrs, "Maximum value (max) not found in attributes"
        attrs.min = self.attrs["min"]
        attrs.max = self.attrs["max"]
        return bytes(tffi.buffer(attrs))


@register_op
class Clip(Operator, BaseLayout):
    """Clip operator to limit tensor values within specified bounds."""
    
    def __init__(self, attrs: dict = {}):
        super().__init__()
        self.attrs = ClipAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor with clipped values."""
        assert len(self.inputs) in {1, 3}, "Clip operator expects 1 or 3 input tensors"
        X = self.inputs[0]
        assert X.dtype in {np.int8, np.int16, np.int32}, "Input must be int8, int16, or int32"

        Y = X.clone()

        if all(tensor.has_data() for tensor in self.inputs):
            min_val = self.inputs[1].data
            max_val = self.inputs[2].data
            Y.data = np.clip(X.data, min_val, max_val)

        self.outputs = [Y]


__all__ = ["Clip"]