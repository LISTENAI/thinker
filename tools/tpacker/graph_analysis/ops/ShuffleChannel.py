from ...graph import Tensor
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class ShuffleChannelAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check and set default parameters for ShuffleChannel"""
        self.attrs["axis"] = self.attrs.get("axis", 1)  # Set default axis to 1 if not provided
        assert "groups" in self.attrs, "groups must be specified"

    def serialize(self) -> bytes:
        """Serialize ShuffleChannel attributes to bytes"""
        attrs = tffi.new("ShuffleChannelAttrs *")
        attrs.num_group = self.attrs["groups"]
        attrs.axis = self.attrs["axis"]
        return bytes(tffi.buffer(attrs))

@register_op
class ShuffleChannel(Operator):
    def __init__(self, attrs={}):
        self.attrs = ShuffleChannelAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape based on input tensor"""
        inputs = self.inputs
        assert len(inputs) == 1, "ShuffleChannel expects exactly one input"
        X = inputs[0]
        assert len(X.shape) == 4, "Input tensor must be 4-dimensional (NCHW)"
        
        # Create output tensor with same properties as input
        Y = Tensor.clone(X, scale=X.scale)
        self.outputs = [Y]

__all__ = ["ShuffleChannel"]