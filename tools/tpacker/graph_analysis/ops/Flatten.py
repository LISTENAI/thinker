from functools import reduce

from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class FlattenAttrs(OperatorAttrs):
    def normalize(self):
        self.attrs.setdefault("axis", 1)

    def serialize(self) -> bytes:
        attrs = tffi.new("FlattenAttrs *")
        attrs.axis = self.attrs["axis"]
        return bytes(tffi.buffer(attrs))

@register_op
class Flatten(Operator):
    def __init__(self, attrs={}):
        self.attrs = FlattenAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor by flattening the input tensor."""
        inputs = self.inputs
        assert len(inputs) == 1, "Flatten operator must have exactly one input"

        X = inputs[0]
        shape = list(X.shape)
        rank = len(shape)
        axis = self.attrs["axis"]
        assert -rank <= axis <= rank, "Flatten axis out of bounds"
        axis = axis + rank if axis < 0 else axis
        self.attrs["axis"] = axis
        first = reduce(lambda x, y: x * y, shape[:axis], 1)
        second = reduce(lambda x, y: x * y, shape[axis:], 1)
        Y = X.clone(shape=(first, second))
        if X.has_data():
            Y.data = X.data.reshape((first, second))
        self.outputs = [Y]

__all__ = ["Flatten"]
