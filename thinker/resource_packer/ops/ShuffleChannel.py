from ...graph import Tensor
from .._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class ShuffleChannelAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        self.attrs["axis"] = self.attrs.get("axis", 1)
        assert "groups" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("ShuffleChannelAttrs *")

        attrs.num_group = self.attrs["groups"]
        attrs.axis = self.attrs["axis"]

        return bytes(tffi.buffer(attrs))


@register_op
class ShuffleChannel(Operator):
    def __init__(self, attrs={}):
        self.attrs = ShuffleChannelAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1
        X = inputs[0]
        assert len(X.shape) == 4
        Y = Tensor.clone(self.inputs[0])
        self.outputs = [Y]


__all__ = ["ShuffleChannel"]
