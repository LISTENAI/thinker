import numpy as np

from .._type import tffi
from .base import Operator, OperatorAttrs, BaseLayout, register_op


class ClipAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        attrs = tffi.new("ClipAttrs")

        assert "min" in self.attrs.attrs
        assert "max" in self.attrs.attrs
        attrs.min = self.attrs["min"]
        attrs.max = self.attrs["max"]

        return bytes(tffi.buffer(attrs))


@register_op
class Clip(Operator, BaseLayout):
    def __init__(self, attrs={}):
        self.attrs = ClipAttrs(attrs)

    def infer_tensor(self):
        assert len(self.inputs) == 1 or len(self.inputs) == 3
        X = self.inputs[0]
        Y = X.clone()
        if all([x.has_data() for x in self.inputs]):
            min = self.inputs[1].data
            max = self.inputs[2].data

            Y.data = np.clip(self.inputs[0].data, min, max)
        self.outputs = [Y]


__all__ = ["Clip"]
