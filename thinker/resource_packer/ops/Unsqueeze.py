import numpy as np

from .._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class UnsqueezeAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        self.attrs["axes"] = self.attrs.get("axes", [])
        assert len(self.attrs["axes"]) < 7

    def serialize(self) -> bytes:
        attrs = tffi.new("SqueezeAttrs *")

        attrs.ndim = self.attrs["ndim"]
        attrs.axes = self.attrs["axes"]

        return bytes(tffi.buffer(attrs))


@register_op
class Unsqueeze(Operator):
    def __init__(self, attrs={}):
        self.attrs = UnsqueezeAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1
        X = self.inputs[0]

        tShape = list(X.shape)
        axes = self.attrs["axes"]
        rank = len(tShape) + len(axes)
        for x in axes:
            assert x < rank and x >= -rank
            if x < 0:
                x += rank

        assert len(axes) == len(set(axes))
        axes = sorted(axes)
        for x in axes:
            if x < 0:
                if x == -1:
                    tShape.append(1)
                else:
                    tShape.insert(x + 1, 1)
            else:
                tShape.insert(x, 1)

        yshape = tShape
        Y = X.clone(shape=tuple(yshape))
        if X.has_data():
            if isinstance(X.data, int):
                Y.data = np.array([X.data]).astype(X.dtype.type)
            else:
                Y.data = X.data.reshape(yshape)
        self.outputs = [Y]

    def is_inplace(self):
        return True


__all__ = ["Unsqueeze"]
