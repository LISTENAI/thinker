from .._type._ctype import tffi

from .base import Operator, OperatorAttrs, register_op
from ...enum_defines import DevType


class SqueezeAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        attrs = tffi.new("SqueezeAttrs *")
        attrs.axes = self.attrs.get("axes", [])
        attrs.ndim = len(attrs.axes)
        return bytes(tffi.buffer(attrs))


@register_op
class Squeeze(Operator):
    def __init__(self, attrs={}):
        self.attrs = SqueezeAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1
        X = self.inputs[0]
        tShape = list(X.shape)
        axes = self.attrs.get("axes", [])
        if len(axes) == 0:
            yshape = []
            for i, s in enumerate(tShape):
                if s != 1:
                    yshape.append(s)
                else:
                    axes.append(i)
                if len(yshape) == 1:
                    yshape = [1, yshape[0]]                    
        else:
            for x in axes:
                assert x < len(tShape) and x >= -len(tShape)
                assert tShape[x] == 1, "invalid config for squeeze in axis:{}".format(x)
                tShape[x] = None
            yshape = [x for x in tShape if x is not None]
            if len(yshape) == 1:
                yshape = [1, yshape[0]]
                
        Y = X.clone(shape=tuple(yshape))
        self.outputs = [Y]

    def is_inplace(self):
        return True


__all__ = ["Squeeze"]
