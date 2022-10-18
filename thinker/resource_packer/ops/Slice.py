from .utils.utils import *
from .._type._ctype import tffi
from ...enum_defines import Layout
from ...enum_defines import DevType
from .base import Operator, OperatorAttrs, register_op


class SliceAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        attrs = tffi.new("SliceAttrs *")

        attrs.axis = self.attrs["axis"]
        attrs.dims = self.attrs["dims"]
        attrs.split = self.attrs["split"]

        return bytes(tffi.buffer(attrs))


@register_op
class Slice(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) >= 3
        X = inputs[0]
        starts = inputs[1].data[0]
        ends = inputs[2].data[0]
        if len(inputs) >= 4:
            axes = inputs[3].data[0]
        else:
            axes = 0
        if len(inputs) == 5:
            steps = inputs[4].data[0]
        else:
            steps = 1
        shape = list(X.shape)

        assert axes < len(shape)
        axes = axes + len(shape) if axes < 0 else axes

        if starts < 0:
            starts += shape[axes]

        if ends > 65536:
            ends = shape[axes]
        elif ends < 0:

            ends += shape[axes]

        if steps < 0:
            shape[axes] = (starts - ends + 1 + steps) // (-1 * steps)
        else:
            shape[axes] = (ends - starts + steps - 1) // steps

        Y = X.clone(shape=tuple(shape))
        self.outputs = [Y]

    def sub_layout_convert(self):
        inputs = self.inputs
        if inputs[0].layout == Layout.NHWC:
            axes = inputs[3].data[0]
            if axes == 1:
                axes = 3
            elif axes == 2:
                axes = 1
            elif axes == 3:
                axes = 2
            inputs[3].data[0] = axes


__all__ = ["Slice"]
