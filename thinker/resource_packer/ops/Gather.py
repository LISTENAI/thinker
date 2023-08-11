import math
import numpy as np

from .._type._ctype import tffi
from ...enum_defines import DevType
from .base import Operator, OperatorAttrs, register_op


class GatherAttrs(OperatorAttrs):
    def normalize(self):
        self.attrs["axis"] = self.attrs.get("axis", 0)

    def serialize(self) -> bytes:
        attrs = tffi.new("GatherAttrs *")

        attrs.axis = self.attrs["axis"]
        return bytes(tffi.buffer(attrs))


@register_op
class Gather(Operator):
    def __init__(self, attrs={}):
        self.attrs = GatherAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 2

        data = self.inputs[0]
        indices = self.inputs[1]
        tShape = list(data.shape)
        ishape = list(indices.shape)
        axis = self.attrs["axis"]
        ndim = len(tShape)
        assert axis < ndim and axis >= -ndim
        if axis < 0:
            axis += ndim

        yshape = tShape[:axis] + ishape + tShape[axis + 1 :]
        Y = data.clone(shape=yshape, scale=data.scale)
        if indices.has_data() and data.has_data():
            Y.data = np.take(data.data, indices.data.reshape(-1), axis=axis)
            if len(indices.shape) == 0:
                Y.data = np.array(Y.data)


        scale_o = self.attrs.get("scale_o", -1)
        if scale_o < 0:    
            Y.scale = self.inputs[0].scale
        else:
            temp = math.log(scale_o, 2)
            assert abs(temp - int(temp)) < 0.000001
            Y.scale = int(temp)
        self.outputs = [Y]


__all__ = ["Gather"]
