import math
import numpy as np
from typing import Any, Dict, Optional

from ...graph import Tensor
from .._type._ctype import tffi
from ...enum_defines import Layout
from ...enum_defines import DevType
from .base import Operator, OperatorAttrs, register_op, BaseLayout


class iqCatAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "scale_o" in self.attrs
        assert "dim" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("iqCatAttrs *")
        attrs.axis = self.attrs["dim"]
        return bytes(tffi.buffer(attrs))


@register_op
class iqCat(Operator, BaseLayout):
    def __init__(self, attrs={}):
        self.attrs = iqCatAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        num_input = len(inputs)

        for i in range(len(inputs)):
            scale_x = self.attrs.get("scale_x_{}".format(i), 1.0)
            temp = math.log(scale_x, 2)
            assert abs(temp - int(temp)) < 0.000001
            assert inputs[i].scale == int(
                temp
            ), "scale in tensor must be same with scale in attribute"

        axis = int(self.attrs["dim"])
        shape = list(inputs[0].shape)
        ndims = len(shape)
        assert axis < ndims and axis >= -ndims
        if axis < 0:
            axis += ndims

        for num in range(1, num_input):
            assert len(inputs[0].shape) == len(inputs[num].shape)
            for i in range(ndims):
                if i != axis:
                    assert inputs[0].shape[i] == inputs[num].shape[i]
                    shape[i] = inputs[0].shape[i]
                else:
                    shape[i] += inputs[num].shape[i]

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001
        Y = Tensor.clone(inputs[0], shape=tuple(shape), scale=int(temp))

        inputdata = [x.data for x in inputs]
        if all([x.has_data() for x in inputs]):
            inputdata = [x.data for x in inputs]
            Y.data = np.concatenate(inputdata, axis=axis)

        self.outputs = [Y]

    def sub_layout_convert(self):
        inputs = self.inputs
        if inputs[0].layout == Layout.NHWC:
            axis = self.attrs["dim"]
            if axis == 1:
                axis = 3
            elif axis == 2:
                axis = 1
            elif axis == 3:
                axis = 2
            self.attrs["dim"] = axis


__all__ = ["iqCat"]
