import numpy as np
from typing import Any, Dict, Optional

from ...graph import Tensor
from ...enum_defines import Layout
from .base import Operator, OperatorAttrs, register_op

class iqCatAttrs(OperatorAttrs):
    def __init__(self, attrs:Optional[Dict[str, Any]] = {}) -> None:
        super().__init__(attrs, "iqCatAttrs")

    def checkparams(self) -> None:
        assert "axis" in self.attrs

@register_op
class Concat(Operator):
    def __init__(self, attrs = {}):
        self.attrs = iqCatAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs 
        num_input = len(inputs)
        axis = int(self.attrs["axis"])
        shape = list(inputs[0].shape)
        ndims = len(shape)
        assert axis < ndims and axis >= -ndims
        if axis < 0 :
            axis += ndims

        for num in range(1, num_input):
            assert len(inputs[0].shape) == len(inputs[num].shape)
            for i in range(ndims):
                if i != axis:
                    assert inputs[0].shape[i] == inputs[num].shape[i]
                    shape[i] = inputs[0].shape[i]
                else:
                    shape[i] += inputs[num].shape[i]

        Y = Tensor.clone(inputs[0], shape=tuple(shape),scale=inputs[0].scale)

        inputdata =[x.data for x in inputs]
        if all([x.has_data() for x in inputs]):
            inputdata =[x.data for x in inputs]
            Y.data = np.concatenate(inputdata, axis=axis)
        self.outputs = [Y]

    def sub_layout_convert(self):
        inputs = self.inputs
        if inputs[0].layout == Layout.NHWC:
            axis = self.attrs['axis']
            if axis == 1:
                axis = 3
            elif axis == 2:
                axis = 1
            elif axis == 3:
                axis = 2
            self.attrs['axis'] = axis

__all__ = ['Concat']
