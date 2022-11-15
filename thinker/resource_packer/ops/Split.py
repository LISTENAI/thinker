from ...graph import *
from .._type._ctype import tffi
from ...enum_defines import Layout
from .base import Operator, OperatorAttrs, register_op


class SplitAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "axis" in self.attrs
        assert "dims" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("SplitAttrs *")

        attrs.axis = self.attrs["axis"]
        attrs.dims = self.attrs["dims"]
        attrs.split = self.attrs["split"]

        return bytes(tffi.buffer(attrs))


@register_op
class Split(Operator):
    def __init__(self, attrs={}):
        self.attrs = SplitAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1
        X = inputs[0]
        axis = self.attrs["axis"]
        assert axis < len(X.shape)

        if self.attrs.get("split", None) == None:
            dims = self.attrs["dims"]
            self.attrs["split"] = []
            for i in range(dims):
                self.attrs["split"].append(X.shape[axis] // dims)

        split = self.attrs["split"]
        assert sum(split) == X.shape[axis]

        outputs = []
        for idx in split:
            shape = list(X.shape)
            shape[axis] = int(idx)
            yi = X.clone(shape=shape)
            outputs.append(yi)
        self.outputs = outputs

    def sub_layout_convert(self):
        inputs = self.inputs
        if inputs[0].layout == Layout.NHWC:
            axis = self.attrs["axis"]
            if axis == 1:
                axis = 3
            elif axis == 2:
                axis = 1
            elif axis == 3:
                axis = 2
            self.attrs["axis"] = axis


__all__ = ["Split"]
