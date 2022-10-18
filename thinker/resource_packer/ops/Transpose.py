from typing import Any, Dict, List, Optional

from .._type._ctype import tffi
from ...enum_defines import DevType, Layout
from .base import Operator, OperatorAttrs, register_op


class TransposeAttrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}) -> None:
        super().__init__(attrs, "TransposeAttrs")

    def checkparams(self) -> None:
        self.attrs["perm"] = self.attrs.get("perm", [])
        assert len(self.attrs["perm"]) < 7
        self.attrs["ndim"] = len(self.attrs["perm"])

    def serialize(self) -> bytes:
        attrs = tffi.new("TransposeAttrs *")

        attrs.axes_ = self.attrs["perm"]
        attrs.ndim_ = self.attrs["ndim"]

        return bytes(tffi.buffer(attrs))


@register_op
class Transpose(Operator):
    def __init__(self, attrs={}):
        self.attrs = TransposeAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1, "Check fail: number of input not equal 1"
        X = inputs[0]
        shape = list(X.shape)
        axes = self.attrs["perm"]
        if len(axes) == 0:
            axes = list(range(len(shape) - 1, -1, -1))
        assert len(axes) <= len(shape)
        new_shape = [shape[x] for x in axes]
        new_shape += shape[len(axes) :]

        Y = X.clone(shape=tuple(new_shape))
        if X.layout == Layout.NCHW and tuple(axes) == (0, 1, 3, 2):
            Y.layout = Layout.NCWH
        elif X.layout == Layout.NCWH and tuple(axes) == (0, 1, 3, 2):
            Y.layout = Layout.NCHW

        self.outputs = [Y]

__all__ = ["Transpose"]
