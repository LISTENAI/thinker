from typing import Any, Dict, Optional, List
from ...enum_defines import DevType, Layout
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class PackingAttrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}) -> None:
        super().__init__(attrs, "TransposeAttrs")

    def checkparams(self) -> None:
        """Check and set the parameters for the Packing operation."""
        self.attrs["perm"] = self.attrs.get("perm", [])
        assert len(self.attrs["perm"]) < 7, "Maximum permutation length is 6"
        self.attrs["ndim"] = len(self.attrs["perm"])

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the Packing operation."""
        attrs = tffi.new("PackingAttrs *")
        attrs.axes_ = self.attrs["perm"]
        attrs.ndim_ = self.attrs["ndim"]
        return bytes(tffi.buffer(attrs))

@register_op
class Packing(Operator):
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        """Initialize the Packing operator with given attributes."""
        self.attrs = PackingAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on input."""
        inputs = self.inputs
        assert len(inputs) == 1, "Packing operator must have exactly one input"
        X = inputs[0]
        shape = list(X.shape)
        axes = self.attrs["perm"]

        if not axes:
            axes = list(range(len(shape) - 1, -1, -1))

        assert len(axes) <= len(shape), "Permutation axes exceed input dimensions"

        new_shape = [shape[x] for x in axes]
        new_shape += shape[len(axes):]

        Y = X.clone(shape=tuple(new_shape), scale=X.scale)

        if X.layout == Layout.NCHW and tuple(axes) == (0, 1, 3, 2):
            Y.layout = Layout.NCWH
        elif X.layout == Layout.NCWH and tuple(axes) == (0, 1, 3, 2):
            Y.layout = Layout.NCHW

        self.outputs = [Y]

__all__ = ["Packing"]