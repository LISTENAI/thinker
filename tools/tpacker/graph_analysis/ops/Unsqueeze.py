import numpy as np

from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class UnsqueezeAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Validate and set default parameters for Unsqueeze operation."""
        self.attrs["axes"] = self.attrs.get("axes", [])
        assert len(self.attrs["axes"]) < 7, "axes length must be less than 7"

    def serialize(self) -> bytes:
        """Serialize Unsqueeze attributes to bytes."""
        attrs = tffi.new("SqueezeAttrs *")
        attrs.axes = self.attrs["axes"]
        return bytes(tffi.buffer(attrs))

@register_op
class Unsqueeze(Operator):
    def __init__(self, attrs={}):
        """Initialize Unsqueeze operator with given attributes."""
        self.attrs = UnsqueezeAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape by adding dimensions at specified axes."""
        inputs = self.inputs
        assert len(inputs) == 1, "Unsqueeze operation requires exactly one input"
        X = inputs[0]

        tShape = list(X.shape)
        axes = self.attrs["axes"]
        rank = len(tShape) + len(axes)

        # Validate and adjust axes
        for i in range(len(axes)):
            x = axes[i]
            assert x < rank and x >= -rank, f"Axis {x} out of bounds"
            if x < 0:
                x += rank
                axes[i] = x

        assert len(axes) == len(set(axes)), "Duplicate axes are not allowed"
        axes = sorted(axes)

        # Insert new dimensions
        for x in axes:
            if x < 0:
                if x == -1:
                    tShape.append(1)
                else:
                    tShape.insert(x + 1, 1)
            else:
                tShape.insert(x, 1)

        yshape = tShape
        Y = X.clone(shape=tuple(yshape), scale=X.scale)

        # Reshape data if available
        if X.has_data():
            if isinstance(X.data, int):
                Y.data = np.array([X.data], dtype=X.dtype.type)
            else:
                Y.data = X.data.reshape(yshape)

        self.outputs = [Y]

    def is_inplace(self) -> bool:
        """Check if the operation can be performed in-place."""
        return self.inputs[0].mem_type == self.outputs[0].mem_type

__all__ = ["Unsqueeze"]