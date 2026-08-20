import numpy as np

from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class UnsqueezeAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Validate and set default parameters for Unsqueeze operation."""
        axes = self.attrs.get("axes")
        if axes is not None:
            assert isinstance(axes, (list, tuple, np.ndarray)), \
                "Unsqueeze axes must be a sequence"
            assert len(axes) <= 7, "Unsqueeze supports at most 7 axes"

    def serialize(self) -> bytes:
        """Serialize Unsqueeze attributes to bytes."""
        attrs = tffi.new("SqueezeAttrs *")
        axes = self.attrs.get("axes", [])
        attrs.axes = axes
        attrs.ndim = len(axes)
        return bytes(tffi.buffer(attrs))

@register_op
class Unsqueeze(Operator):
    def __init__(self, attrs={}):
        """Initialize Unsqueeze operator with given attributes."""
        self.attrs = UnsqueezeAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape by adding dimensions at specified axes."""
        inputs = self.inputs
        X = inputs[0]

        tShape = list(X.shape)
        if len(inputs) == 1:
            assert "axes" in self.attrs, "Unsqueeze requires axes"
            axes = list(self.attrs["axes"])
        elif len(inputs) == 2:
            axes_input = inputs[1]
            assert axes_input.has_data(), "Unsqueeze axes input must be constant"
            assert axes_input.dtype == np.int64 and len(axes_input.shape) == 1, \
                "Unsqueeze axes input must be a one-dimensional int64 tensor"
            axes = np.asarray(axes_input.data).reshape(-1).tolist()
        else:
            raise ValueError("Unsqueeze operation requires exactly one or two inputs")
        rank = len(tShape) + len(axes)
        assert rank <= 7, "Unsqueeze runtime supports at most 7 dimensions"

        # Validate and adjust axes
        for i in range(len(axes)):
            x = int(axes[i])
            assert x < rank and x >= -rank, f"Axis {x} out of bounds"
            if x < 0:
                x += rank
            axes[i] = x

        assert len(axes) == len(set(axes)), "Duplicate axes are not allowed"
        axes = sorted(axes)
        self.attrs["axes"] = axes

        # Insert new dimensions
        for x in axes:
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
        assert self.inputs[0].mem_type == self.outputs[0].mem_type, \
            "Unsqueeze input and output must use the same memory type"
        return True

__all__ = ["Unsqueeze"]
