import numpy as np

from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class SqueezeAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert self.attrs.get("noop_with_empty_axes", 0) in (0, 1), \
            "Squeeze noop_with_empty_axes must be 0 or 1"

    def serialize(self) -> bytes:
        """Serialize Squeeze attributes to bytes."""
        attrs = tffi.new("SqueezeAttrs *")
        axes = self.attrs.get("axes", [])
        attrs.axes = axes
        attrs.ndim = len(axes)
        return bytes(tffi.buffer(attrs))

@register_op
class Squeeze(Operator):
    def __init__(self, attrs={}):
        self.attrs = SqueezeAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape based on input tensor and squeeze parameters."""
        inputs = self.inputs
        assert len(inputs) in (1, 2), "Squeeze expects one or two inputs"
        X = inputs[0]
        input_shape = list(X.shape)

        if len(inputs) == 2:
            axes_input = inputs[1]
            assert axes_input.has_data(), "Squeeze axes input must be constant"
            assert axes_input.dtype == np.int64 and len(axes_input.shape) == 1, \
                "Squeeze axes input must be a one-dimensional int64 tensor"
            axes = np.asarray(axes_input.data).reshape(-1).tolist()
        else:
            axes = list(self.attrs.get("axes", []))

        # Automatically detect axes to squeeze if not provided
        if not axes:
            if self.attrs.get("noop_with_empty_axes", 0) == 0:
                axes = [i for i, size in enumerate(input_shape) if size == 1]
        else:
            rank = len(input_shape)
            normalized_axes = []
            for axis in axes:
                axis = int(axis)
                assert -rank <= axis < rank, f"Axis {axis} out of bounds"
                normalized_axes.append(axis + rank if axis < 0 else axis)
            assert len(normalized_axes) == len(set(normalized_axes)), \
                "Duplicate axes are not allowed"
            axes = sorted(normalized_axes)

        for axis in axes:
            assert input_shape[axis] == 1, \
                f"Cannot squeeze axis {axis} with size {input_shape[axis]}"
        yshape = [size for i, size in enumerate(input_shape) if i not in axes]
        self.attrs["axes"] = axes

        # Create output tensor
        Y = X.clone(shape=tuple(yshape), scale=X.scale)
        if X.has_data():
            Y.data = X.data.reshape(yshape)
        self.outputs = [Y]

    def is_inplace(self) -> bool:
        """Check if the operation can be performed in-place."""
        assert self.inputs[0].mem_type == self.outputs[0].mem_type, \
            "Squeeze input and output must use the same memory type"
        return True

__all__ = ["Squeeze"]
