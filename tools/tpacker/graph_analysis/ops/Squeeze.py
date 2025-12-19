from ...enum_defines import DevType
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class SqueezeAttrs(OperatorAttrs):
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
        assert len(inputs) == 1, "Squeeze expects exactly one input"
        X = inputs[0]
        tShape = list(X.shape)
        axes = self.attrs.get("axes", [])

        # Automatically detect axes to squeeze if not provided
        if not axes:
            yshape = []
            for i, s in enumerate(tShape):
                if s != 1:
                    yshape.append(s)
                else:
                    axes.append(i)
            # Ensure at least two dimensions
            if len(yshape) == 1:
                yshape = [1, yshape[0]]
        else:
            # Validate and squeeze specified axes
            for x in axes:
                assert x < len(tShape) and x >= -len(tShape), f"Axis {x} out of bounds"
                assert tShape[x] == 1, f"Cannot squeeze axis {x} with size {tShape[x]}"
                tShape[x] = None
            yshape = [x for x in tShape if x is not None]
            # Ensure at least two dimensions
            if len(yshape) == 1:
                yshape = [1, yshape[0]]

        # Create output tensor
        Y = X.clone(shape=tuple(yshape), scale=X.scale)
        self.outputs = [Y]

    def is_inplace(self) -> bool:
        """Check if the operation can be performed in-place."""
        return self.inputs[0].mem_type == self.outputs[0].mem_type

__all__ = ["Squeeze"]