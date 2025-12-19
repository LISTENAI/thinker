from .utils.utils import *
from ...xsympy import *
from ...resource_packer._type._ctype import tffi
from ...enum_defines import Layout, DevType
from .base import Operator, OperatorAttrs, register_op

def conv_int(value):
    """Convert value to int if it's not a symbolic variable."""
    return value if is_sympy(value) else int(value)

class SliceAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        """Serialize Slice attributes to bytes."""
        attrs = tffi.new("SliceAttrs *")
        attrs.axis = self.attrs["axis"]
        attrs.dims = self.attrs["dims"]
        attrs.split = self.attrs["split"]
        return bytes(tffi.buffer(attrs))

@register_op
class Slice(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape based on input tensor and slicing parameters."""
        inputs = self.inputs
        assert len(inputs) >= 3, "At least three inputs are required"
        X = inputs[0]
        shape = list(X.shape)

        # Parse slicing parameters
        starts = conv_int(inputs[1].data[0])
        ends = conv_int(inputs[2].data[0])
        steps = 1
        axes = 0

        if len(inputs) >= 5:
            steps = conv_int(inputs[4].data[0])
            axes = conv_int(inputs[3].data[0])
        elif len(inputs) == 4:
            axes = conv_int(inputs[3].data[0])

        assert axes < len(shape), "Axis out of bounds"
        axes = axes + len(shape) if axes < 0 else axes

        # Adjust starts and ends for negative values
        if not is_sympy(starts):
            starts = starts + shape[axes] if starts < 0 else starts
            starts = max(starts, 0)
        if not is_sympy(ends):
            ends = ends + shape[axes] if ends < 0 else ends
            ends = min(ends, shape[axes])

        # Calculate output shape
        if steps < 0:
            shape[axes] = (starts - ends + 1 + steps) // (-steps)
        else:
            shape[axes] = (ends - starts + steps - 1) // steps

        # Create output tensor
        Y = X.clone(shape=tuple(shape))
        if X.has_data() and not is_sympy(starts) and not is_sympy(ends):
            if axes == 0:
                Y.data = X.data[starts:ends:steps]
            elif axes == 1:
                Y.data = X.data[:, starts:ends:steps]

        # Handle dynamic data
        if is_sympy(starts):
            inputs[1].is_dynamic_data = True
        if is_sympy(ends):
            inputs[2].is_dynamic_data = True
        self.outputs = [Y]

    def sub_layout_convert(self):
        """Convert layout for NHWC format."""
        inputs = self.inputs
        if inputs[0].layout == Layout.NHWC:
            axes = inputs[3].data[0]
            if axes == 1:
                axes = 3
            elif axes == 2:
                axes = 1
            elif axes == 3:
                axes = 2
            inputs[3].data[0] = axes

__all__ = ["Slice"]