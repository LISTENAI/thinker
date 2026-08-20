from .utils.utils import *
from ...xsympy import *
from ...resource_packer._type._ctype import tffi
from ...enum_defines import Layout, DevType, MemType
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
        assert 3 <= len(inputs) <= 5, "Slice requires three to five inputs"
        X = inputs[0]
        assert X.dtype.itemsize > 0 and float(X.bits) >= 1, \
            "Slice only supports byte-addressable data types"
        shape = list(X.shape)

        # VenusA consumes scalar begin/end/axis/step values, not vectors.
        for parameter in inputs[1:]:
            assert len(parameter.shape) == 1 and parameter.size == 1, \
                "Slice parameters must be one-element tensors"
            assert parameter.has_data(), "Slice parameters must be constant"
            assert parameter.dtype in (np.dtype(np.int32), np.dtype(np.int64)), \
                "Slice parameters must be int32 or int64"

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

        assert steps == 1, "Slice only supports step == 1"

        assert -len(shape) <= axes < len(shape), "Axis out of bounds"
        axes = axes + len(shape) if axes < 0 else axes

        # Adjust starts and ends for negative values
        dim_size = shape[axes]
        if not is_sympy(starts):
            starts = max(0, min(starts + dim_size if starts < 0 else starts, dim_size))
        if not is_sympy(ends):
            ends = max(0, min(ends + dim_size if ends < 0 else ends, dim_size))

        # Calculate output shape
        if steps < 0:
            shape[axes] = max(0, starts - ends)
        else:
            shape[axes] = max(0, (ends - starts + steps - 1) // steps)

        # Create output tensor
        Y = X.clone(shape=tuple(shape))
        if X.has_data() and not is_sympy(starts) and not is_sympy(ends):
            slices = [slice(None)] * len(shape)
            slices[axes] = slice(starts, ends, steps)
            Y.data = X.data[tuple(slices)]

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
