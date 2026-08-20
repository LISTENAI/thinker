import math
import numpy as np
from ...resource_packer._type._ctype import tffi
from .utils import combine4bit_8bit
from ...enum_defines import DevType, Layout
from ...xsympy import is_sympy
from .base import Operator, OperatorAttrs, register_op

class GatherAttrs(OperatorAttrs):
    def normalize(self):
        """Normalize the attributes, setting default values if necessary."""
        self.attrs["axis"] = self.attrs.get("axis", 0)

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the Gather operation."""
        attrs = tffi.new("GatherAttrs *")
        attrs.axis = self.attrs["axis"]
        return bytes(tffi.buffer(attrs))

@register_op
class Gather(Operator):
    def __init__(self, attrs={}):
        """Initialize the Gather operator with given attributes."""
        self.attrs = GatherAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and data based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 2, "Gather operator must have exactly two inputs"

        data = inputs[0]
        indices = inputs[1]
        scale_w = self.attrs.get("scale_w")
        if scale_w is not None:
            input_scale = math.log(scale_w, 2)
            assert abs(input_scale - int(input_scale)) < 0.000001, "Scale must be a power of 2"
            data.scale = int(input_scale)
        tShape = list(data.shape)
        ishape = list(indices.shape)
        axis = self.attrs["axis"]
        ndim = len(tShape)

        # Validate axis
        assert -ndim <= axis < ndim, "Axis out of bounds"
        axis = axis + ndim if axis < 0 else axis
        self.attrs["axis"] = axis

        platform = self.attrs.get("platform", "venus")
        assert indices.dtype in (np.dtype(np.int32), np.dtype(np.int64)), \
            "Gather indices must be int32 or int64"
        input_bits = self.attrs.get("parameter_bits", 8)
        assert input_bits in (4, 8), "Gather parameter_bits must be 4 or 8"
        if input_bits == 4:
            assert platform == "arcs", "Packed int4 Gather is only supported on arcs"
            assert data.dtype == np.dtype(np.int8) and data.has_data(), \
                "Packed int4 Gather requires constant int8 data"
            tail = int(np.prod(tShape[axis + 1:]))
            assert tail % 2 == 0, \
                "Packed int4 Gather requires an even number of elements per gathered slice"
        if platform in {"venus", "venusA"}:
            assert data.bits != 0.5 and self.attrs.get("parameter_bits", 8) != 4, \
                "Gather on venus does not support packed int4 data"
        if indices.has_data():
            assert np.all(indices.data >= -1), "Gather only supports indices >= -1"
            if not is_sympy(tShape[axis]):
                assert np.all(indices.data < tShape[axis]), \
                    "Gather indices must be less than the axis size"

        # Compute output shape
        yshape = tShape[:axis] + ishape + tShape[axis + 1:]
        Y = data.clone(shape=yshape, scale=data.scale)

        # Perform data gathering if applicable
        if indices.has_data() and data.has_data():
            indices_flat = indices.data.reshape(-1)
            Y.data = np.take(data.data, indices_flat, axis=axis)
            if len(indices.shape) == 0:
                Y.data = np.array(Y.data)

        # Set output scale
        scale_o = self.attrs.get("scale_o", -1)
        if scale_o < 0:
            Y.scale = data.scale
        else:
            temp = math.log(scale_o, 2)
            assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
            Y.scale = int(temp)
        assert Y.scale == data.scale, "Gather cannot change the input scale"

        self.outputs = [Y]

    def pack_params(self):
        """Pack the parameters for the Gather operation, handling data type conversions."""
        data = self.inputs[0]
        input_bits = self.attrs.get('parameter_bits', 8)
        if data.dtype == np.int8 and input_bits in (4, 8):
            if input_bits == 4:
                new_weight_data = combine4bit_8bit(data.data)
                self.inputs[0].update(data=new_weight_data, bits=np.float32(input_bits / 8))

    def sub_layout_convert(self):
        axis = self.attrs["axis"]
        rank = len(self.inputs[0].shape)
        layout = self.inputs[0].layout
        if rank == 4 and layout == Layout.NHWC:
            self.attrs["axis"] = {1: 3, 2: 1, 3: 2}.get(axis, axis)
        elif rank == 4 and layout == Layout.NCWH:
            self.attrs["axis"] = {2: 3, 3: 2}.get(axis, axis)
        elif rank == 3 and layout in (Layout.NHWC, Layout.NCWH):
            self.attrs["axis"] = {1: 2, 2: 1}.get(axis, axis)

__all__ = ["Gather"]
