import math
import numpy as np
from ...xsympy import is_sympy
from ...graph import Tensor
from .utils import calc_expr
from ...resource_packer._type._ctype import tffi
from ...enum_defines import MemType, Layout
from .base import Operator, OperatorAttrs, register_op

class GluIntAttrs(OperatorAttrs):
    def normalize(self):
        axis = self.attrs.get("axis", self.attrs.get("dim"))
        assert axis is not None, "GluInt requires axis or dim"
        if "axis" in self.attrs and "dim" in self.attrs:
            assert self.attrs["axis"] == self.attrs["dim"], \
                "GluInt axis and dim must agree"
        self.attrs["axis"] = axis

    def checkparams(self):
        for name in ("scale_x", "scale_o", "o_bits", "platform"):
            assert name in self.attrs, f"Missing required attribute: {name}"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the GluInt operation."""
        attrs = tffi.new("GluIntAttrs *")
        axis = self.attrs.get("axis", self.attrs.get("dim"))
        attrs.axis = axis
        return bytes(tffi.buffer(attrs))

@register_op
class GluInt(Operator):
    def __init__(self, attrs={}):
        """Initialize the GluInt operator with given attributes."""
        self.attrs = GluIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "GluInt operator must have exactly one input"

        X = inputs[0]
        shape = list(X.shape)
        axis = self.attrs["axis"]
        ndims = len(shape)
        assert ndims > 0, "GluInt does not support scalar input"
        assert -ndims <= axis < ndims, "GluInt axis out of bounds"
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            assert axis == -1, "GluInt on venus only supports the last axis"
            assert X.dtype == np.int8, "GluInt on venus only supports int8 input"
            assert np.prod(shape[:-1]) == 1, "GluInt on venus only supports one contiguous split row"
        elif platform in {"arcs", "venusA"}:
            assert X.dtype == np.int8, "GluInt on arcs/venusA runtime only supports int8 input"
            if platform == "arcs":
                assert axis in (-1, ndims - 1), "GluInt on arcs only supports the last axis"
        axis = axis + ndims if axis < 0 else axis
        self.attrs["axis"] = axis
        assert shape[axis] % 2 == 0, "GluInt split axis dimension must be even"
        assert shape[axis] > 0, "GluInt split axis dimension must be positive"
        shape[axis] = shape[axis] // 2

        # Process scales
        scale_x = self.attrs.get("scale_x")
        if isinstance(scale_x, tuple):
            assert len(scale_x) == 1, "GluInt only supports per-tensor input scale"
            scale_x = scale_x[0]
        assert np.isfinite(scale_x) and scale_x > 0, \
            "GluInt scale_x must be finite and positive"
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        if X.scale != -1:
            assert X.scale == int(temp), "Input scale must match attribute scale_x"
        else:
            self.inputs[0].scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        if isinstance(scale_o, tuple):
            assert len(scale_o) == 1, "GluInt only supports per-tensor output scale"
            scale_o = scale_o[0]
        assert np.isfinite(scale_o) and scale_o > 0, \
            "GluInt scale_o must be finite and positive"
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert X.zero == 0, "GluInt only supports zero point 0"

        # Determine output data type
        output_bits = self.attrs.get("o_bits")
        if platform in {"venus", "arcs"}:
            assert output_bits == 8, "GluInt on venus/arcs only supports int8 output"
        assert output_bits in (8, 16, 32), "Output bits must be 8, 16, or 32"
        dtype = np.int8 if output_bits == 8 else np.int16 if output_bits == 16 else np.int32

        if platform == "venusA":
            x_q = X.scale
            y_q = int(temp)
            sigmoid_shift = 27 - x_q
            mul_shift = 15 + x_q - y_q
            assert -63 <= sigmoid_shift <= 30, "GluInt on venusA sigmoid input shift exceeds Luna/scalar limits"
            assert mul_shift <= 63, "GluInt on venusA output right shift exceeds Luna limit"
            assert mul_shift >= -30, "GluInt on venusA output left shift exceeds scalar range"
        elif platform == "arcs":
            sigmoid_shift = 27 - X.scale
            mul_shift = 31 + X.scale - int(temp)
            assert 0 <= sigmoid_shift <= 30, "GluInt on arcs requires sigmoid shift in [0, 30]"
            assert 0 <= mul_shift <= 63, "GluInt on arcs requires output shift in [0, 63]"
        elif platform == "venus":
            sigmoid_shift = 11 - X.scale
            mul_shift = 7 + X.scale - int(temp)
            assert -63 <= sigmoid_shift <= 6, "GluInt on venus Q11 conversion exceeds q7 scalar limits"
            assert 0 <= mul_shift <= 63, "GluInt on venus requires output shift in [0, 63]"

        Y = X.clone(shape=tuple(shape), scale=int(temp), dtype=dtype,
                    bits=int(output_bits / 8), zero=0)
        self.outputs = [Y]

    def get_workspace(self):
        """Calculate the required workspace for the GluInt operation."""
        axis = self.attrs["axis"]
        ndims = len(self.inputs[0].shape)
        axis = axis + ndims if axis < 0 else axis
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            assert self.inputs[0].mem_type == MemType.SHARE_MEM, "GluInt on venus requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, "GluInt on venus requires SHARE_MEM output"
        if platform == "arcs":
            assert self.inputs[0].mem_type == MemType.SHARE_MEM, "GluInt on arcs requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, "GluInt on arcs requires SHARE_MEM output"
        if platform == "venusA":
            assert self.inputs[0].mem_type in (MemType.SHARE_MEM, MemType.PSRAM), \
                "GluInt on venusA input must be in SHARE_MEM or PSRAM"
            assert self.outputs[0].mem_type in (MemType.SHARE_MEM, MemType.PSRAM), \
                "GluInt on venusA output must be in SHARE_MEM or PSRAM"
            bytes_per_element = 11 if self.inputs[0].mem_type == MemType.PSRAM else 10
            workspace_size = min(max(self.outputs[0].size * bytes_per_element + 3, 13), 65536)
        else:
            workspace_size = self.inputs[0].shape[axis] * 4 if platform == "arcs" else self.inputs[0].size * 7

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

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

__all__ = ["GluInt"]
