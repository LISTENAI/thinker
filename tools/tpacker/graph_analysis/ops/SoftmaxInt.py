import math
import numpy as np

from ...graph import Tensor
from ...resource_packer._type._ctype import tffi
from ...enum_defines import DevType, MemType, ALIGN4
from ...xsympy import is_sympy
from .utils import calc_expr
from .base import Operator, OperatorAttrs, iqUnaryOperator, register_op

class SoftmaxIntAttrs(OperatorAttrs):
    def checkparams(self):
        platform = self.attrs.get("platform", "venus")
        assert platform in {"venus", "arcs", "mars", "venusA"}, \
            f"Unsupported SoftmaxInt platform: {platform}"
        axis = self.attrs.get("axis", self.attrs.get("dim"))
        assert axis is not None and isinstance(axis, (int, np.integer)), \
            "SoftmaxInt requires an integer axis or dim"

    def serialize(self) -> bytes:
        """Serialize SoftmaxInt attributes to bytes."""
        attrs = tffi.new("SoftmaxIntAttrs *")
        platform = self.attrs.get("platform", "venus")
        axis = self.attrs.get("axis", self.attrs.get("dim"))
        attrs.axis = axis
        return bytes(tffi.buffer(attrs))

@register_op
class SoftmaxInt(iqUnaryOperator):
    def __init__(self, attrs={}):
        self.attrs = SoftmaxIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape and properties based on input tensor."""
        inputs = self.inputs
        assert len(inputs) == 1, "SoftmaxInt expects exactly one input"
        X = inputs[0]

        platform = self.attrs.get("platform", "venus")
        axis = self.attrs.get("axis", self.attrs.get("dim"))

        assert axis is not None, "SoftmaxInt requires axis or dim"
        axis = axis + len(X.shape) if axis < 0 else axis
        assert 0 <= axis < len(X.shape), "Axis out of bounds"
        self.attrs["axis"] = axis
        if platform == "venus":
            assert X.dtype == np.int8, "SoftmaxInt on venus only supports int8 input"
            assert axis == len(X.shape) - 1, "SoftmaxInt on venus only supports the last axis"
        if platform in {"arcs", "mars"}:
            assert X.dtype == np.int8, "SoftmaxInt on arcs only supports int8 input"
            assert axis == len(X.shape) - 1, "SoftmaxInt on arcs only supports the last axis"
        if platform == "venusA":
            assert X.dtype in (np.int8, np.int16, np.int32), \
                "SoftmaxInt on venusA only supports int8/int16/int32 input"
            assert axis == len(X.shape) - 1, "SoftmaxInt on venusA only supports the last axis"

        # Check softmax dimension limit
        if is_sympy(X.shape[axis]):
            axis_size = calc_expr(str(X.shape[axis]), dynamic_shape)
            assert 1 <= axis_size <= 2048, "Softmax dimension must be in [1, 2048]"
        else:
            assert 1 <= X.shape[axis] <= 2048, "Softmax dimension must be in [1, 2048]"

        assert X.zero == 0, "SoftmaxInt only supports symmetric input quantization"

        # Handle scale_x
        scale_x = self.attrs.get("scale_x")
        assert scale_x is not None and scale_x > 0, "SoftmaxInt scale_x must be positive"
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "scale_x must be a power of 2"
        expected_scale = int(temp)
        if X.scale != -1:
            assert X.scale == expected_scale, "Input scale must match scale_x"
        else:
            X.scale = expected_scale

        # Handle scale_o
        scale_o = self.attrs.get("scale_o")
        assert scale_o is not None and scale_o > 0, "SoftmaxInt scale_o must be positive"
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "scale_o must be a power of 2"

        # Determine output data type and bits
        o_bits = self.attrs.get("o_bits", 8)
        if platform == "venus":
            assert o_bits == 8, "SoftmaxInt on venus only supports int8 output"
            assert 0 <= 25 - X.scale <= 30, "SoftmaxInt on venus input shift exceeds runtime limits"
            assert 0 <= 15 - int(temp) <= 63, "SoftmaxInt on venus output shift exceeds runtime limits"
        if platform in {"arcs", "mars"}:
            assert o_bits in (8, 32), "SoftmaxInt on arcs only supports int8 or int32 output"
            input_shift = 25 - X.scale
            assert 0 <= input_shift <= 30, "SoftmaxInt on arcs input shift exceeds Luna/scalar limits"
            if o_bits == 8:
                output_shift = 15 - int(temp)
                assert 0 <= output_shift <= 63, "SoftmaxInt on arcs output shift exceeds Luna limits"
            else:
                assert int(temp) == 15, "SoftmaxInt on arcs int32 output scale must be 2^15"
        if platform == "venusA":
            input_shift = 25 - X.scale
            output_shift = 15 - int(temp)
            assert -63 <= input_shift <= 30, "SoftmaxInt on venusA input shift exceeds Luna/scalar limits"
            assert -30 <= output_shift <= 63, "SoftmaxInt on venusA output shift exceeds Luna/scalar limits"

        if o_bits == 32:
            data_type = np.dtype("i4")
            bits = 4
        elif o_bits == 16:
            data_type = np.dtype("i2")
            bits = 2
        elif o_bits == 8:
            data_type = np.dtype("i1")
            bits = 1
        else:
            raise ValueError(f"Unsupported o_bits value: {o_bits}")

        # Create output tensor
        Y = X.clone(dtype=data_type, bits=bits, scale=int(temp))
        Y.zero = 0
        self.outputs = [Y]

    def get_workspace(self):
        """Calculate the required workspace size for the operation."""
        axis = self.attrs.get("axis", self.attrs.get("dim", -1))
        input_shape = self.inputs[0].shape
        axis = axis + len(input_shape) if axis < 0 else axis
        size = 1
        for i in range(axis, len(input_shape)):
            size *= input_shape[i]

        platform = self.attrs.get("platform", "venus")
        workspace_sizes = 0
        if platform in {"arcs", "mars"}:
            input_size = np.prod(input_shape)
            workspace_sizes += input_size * 4
            if self.outputs[0].dtype == np.int8 or self.outputs[0].mem_type != MemType.SHARE_MEM:
                workspace_sizes += size * 4
            if (self.inputs[0].mem_type != MemType.SHARE_MEM or
                    (self.outputs[0].dtype == np.int8 and self.outputs[0].mem_type != MemType.SHARE_MEM)):
                workspace_sizes += size
        elif platform == "venusA":
            input_size = np.prod(input_shape)
            stride = np.prod(input_shape[axis:])
            assert stride <= 2048, "SoftmaxInt on venusA exceeds Luna softmax stride limit"
            if self.inputs[0].dtype == np.int8:
                workspace_sizes += ALIGN4(input_size * 2) + input_size * 4
            else:
                workspace_sizes += input_size * 4
            workspace_sizes += stride * 4
            workspace_sizes = min(workspace_sizes, 65536)
        else:
            workspace_sizes = input_shape[axis] * 8

        if workspace_sizes != 0:
            max_workspace = Tensor.from_shape([workspace_sizes], np.int8, MemType.SHARE_MEM)
            return [max_workspace]

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations."""
        X = self.inputs[0]
        Y = self.outputs[0]
        xshape = list(X.shape)
        yshape = list(Y.shape)

        # Evaluate symbolic dimensions
        for i, s in enumerate(xshape):
            if is_sympy(s):
                xshape[i] = calc_expr(str(s), dynamic_shape)
        for i, s in enumerate(yshape):
            if is_sympy(s):
                yshape[i] = calc_expr(str(s), dynamic_shape)

        flops = int(np.prod(yshape)) * 4
        return flops

__all__ = ["SoftmaxInt"]
