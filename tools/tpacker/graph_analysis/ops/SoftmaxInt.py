import math
import numpy as np

from ...graph import Tensor
from ...resource_packer._type._ctype import tffi
from ...enum_defines import DevType, MemType
from ...xsympy import is_sympy
from .utils import calc_expr
from .base import Operator, OperatorAttrs, iqUnaryOperator, register_op

class SoftmaxIntAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        """Serialize SoftmaxInt attributes to bytes."""
        attrs = tffi.new("SoftmaxIntAttrs *")
        platform = self.attrs.get("platform", "venus")
        if platform in ["arcs", "venusA"]:
            axis = self.attrs["axis"]
        elif platform == "venus":
            axis = self.attrs["dim"]
        else:
            raise AssertionError(f"Unsupported platform: {platform}")
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
        if platform in ["arcs", "venusA"]:
            axis = self.attrs["axis"]
        elif platform == "venus":
            axis = self.attrs["dim"]
        else:
            raise AssertionError(f"Unsupported platform: {platform}")

        # Check softmax dimension limit
        if is_sympy(X.shape[axis]):
            assert calc_expr(str(X.shape[axis]), dynamic_shape) <= 2048, "Exceed softmax limit of arcs"
        else:
            assert X.shape[axis] <= 2048, "Exceed softmax limit of arcs"

        # Handle scale_x
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "scale_x must be a power of 2"
        expected_scale = int(temp)
        if X.scale != -1:
            assert X.scale == expected_scale, "Input scale must match scale_x"
        else:
            X.scale = expected_scale

        # Handle scale_o
        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "scale_o must be a power of 2"

        # Create output tensor
        Y = X.clone(scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self):
        """Calculate the required workspace size for the operation."""
        axis = self.attrs.get("axis", -1)
        input_shape = self.inputs[0].shape
        axis = axis + len(input_shape) if axis < 0 else axis
        size = 1
        for i in range(axis, len(input_shape)):
            size *= input_shape[i]

        platform = self.attrs.get("platform", "venus")
        workspace_sizes = 0
        if platform == "arcs":
            if self.inputs[0].dtype == np.int8:
                workspace_sizes += self.inputs[0].nbytes * 4
            if self.outputs[0].dtype == np.int8:
                workspace_sizes += size * 4
        elif platform == "venusA":
            input_size = np.prod(input_shape)
            stride = np.prod(input_shape[axis:])
            if self.inputs[0].dtype == np.int8:
                workspace_sizes += input_size * 6
            else:
                workspace_sizes += input_size * 4
            workspace_sizes += stride * 4
        else:
            workspace_sizes = input_shape[axis] * 2

        workspace_sizes = min(workspace_sizes, 65536)
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