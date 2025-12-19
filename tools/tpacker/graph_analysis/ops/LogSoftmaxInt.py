import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from ...xsympy import is_sympy
from .utils import calc_expr
from ...resource_packer._type._ctype import tffi
from .base import iqUnaryOperator, iqUnaryOperatorAttrs, register_op


class LogSoftmaxIntAttrs(iqUnaryOperatorAttrs):
    def checkparams(self) -> None:
        """Check if required parameters are present and valid."""
        platform = self.attrs.get("platform", "venus")
        if platform in {"arcs", "venusA"}:
            assert "axis" in self.attrs, "Missing required attribute: axis"
        elif platform == "venus":
            assert "dim" in self.attrs, "Missing required attribute: dim"
        else:
            raise AssertionError("Unsupported platform: {}".format(platform))

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the LogSoftmaxInt operation."""
        attrs = tffi.new("LogSoftmaxIntAttrs *")
        platform = self.attrs.get("platform", "venus")
        if platform in {"arcs", "venusA"}:
            attrs.axis = self.attrs["axis"]
        elif platform == "venus":
            attrs.axis = self.attrs["dim"]
        else:
            raise AssertionError("Unsupported platform: {}".format(platform))
        return bytes(tffi.buffer(attrs))

@register_op
class LogSoftmaxInt(iqUnaryOperator):
    def __init__(self, attrs={}):
        """Initialize the LogSoftmaxInt operator with given attributes."""
        self.attrs = LogSoftmaxIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on input."""
        X = self.inputs[0]

        # Process input scale
        scale_x = self.attrs["scale_x"]
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Input scale must be a power of 2"
        if X.scale != -1:
            assert X.scale == int(temp), "Input scale must match attribute scale_x"
        else:
            X.scale = int(temp)

        # Process output scale
        scale_o = self.attrs["scale_o"]
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Output scale must be a power of 2"

        # Create output tensor
        Y = X.clone(scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the LogSoftmaxInt operation."""
        platform = self.attrs.get("platform", "venus")
        workspace_size = 0

        if platform in {"arcs", "venusA"}:
            axis = self.attrs["axis"]
            input_size = np.prod(self.inputs[0].shape)
            stride = np.prod(self.inputs[0].shape[axis:])
            if self.inputs[0].dtype == np.int8:
                workspace_size += input_size * 6
            else:
                workspace_size += input_size * 4
            workspace_size += stride * 4
        elif platform == "venus":
            axis = self.attrs["dim"]
            workspace_size = self.inputs[0].shape[axis] * 2

        workspace_size = min(workspace_size, 65536)
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the LogSoftmaxInt operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        xshape = list(X.shape)
        yshape = list(Y.shape)

        # Resolve symbolic expressions in shapes
        for i, s in enumerate(xshape):
            if is_sympy(s):
                xshape[i] = calc_expr(str(s), dynamic_shape)
        for i, s in enumerate(yshape):
            if is_sympy(s):
                yshape[i] = calc_expr(str(s), dynamic_shape)

        # Calculate FLOPs
        flops = int(np.prod(yshape)) * 4
        return flops

__all__ = ["LogSoftmaxInt"]