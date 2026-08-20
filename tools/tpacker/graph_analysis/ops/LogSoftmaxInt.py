import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType, MemType, ALIGN4
from ...xsympy import is_sympy
from .utils import calc_expr
from ...resource_packer._type._ctype import tffi
from .base import iqUnaryOperator, iqUnaryOperatorAttrs, register_op


class LogSoftmaxIntAttrs(iqUnaryOperatorAttrs):
    def normalize(self) -> None:
        super().normalize()
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            dim = self.attrs.get("dim")
            axis = self.attrs.get("axis", dim)
            assert axis is not None, "Missing required attribute: dim"
            if dim is not None:
                assert int(axis) == int(dim), "LogSoftmaxInt axis and dim must match"
            self.attrs["axis"] = int(axis)
            self.attrs["dim"] = int(axis)

    def checkparams(self) -> None:
        """Check if required parameters are present and valid."""
        platform = self.attrs.get("platform", "venus")
        if platform in {"arcs", "venusA"}:
            assert "axis" in self.attrs, "Missing required attribute: axis"
        elif platform == "venus":
            assert "axis" in self.attrs, "Missing required attribute: dim"
        else:
            raise AssertionError("Unsupported platform: {}".format(platform))

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the LogSoftmaxInt operation."""
        attrs = tffi.new("LogSoftmaxIntAttrs *")
        platform = self.attrs.get("platform", "venus")
        axis = self.attrs["axis"]
        assert -128 <= axis <= 127, "LogSoftmaxInt axis exceeds serialized range"
        attrs.axis = axis
        return bytes(tffi.buffer(attrs))

@register_op
class LogSoftmaxInt(iqUnaryOperator):
    def __init__(self, attrs={}):
        """Initialize the LogSoftmaxInt operator with given attributes."""
        self.attrs = LogSoftmaxIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on input."""
        X = self.inputs[0]
        platform = self.attrs.get("platform", "venus")
        assert len(self.inputs) == 1, "LogSoftmaxInt requires one input"
        assert len(X.shape) > 0, "LogSoftmaxInt requires a non-scalar input"
        assert X.zero == 0, "LogSoftmaxInt only supports zero point 0"
        if platform == "venus":
            axis = self.attrs["axis"]
            assert -len(X.shape) <= axis < len(X.shape), "Axis out of bounds"
            normalized_axis = axis + len(X.shape) if axis < 0 else axis
            assert normalized_axis == len(X.shape) - 1, "LogSoftmaxInt on venus only supports the last axis"
            assert X.dtype == np.int8, "LogSoftmaxInt on venus only supports int8 input"
            if is_sympy(X.shape[normalized_axis]):
                assert calc_expr(str(X.shape[normalized_axis]), dynamic_shape) <= 2048, "Exceed logsoftmax limit"
            else:
                assert X.shape[normalized_axis] <= 2048, "Exceed logsoftmax limit"
        else:
            axis = int(self.attrs["axis"])
            axis = axis + len(X.shape) if axis < 0 else axis
            assert 0 <= axis < len(X.shape), "Axis out of bounds"
            assert axis == len(X.shape) - 1, "LogSoftmaxInt only supports the last axis"
            assert X.dtype == np.int8, "LogSoftmaxInt only supports int8 input"
            if is_sympy(X.shape[axis]):
                assert calc_expr(str(X.shape[axis]), dynamic_shape) <= 2048, "Exceed logsoftmax limit"
            else:
                assert 0 < X.shape[axis] <= 2048, "Exceed logsoftmax limit"

        for dim in X.shape:
            if not is_sympy(dim):
                assert dim > 0, "LogSoftmaxInt dimensions must be positive"

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

        if platform == "arcs":
            input_shift = 25 - X.scale
            output_shift = 25 - int(temp)
            assert 0 <= input_shift <= 30, "LogSoftmaxInt on arcs input shift exceeds Luna/scalar limits"
            assert 0 <= output_shift <= 63, "LogSoftmaxInt on arcs output shift exceeds Luna limit"
        elif platform == "venusA":
            input_shift = 25 - X.scale
            output_shift = 15 - int(temp)
            assert -63 <= input_shift <= 30, "LogSoftmaxInt on venusA input shift exceeds Luna/scalar limits"
            assert -30 <= output_shift <= 63, "LogSoftmaxInt on venusA output shift exceeds Luna/scalar limits"
        elif platform == "venus":
            assert 0 <= 25 - X.scale <= 30, "LogSoftmaxInt on venus input shift exceeds runtime limits"
            assert 0 <= 25 - int(temp) <= 63, "LogSoftmaxInt on venus output shift exceeds runtime limits"

        # Create output tensor
        Y = X.clone(scale=int(temp))
        assert Y.zero == 0, "LogSoftmaxInt output zero point must be 0"
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the LogSoftmaxInt operation."""
        platform = self.attrs.get("platform", "venus")
        workspace_size = 0

        if platform in {"arcs", "venusA"}:
            axis = self.attrs["axis"]
            input_shape = self.inputs[0].shape
            axis = axis + len(input_shape) if axis < 0 else axis
            input_size = np.prod(input_shape)
            stride = np.prod(input_shape[axis:])
            if platform == "arcs":
                assert stride <= 2048, "LogSoftmaxInt on arcs exceeds Luna softmax stride limit"
                workspace_size = stride * 8
            else:
                assert self.inputs[0].mem_type == MemType.SHARE_MEM, "LogSoftmaxInt on venusA requires SHARE_MEM input"
                assert self.outputs[0].mem_type == MemType.SHARE_MEM, "LogSoftmaxInt on venusA requires SHARE_MEM output"
                assert stride <= 2048, "LogSoftmaxInt on venusA exceeds Luna logsoftmax stride limit"
                if self.inputs[0].dtype == np.int8:
                    workspace_size += ALIGN4(input_size * 2) + input_size * 8
                else:
                    workspace_size += input_size * 8
        elif platform == "venus":
            assert self.inputs[0].mem_type == MemType.SHARE_MEM, "LogSoftmaxInt on venus requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, "LogSoftmaxInt on venus requires SHARE_MEM output"
            axis = self.attrs.get("axis", self.attrs.get("dim"))
            axis = axis + len(self.inputs[0].shape) if axis < 0 else axis
            workspace_size = self.inputs[0].shape[axis] * 8

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
