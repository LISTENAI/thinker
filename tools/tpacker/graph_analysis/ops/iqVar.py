import math
import numpy as np
from typing import List

from ...graph import Tensor
from .utils import calc_expr
from ...xsympy import is_sympy
from ...enum_defines import DevType, MemType, ALIGN4
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class iqVarAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        self.attrs["dims"] = self.attrs.get("dims", 2)
        assert isinstance(self.attrs["dims"], (int, np.integer)) and \
               not isinstance(self.attrs["dims"], bool), "iqVar dims must be an integer"
        assert -128 <= self.attrs["dims"] <= 127, "iqVar dims must fit in int8"

    def serialize(self) -> bytes:
        attrs = tffi.new("iqvarAttrs *")
        attrs.ndim_ = 3
        attrs.dims = self.attrs["dims"]
        return bytes(tffi.buffer(attrs))

@register_op
class iqVar(Operator):
    def __init__(self, attrs={}):
        self.attrs = iqVarAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "iqVar operator must have exactly one input"

        X = inputs[0]
        x_shape = list(X.shape)
        axis = self.attrs.get("dims", 2)
        axis = axis + len(x_shape) if axis < 0 else axis
        assert len(x_shape) >= 3, "iqVar requires at least rank-3 input"
        assert axis in (len(x_shape) - 1, len(x_shape) - 2), "iqVar only supports the last two axes"
        reduction_size = x_shape[axis]
        assert is_sympy(reduction_size) or int(reduction_size) > 0, \
            "iqVar reduction axis must be non-empty"
        platform = self.attrs.get("platform")
        assert platform is None or platform in {"venus", "arcs", "venusA"}, \
            "iqVar platform must be venus, arcs, or venusA"
        assert X.dtype == np.int8, "iqVar only supports int8 input"
        assert X.zero == 0, "iqVar only supports zero point 0"
        if platform == "venus":
            assert all(dim == 1 for dim in x_shape[:-3]), \
                "iqVar on venus requires singleton dimensions before the final three dimensions"
        elif platform in {"arcs", "venusA"}:
            assert len(x_shape) == 3, f"iqVar on {platform} only supports rank-3 input"
        x_shape[axis] = 1

        # Process input scale
        scale_x = self.attrs.get("scale_x")
        assert scale_x is not None and scale_x > 0, "iqVar scale_x must be positive"
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert X.scale == int(temp), "Input scale must match attribute scale_x"

        # Process output scale
        scale_o = self.attrs.get("scale_o")
        assert scale_o is not None and scale_o > 0, "iqVar scale_o must be positive"
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        if platform == "venus":
            shift = X.scale * 2 - int(temp)
            assert 0 <= shift <= 30, "iqVar on venus requires output shift in [0, 30]"
            if not is_sympy(reduction_size):
                assert int(reduction_size) <= 23726566, \
                    "iqVar on venus reduction axis exceeds safe int64 accumulation range"

        if platform == "arcs":
            shift = X.scale * 2 - int(temp)
            assert -30 <= shift <= 30, "iqVar on arcs output shift exceeds integer runtime range"
            if not is_sympy(reduction_size):
                assert int(reduction_size) <= 131071, \
                    "iqVar on arcs reduction axis exceeds safe int32 accumulation range"

        if platform == "venusA":
            shift = X.scale * 2 - int(temp)
            assert -30 <= shift <= 30, "iqVar on venusA output shift exceeds integer runtime range"
            if not is_sympy(reduction_size):
                assert int(reduction_size) <= 131071, "iqVar on venusA reduction axis exceeds safe accumulation range"
        if platform is None:
            shift = X.scale * 2 - int(temp)
            assert -30 <= shift <= 30, "iqVar output shift exceeds supported range"

        # Create output tensor
        Y = X.clone(shape=tuple(x_shape), scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqVar operation."""
        input_data = self.inputs[0]
        output_data = self.outputs[0]
        platform = self.attrs.get("platform", "venus")
        assert platform in {"venus", "arcs", "venusA"}, \
            "iqVar platform must be venus, arcs, or venusA"
        assert input_data.mem_type == MemType.SHARE_MEM, \
            f"iqVar on {platform} requires SHARE_MEM input"
        assert output_data.mem_type == MemType.SHARE_MEM, \
            f"iqVar on {platform} requires SHARE_MEM output"
        if platform in {"arcs", "venusA"}:
            assert len(input_data.shape) == 3, f"iqVar on {platform} only supports rank-3 input"
        else:
            assert all(dim == 1 for dim in input_data.shape[:-3]), \
                "iqVar on venus requires singleton leading dimensions"
        axis = self.attrs.get("dims", len(input_data.shape) - 1)
        axis = axis + len(input_data.shape) if axis < 0 else axis
        f = input_data.shape[axis]
        shift = input_data.scale * 2 - output_data.scale
        if platform == "venus":
            assert 0 <= shift <= 30 and (is_sympy(f) or int(f) <= 23726566), \
                "iqVar on venus has unsupported shift or reduction length"
        else:
            assert -30 <= shift <= 30 and (is_sympy(f) or int(f) <= 131071), \
                f"iqVar on {platform} has unsupported shift or reduction length"
        if platform == "venus":
            workspace_size = ALIGN4(input_data.nbytes) if axis == len(input_data.shape) - 2 else 4
        elif platform == "arcs":
            workspace_size = f * 4 + 8
            if axis == len(input_data.shape) - 2:
                workspace_size += ALIGN4(input_data.nbytes)
        else:
            workspace_size = ALIGN4(f * 2) + f * 4 + 8
            if axis == len(input_data.shape) - 2:
                workspace_size += ALIGN4(input_data.nbytes)
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the iqVar operation."""
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
        flops = int(np.prod(xshape)) + int(np.prod(yshape)) * 6
        return flops

__all__ = ["iqVar"]
