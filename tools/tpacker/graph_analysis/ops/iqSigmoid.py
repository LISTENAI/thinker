import math
import numpy as np
from typing import List
from ...graph import Tensor
from ...xsympy import is_sympy
from .utils import QuantType, calc_expr
from ...enum_defines import DevType, MemType
from .base import Operator, register_op

@register_op
class iqSigmoid(Operator):
    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "iqSigmoid operator must have exactly one input"
        platform = self.attrs.get("platform", "venus")

        X = inputs[0]
        if platform == "venus":
            assert X.dtype == np.int16, "input data type of iqSigmoid must be int16"
        # Process input scale
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert X.scale == int(temp), "Input scale must match attribute scale_x"

        # Process output scale
        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        # Create output tensor
        Y = X.clone(scale=int(temp), dtype=np.int8, bits=1)
        self.outputs = [Y]

        # Perform forward computation if all inputs have data
        if all(x.has_data() for x in inputs):
            self.forward()

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqSigmoid operation."""
        x = self.inputs[0]
        platform = self.attrs.get("platform", "venus")

        workspace_size = 0
        if platform in {"arcs", "venusA"}:
            workspace_size = x.nbytes * 4
        elif platform == "venus":
            assert x.mem_type == MemType.SHARE_MEM and self.outputs[0].mem_type == MemType.SHARE_MEM
            workspace_size = x.nbytes * 2 if x.scale != 11 else 0

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the iqSigmoid operation."""
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

__all__ = ["iqSigmoid"]