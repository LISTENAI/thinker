import math
import numpy as np
from typing import List
from ...graph import Tensor
from .utils import calc_expr
from ...enum_defines import DevType, MemType
from ...xsympy import is_sympy
from .base import iqBinaryOperator, register_op, BaseLayout

@register_op
class iqAdd(iqBinaryOperator, BaseLayout):
    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 2, "iqAdd operator must have exactly two inputs"

        X1 = inputs[0]
        X2 = inputs[1]

        # Expand shapes to the same dimension
        shape1 = list(X1.shape)
        shape2 = list(X2.shape)
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2), "Shapes must have the same dimensions after expansion"

        # Check shape compatibility
        diff_count = 0
        for i in range(len(shape1)):
            if is_sympy(shape1[i]) and is_sympy(shape2[i]):
                temp_shape1 = calc_expr(str(shape1[i]), dynamic_shape)
                temp_shape2 = calc_expr(str(shape2[i]), dynamic_shape)
                if temp_shape1 != temp_shape2:
                    diff_count += 1
            elif shape1[i] != shape2[i]:
                assert shape1[i] == 1 or shape2[i] == 1, "Incompatible dimensions"
                diff_count += 1
        assert diff_count <= 1, "iqAdd does not support this type of broadcasting"

        # Process scales
        scale_x = self.attrs.get('scale_x', 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        if X1.scale != -1:
            assert X1.scale == int(temp), "Input scale must match attribute scale_x"
        else:
            X1.scale = int(temp)

        scale_y = self.attrs.get('scale_y', 1.0)
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        X2.scale = int(temp)

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        Y = X1.clone(shape=tuple(shape1), scale=temp)
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqAdd operation."""
        x1 = self.inputs[0]
        x2 = self.inputs[1]
        size = x1.nbytes
        Y = self.outputs[0]

        scale_x = self.attrs["scale_x"]
        scale_y = self.attrs["scale_y"]
        scale_o = self.attrs["scale_o"]
        platform = self.attrs.get("platform", "venus")

        workspace_size = 0

        if Y.mem_type != MemType.SHARE_MEM:
            if (scale_x != scale_o) or x1.mem_type != MemType.SHARE_MEM:
                workspace_size += size
            if (scale_y != scale_o) or x2.mem_type != MemType.SHARE_MEM:
                workspace_size += size
            if Y.mem_type != MemType.SHARE_MEM:
                workspace_size = max(workspace_size, size)
        else:
            if (scale_x != scale_o or x1.mem_type != MemType.SHARE_MEM) and (scale_y != scale_o or x2.mem_type != MemType.SHARE_MEM):
                workspace_size += size

        if platform == "venusA":
            workspace_size = min(workspace_size, 65536)

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

__all__ = ["iqAdd"]