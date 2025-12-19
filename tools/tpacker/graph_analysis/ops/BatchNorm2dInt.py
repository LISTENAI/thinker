import math
import numpy as np
from typing import List

from ...graph import Tensor
from .utils import calc_expr
from ...xsympy import is_sympy
from .base import Operator, register_op
from ...enum_defines import DevType, MemType


@register_op
class BatchNorm2dInt(Operator):
    """Batch normalization for quantized 2D convolution."""
    
    def infer_tensor(self, dynamic_shape):
        """Infer tensor shapes and create output tensor."""
        # Check input tensor dimensions and data types
        X = self.inputs[0]
        assert len(X.shape) == 4, "Only 4D tensors are supported"
        assert X.dtype == np.int8, "Input must be int8"
        assert self.inputs[1].dtype == np.int8, "Weight must be int8"
        assert self.inputs[2].dtype == np.int32, "Bias must be int32"

        # Validate and set scales
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 1e-6, "Input scale must be power of 2"
        X.scale = int(temp)

        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w, 2)
        assert abs(temp - int(temp)) < 1e-6, "Weight scale must be power of 2"
        self.inputs[1].scale = int(temp)

        self.inputs[2].scale = X.scale + self.inputs[1].scale

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 1e-6, "Output scale must be power of 2"

        # Create output tensor
        Y = X.clone(scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate and return workspace tensor."""
        X = self.inputs[0]
        workspace_bytes = X.shape[2] * X.shape[3] * 4
        return [Tensor.from_shape([workspace_bytes], np.int8, MemType.SHARE_MEM)]

    def flops_counter(self, dynamic_shape) -> int:
        """Count floating-point operations."""
        X = self.inputs[0]
        Y = self.outputs[0]

        # Handle symbolic shapes
        xshape = [calc_expr(str(s), dynamic_shape) if is_sympy(s) else s for s in X.shape]
        yshape = [calc_expr(str(s), dynamic_shape) if is_sympy(s) else s for s in Y.shape]

        # Calculate FLOPs (2 operations per output element)
        output_dims = yshape[1:]
        return int(np.prod(output_dims)) * 2


__all__ = ["BatchNorm2dInt"]