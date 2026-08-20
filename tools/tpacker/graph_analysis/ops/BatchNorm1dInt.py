import math
import numpy as np
from typing import List

from ...graph import Tensor
from .utils import calc_expr
from ...xsympy import is_sympy
from .base import Operator, register_op
from ...enum_defines import DevType, MemType, ALIGN4


@register_op
class BatchNorm1dInt(Operator):
    """Batch normalization for quantized 2D convolution."""
    
    def infer_tensor(self, dynamic_shape):
        """Infer tensor shapes and create output tensor."""
        # Check input tensor dimensions and data types
        platform = self.attrs.get("platform")
        assert platform in ("venus", "arcs", "venusA"), \
            f"BatchNorm1dInt is not supported on {platform}"
        assert len(self.inputs) == 3, \
            "BatchNorm1dInt requires input, weight, and bias"
        X = self.inputs[0]
        assert len(X.shape) == 3, "Only 3D tensors are supported"
        assert all(is_sympy(dim) or dim > 0 for dim in X.shape), \
            "BatchNorm1dInt dimensions must be positive"
        assert X.dtype == np.int8, "Input must be int8"
        assert self.inputs[1].dtype == np.int8, "Weight must be int8"
        assert self.inputs[2].dtype == np.int32, "Bias must be int32"
        assert self.inputs[1].size == X.shape[1], \
            "Weight size must match input channels"
        assert self.inputs[2].size == X.shape[1], \
            "Bias size must match input channels"

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

        shift = X.scale + self.inputs[1].scale - int(temp)
        assert 0 <= shift <= 63, \
            f"BatchNorm1dInt on {platform} requires Luna shift in [0, 63]"

        # Create output tensor
        Y = X.clone(scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate and return workspace tensor."""
        X = self.inputs[0]
        assert X.mem_type == MemType.SHARE_MEM, \
            "BatchNorm1dInt requires SHARE_MEM input"
        assert self.outputs[0].mem_type == MemType.SHARE_MEM, \
            "BatchNorm1dInt requires SHARE_MEM output"
        workspace_factor = 6 if self.attrs.get("platform") == "venusA" else 4
        workspace_bytes = ALIGN4(X.shape[2] * workspace_factor)
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


__all__ = ["BatchNorm1dInt"]
