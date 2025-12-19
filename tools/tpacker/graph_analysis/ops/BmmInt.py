import math
import numpy as np
from typing import List

from ...graph import Tensor
from .utils import calc_expr
from ...xsympy import is_sympy
from .base import iqBinaryOperator, register_op
from ...enum_defines import DevType, MemType, ALIGN4, ALIGN8


@register_op
class BmmInt(iqBinaryOperator):
    """Batch matrix multiplication for quantized integers."""
    
    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape and data type."""
        inputs = self.inputs
        assert len(inputs) == 2, "BmmInt requires exactly two input tensors"

        X1 = inputs[0]
        X2 = inputs[1]
        x1_shape = list(X1.shape)
        x2_shape = list(X2.shape)

        platform = self.attrs.get("platform", "venus")
        if platform == "venusA":
            assert X1.dtype == X2.dtype, "Input tensors must have the same data type"
        else:
            assert X1.dtype == X2.dtype == np.int8, "Inputs must be int8"

        assert len(x1_shape) in {2, 3}, "Input tensors must be 2D or 3D"
        
        if is_sympy(X1.shape[-1]) and is_sympy(X2.shape[-2]):
            assert calc_expr(str(X1.shape[-1]), dynamic_shape) == calc_expr(str(X2.shape[-2]), dynamic_shape), "Matrix dimensions must match"
        else:
            assert X1.shape[-1] == X2.shape[-2], "Matrix dimensions must match"

        # Validate and set scales
        scale_x = self.attrs.get("scale_x", 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 1e-6, "Input scale must be power of 2"
        if X1.scale != -1:
            assert X1.scale == int(temp), "Scale mismatch"
        else:
            X1.scale = int(temp)
        scale_x = int(temp)

        scale_y = self.attrs.get("scale_y", 1.0)
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 1e-6, "Weight scale must be power of 2"
        if X2.scale != -1:
            assert X2.scale == int(temp), "Scale mismatch"
        else:
            X2.scale = int(temp)
        scale_y = int(temp)

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 1e-6, "Output scale must be power of 2"
        assert scale_x + scale_y - temp >= 0, "BmmInt does not support left shift"

        # Determine output data type
        out_bits = self.attrs.get("o_bits", 8)
        assert out_bits in {8, 16, 32}, "Output bits must be 8, 16, or 32"
        dtype = np.int8 if out_bits == 8 else np.int16 if out_bits == 16 else np.int32

        # Create output tensor
        shape = list(x1_shape)
        shape[-1] = x2_shape[-1]
        Y = Tensor(shape=shape, dtype=dtype, scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate and return workspace tensor."""
        M = self.inputs[0].shape[-2]
        N = self.inputs[0].shape[-1]
        L = self.inputs[1].shape[-1]

        int8_condition_l = ALIGN4(M) * ALIGN8(N)
        int8_condition_r = ALIGN8(N) * ALIGN4(L)
        split_M = M

        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            if int8_condition_l > 65536:
                split_num = 2
                while True:
                    split_M = math.ceil(M / split_num)
                    int8_condition_l_split = ALIGN4(split_M) * ALIGN8(N)
                    if int8_condition_l_split <= 65536:
                        break
                    split_num += 1

            assert int8_condition_r <= 32768, "Input2 size must not exceed 32KB"

            # Calculate workspace size
            workspace_bytes = 0
            if self.inputs[0].mem_type != MemType.SHARE_MEM and self.outputs[0].mem_type != MemType.SHARE_MEM:
                workspace_bytes = split_M * max(N, L) + split_M * L * 4
            elif self.inputs[0].mem_type != MemType.SHARE_MEM:
                workspace_bytes = split_M * N
            elif self.outputs[0].mem_type != MemType.SHARE_MEM:
                workspace_bytes = split_M * L

            if self.inputs[1].mem_type != MemType.SHARE_MEM:
                workspace_bytes += N * L

            if workspace_bytes != 0:
                return [Tensor.from_shape([workspace_bytes], np.int8, MemType.SHARE_MEM)]
        return []

    def flops_counter(self, dynamic_shape) -> int:
        """Count floating-point operations."""
        X = self.inputs[0]
        Y = self.outputs[0]

        # Handle symbolic shapes
        xshape = [calc_expr(str(s), dynamic_shape) if is_sympy(s) else s for s in X.shape]
        yshape = [calc_expr(str(s), dynamic_shape) if is_sympy(s) else s for s in Y.shape]

        # Calculate FLOPs
        output_dims = yshape[1:]
        return int(np.prod(output_dims))


__all__ = ["BmmInt"]