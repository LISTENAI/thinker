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
        assert platform in ("venus", "arcs", "mars", "venusA"), \
            f"BmmInt is not supported on {platform}"
        if platform == "venusA":
            assert X1.dtype == X2.dtype and X1.dtype in (np.int8, np.int16, np.int32), \
                "BmmInt on venusA only supports matching int8/int16/int32 inputs"
        elif platform in ("arcs", "mars"):
            assert X1.dtype == X2.dtype and X1.dtype in (np.int8, np.int32), \
                f"BmmInt on {platform} only supports int8/int32 inputs"
        else:
            assert X1.dtype == X2.dtype == np.int8, "Inputs must be int8"

        assert len(x1_shape) in {2, 3}, "Input tensors must be 2D or 3D"
        assert len(x1_shape) == len(x2_shape), "BmmInt inputs must have the same rank"
        if len(x1_shape) == 3:
            batch_x1 = calc_expr(str(x1_shape[0]), dynamic_shape) if is_sympy(x1_shape[0]) else x1_shape[0]
            batch_x2 = calc_expr(str(x2_shape[0]), dynamic_shape) if is_sympy(x2_shape[0]) else x2_shape[0]
            assert batch_x1 == batch_x2, \
                f"BmmInt inputs must have the same batch size, got {x1_shape} and {x2_shape}"
        
        inner_x1 = calc_expr(str(X1.shape[-1]), dynamic_shape) if is_sympy(X1.shape[-1]) else X1.shape[-1]
        inner_x2 = calc_expr(str(X2.shape[-2]), dynamic_shape) if is_sympy(X2.shape[-2]) else X2.shape[-2]
        assert inner_x1 == inner_x2, "Matrix dimensions must match"

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
        assert scale_x + scale_y - temp <= 63, "BmmInt shift exceeds Luna limit"

        # Determine output data type
        out_bits = self.attrs.get("o_bits", 8)
        if platform in ("arcs", "mars"):
            assert out_bits in {8, 32}, \
                f"BmmInt on {platform} only supports int8 or int32 output"
        else:
            assert out_bits in {8, 16, 32}, "Output bits must be 8, 16, or 32"
        if platform == "venus":
            assert out_bits == 8, "BmmInt on venus only supports int8 output"
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

        workspace_bytes = 0
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            if int8_condition_l > 65536:
                assert ALIGN4(1) * ALIGN8(N) <= 65536, \
                    "BmmInt shared dimension exceeds the Venus row-split limit"
                split_num = 2
                while True:
                    split_M = math.ceil(M / split_num)
                    int8_condition_l_split = ALIGN4(split_M) * ALIGN8(N)
                    if int8_condition_l_split <= 65536:
                        break
                    split_num += 1

            if int8_condition_r > 32768:
                assert ALIGN8(N) * ALIGN4(1) <= 32768, \
                    "BmmInt shared dimension exceeds the Venus column-split limit"

            # Calculate workspace size
            lhs_external = self.inputs[0].mem_type != MemType.SHARE_MEM
            rhs_external = self.inputs[1].mem_type != MemType.SHARE_MEM
            out_external = self.outputs[0].mem_type != MemType.SHARE_MEM
            assert not (rhs_external and out_external and not lhs_external), \
                "BmmInt on venus cannot stage external rhs and output with SHARE_MEM lhs"
            workspace_bytes = rhs_external * N * L
            workspace_bytes += lhs_external * split_M * N
            workspace_bytes += out_external * split_M * L
        elif platform == "venusA":
            if self.outputs[0].mem_type != MemType.SHARE_MEM:
                workspace_bytes = M * L * self.outputs[0].dtype.itemsize
        else:
            if self.outputs[0].mem_type != MemType.SHARE_MEM:
                workspace_bytes = M * L * self.outputs[0].dtype.itemsize
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

        batch = xshape[0] if len(xshape) == 3 else 1
        M = xshape[-2]
        N = xshape[-1]
        L = yshape[-1]
        return int(batch * M * L * (2 * N - 1))


__all__ = ["BmmInt"]
