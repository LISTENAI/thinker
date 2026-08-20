import math
import numpy as np
from typing import List
from ...graph import Tensor
from ...xsympy import is_sympy
from .utils import QuantType, calc_expr
from .base import Operator, register_op
from ...enum_defines import DevType, MemType, ALIGN4


@register_op
class iqSigmoid(Operator):
    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "iqSigmoid operator must have exactly one input"
        platform = self.attrs.get("platform")
        assert platform is None or platform in {"venus", "arcs", "venusA"}, \
            "iqSigmoid platform must be venus, arcs, or venusA"

        X = inputs[0]
        if platform == "venus":
            assert X.dtype == np.int16, "input data type of iqSigmoid must be int16 for venus"
        elif platform == "arcs":
            assert X.dtype in (np.int8, np.int32), "iqSigmoid on arcs expects int8 or int32 Q4.27 input"
        elif platform == 'venusA':
            assert X.dtype in (np.int8, np.int16, np.int32), \
                "iqSigmoid on venusA expects int8, int16, or int32 input"
        else:
            assert X.dtype in (np.int8, np.int16, np.int32), \
                "iqSigmoid expects int8, int16, or int32 input"
        assert X.zero == 0, "iqSigmoid only supports zero point 0"

        # Process input scale
        scale_x = self.attrs.get("scale_x")
        assert scale_x is not None and scale_x > 0, "iqSigmoid scale_x must be positive"
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert X.scale == int(temp), "Input scale must match attribute scale_x"

        # Process output scale
        scale_o = self.attrs.get("scale_o")
        assert scale_o is not None and scale_o > 0, "iqSigmoid scale_o must be positive"
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        if platform == "venus":
            assert int(temp) == 7, "iqSigmoid on venus only supports fixed Q7 output"
            assert -63 <= 11 - X.scale <= 14, "iqSigmoid on venus Q11 conversion shift is unsupported"

        if platform == 'venusA':
            shift = 27 - X.scale
            assert -63 <= shift <= 30, "iqSigmoid on venusA input shift exceeds Luna/scalar limits"
            assert int(temp) == 7, "iqSigmoid on venusA int8 output scale must be Q7"
        elif platform == "arcs":
            shift = 27 - X.scale
            if X.dtype == np.int32:
                assert X.scale == 27, "iqSigmoid on arcs int32 input must be Q4.27"
            else:
                assert 0 <= shift <= 30, "iqSigmoid on arcs int8 conversion shift exceeds Luna/scalar limits"
            assert int(temp) == 7, "iqSigmoid on arcs int8 output scale must be Q7"
        elif platform is None:
            assert int(temp) == 7, "iqSigmoid output scale must be Q7"

        out_bits = self.attrs.get('o_bits')
        if out_bits != None:
            assert out_bits == 8, "output type must be int8"

        # Create output tensor
        Y = X.clone(scale=int(temp), dtype=np.int8, bits=1)
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqSigmoid operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        data_size = np.prod(X.shape)
        platform = self.attrs.get("platform", "venus")
        assert platform in {"venus", "arcs", "venusA"}, \
            "iqSigmoid platform must be venus, arcs, or venusA"
        assert Y.dtype == np.int8 and Y.scale == 7 and Y.zero == 0
        assert X.mem_type in (MemType.SHARE_MEM, MemType.PSRAM) and \
               Y.mem_type in (MemType.SHARE_MEM, MemType.PSRAM), \
            "iqSigmoid tensors must be in SHARE_MEM or PSRAM"

        workspace_size = 0
        if platform == "venus":
            assert X.dtype == np.int16, "iqSigmoid on venus expects int16 input"
            assert -63 <= 11 - X.scale <= 14, \
                "iqSigmoid on venus Q11 conversion shift is unsupported"
            assert X.mem_type == MemType.SHARE_MEM, "input mem_type of iqSigmoid must be share memory"
            assert Y.mem_type == MemType.SHARE_MEM, "output mem_type of iqSigmoid must be share memory"
            workspace_size = data_size * 2 if X.scale != 11 else 0
        elif platform == "arcs":
            assert X.dtype in (np.int8, np.int32), \
                "iqSigmoid on arcs expects int8 or int32 input"
            if X.dtype == np.int32:
                assert X.scale == 27, "iqSigmoid on arcs int32 input must be Q4.27"
            else:
                assert 0 <= 27 - X.scale <= 30, \
                    "iqSigmoid on arcs int8 conversion shift is unsupported"
            if X.dtype == np.int8:
                workspace_size = (ALIGN4(data_size) if X.mem_type != MemType.SHARE_MEM else 0) + data_size * 4
                workspace_size = ALIGN4(workspace_size)
                if Y.mem_type != MemType.SHARE_MEM:
                    workspace_size += data_size
            elif X.dtype == np.int32:
                need_i32_buffer = X.mem_type != MemType.SHARE_MEM or X.scale != 27
                workspace_size = data_size * 4 if need_i32_buffer else 0
                workspace_size = ALIGN4(workspace_size)
                if Y.mem_type != MemType.SHARE_MEM:
                    workspace_size += data_size
            workspace_size = min(workspace_size, 65536)
        else:
            assert X.dtype in (np.int8, np.int16, np.int32), \
                "iqSigmoid on venusA expects int8, int16, or int32 input"
            assert -63 <= 27 - X.scale <= 30, \
                "iqSigmoid on venusA input shift is unsupported"
            x_bytes = data_size * X.dtype.itemsize
            input_bytes = ALIGN4(x_bytes) if X.mem_type != MemType.SHARE_MEM else 0
            if X.dtype == np.int8:
                workspace_size = input_bytes + ALIGN4(data_size * 2) + data_size * 4
            elif X.dtype == np.int16:
                workspace_size = input_bytes + data_size * 4
            else:
                workspace_size = data_size * 4 if X.mem_type != MemType.SHARE_MEM or X.scale != 27 else 0
            workspace_size = ALIGN4(workspace_size)
            if Y.mem_type != MemType.SHARE_MEM:
                workspace_size += data_size
            if workspace_size:
                workspace_size = min(max(workspace_size, 12), 65536)
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
