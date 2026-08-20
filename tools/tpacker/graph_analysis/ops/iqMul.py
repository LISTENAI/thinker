import math
import numpy as np
from typing import Any, Dict, Optional, List

from ...graph import Tensor
from ...enum_defines import DevType, MemType, ALIGN4
from .base import iqBinaryOperator, register_op

from ...xsympy import is_sympy
from .utils import QuantType, calc_expr, RoundMethod

def find_min_q(a):
    if a == 0:
        return 0, 0
    _, denom = float(a).as_integer_ratio()
    k = denom.bit_length() - 1
    return k, a * pow(2, k)

@register_op
class iqMul(iqBinaryOperator):
    def infer_tensor(self, dynamic_shape: Dict[str, int]):
        """Infer output tensor shape and data."""
        inputs = self.inputs
        assert len(inputs) == 2, "iqBinaryOperator must have exactly two inputs"
        X1, X2 = inputs
        platform = self.attrs.get("platform", "venus")
        assert platform in {"venus", "arcs", "venusA"}, \
            "iqMul platform must be venus, arcs, or venusA"
        scalar_float = X2.dtype == np.float32 and X2.shape == () and X2.has_data()
        if scalar_float:
            scalar = float(np.asarray(X2.data).item())
            assert np.isfinite(scalar), "iqMul float scalar must be finite"
            shift, new_data = find_min_q(scalar)
            info = np.iinfo(X1.dtype)
            assert info.min <= new_data <= info.max, \
                "iqMul float scalar exceeds input dtype range"
            X2.data = np.asarray(new_data, dtype=X1.dtype)
            X2.dtype = X1.dtype
            X2.bits = X1.bits
        assert X1.dtype == X2.dtype, "Input tensors must have the same data type"
        assert X1.zero == X2.zero == 0, "iqMul only supports zero point 0"

        shape1 = list(X1.shape)
        shape2 = list(X2.shape)

        # Expand to the same dimension
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2), "Input shapes must have the same dimensions after expansion"

        shape = []
        for s1, s2 in zip(shape1, shape2):
            if s1 == 1:
                shape.append(s2)
            elif s2 == 1:
                shape.append(s1)
            elif s1 == s2:
                shape.append(s1)
            elif is_sympy(s1) and is_sympy(s2):
                s1_val = calc_expr(str(s1), dynamic_shape)
                s2_val = calc_expr(str(s2), dynamic_shape)
                assert s1_val == s2_val, "Dynamic shapes must match"
                shape.append(s1_val)
            else:
                raise AttributeError("Incompatible shapes")

        for name in ("scale_x", "scale_y", "scale_o"):
            value = self.attrs.get(name, 1.0)
            assert np.isfinite(value) and value > 0, \
                f"iqMul {name} must be finite and positive"
            exponent = math.log(value, 2)
            assert abs(exponent - round(exponent)) < 1e-6, \
                f"iqMul {name} must be a power of 2"

        scale_x = self.attrs.get('scale_x', 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_x must be a power of 2"
        if self.inputs[0].scale != -1:
            assert self.inputs[0].scale == int(temp), "Input scale mismatch"
        else:
            self.inputs[0].scale = int(temp)

        scale_y = self.attrs.get('scale_y', 1.0)
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_y must be a power of 2"
        if scalar_float:
            self.attrs['scale_y'] = scale_y * 2 ** shift
            self.inputs[1].scale = int(temp+shift)
        elif self.inputs[1].scale != -1:
            assert self.inputs[1].scale == int(temp), "Input scale mismatch"
        else:
            self.inputs[1].scale = int(temp)

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_o must be a power of 2"

        o_bits = self.attrs.get("o_bits", 8)
        bits_map = {8: (np.dtype("i1"), 1), 16: (np.dtype("i2"), 2), 32: (np.dtype("i4"), 4)}
        assert o_bits in bits_map, "iqMul o_bits must be 8, 16, or 32"
        data_type, bits = bits_map[o_bits]

        shift = int(X1.scale) + int(X2.scale) - int(temp)
        assert 0 <= shift <= 63, f"iqMul on {platform} requires Luna shift in [0, 63]"
        is_scalar = len(X2.shape) == 0
        is_broadcast_h1w1 = (
            len(X1.shape) == len(X2.shape) == 4 and
            X1.shape[1] == X2.shape[1] and X2.shape[2:] == (1, 1) and
            X2.shape[0] in (1, X1.shape[0]) and tuple(X1.shape) != tuple(X2.shape)
        )
        if platform == "venus":
            assert X1.dtype in (np.int8, np.int16, np.int32), "iqMul on venus only supports integer inputs"
            assert o_bits == X1.dtype.itemsize * 8, "iqMul on venus output dtype must match inputs"
            is_broadcast_h1w1 = (
                len(X1.shape) == len(X2.shape) == 4 and
                X1.shape[0] == X2.shape[0] == 1 and
                X1.shape[1] == X2.shape[1] and X2.shape[2:] == (1, 1)
            )
            assert is_scalar or tuple(X1.shape) == tuple(X2.shape) or is_broadcast_h1w1, \
                "iqMul on venus only supports scalar, same-shape, or 1CHW x 1C11 broadcast"
            assert tuple(shape) == tuple(X1.shape), "iqMul on venus output must match lhs shape"
        elif platform == "arcs":
            is_broadcast_h1w1 = (len(X1.shape) == len(X2.shape) == 4 and X1.shape[1] == X2.shape[1]
                                   and X2.shape[2:] == (1, 1) and X2.shape[0] in (1, X1.shape[0]))
            if is_broadcast_h1w1:
                assert X1.shape[2] * X1.shape[3] <= 16384, "iqMul on arcs broadcast exceeds Luna matmul limit"
            assert is_scalar or tuple(X1.shape) == tuple(X2.shape) or is_broadcast_h1w1, "iqMul on arcs only supports scalar, same-shape, or NCHW x NC11 broadcast"
            assert tuple(shape) == tuple(X1.shape), "iqMul on arcs output must match lhs shape"
            assert X1.dtype in (np.int8, np.int32), "iqMul on arcs only supports int8 or int32"
            assert o_bits == X1.dtype.itemsize * 8, "iqMul on arcs output dtype must match inputs"
        if platform == "venusA":
            assert is_scalar or tuple(X1.shape) == tuple(X2.shape) or (
                X1.dtype == np.int8 and is_broadcast_h1w1
            ), "iqMul on venusA only supports scalar, same-shape, or int8 NC11 broadcast"
            if X1.dtype == np.int8:
                assert o_bits == 8, "iqMul on venusA int8 input only supports int8 output"
            elif X1.dtype == np.int16:
                assert o_bits in (8, 16), "iqMul on venusA int16 input only supports int8/int16 output"
            elif X1.dtype == np.int32:
                assert o_bits == 32, "iqMul on venusA int32 input only supports int32 output"
            else:
                raise AssertionError("iqMul on venusA only supports int8/int16/int32 input")
        Y = X1.clone(shape=tuple(shape), dtype=data_type, bits=bits,
                     scale=int(temp), zero=0)

        self.outputs = [Y]
        if all(x.has_data() for x in inputs):
            self.forward()

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqMul operation."""
        input1 = self.inputs[0]
        input2 = self.inputs[1]
        output = self.outputs[0]
        platform = self.attrs.get("platform", "venus")
        workspace_size = 0

        def workspace_need(elem_size, input_bytes, output_bytes, x1_in_psram, x2_in_psram, y_in_psram):
            size = 0
            if x1_in_psram:
                size += ALIGN4(elem_size * input_bytes)
            if x2_in_psram:
                size += ALIGN4(elem_size * input_bytes)
            if y_in_psram:
                size += elem_size * output_bytes
            return size

        def split_workspace(full_workspace, min_workspace):
            if full_workspace == 0:
                return 0
            return max(min(full_workspace, 65536), min_workspace)

        def broadcast_workspace_need(cur_c, hw, x1_in_psram, x2_in_psram, y_in_psram):
            cur_size = cur_c * hw
            size = ALIGN4(hw) + ALIGN4(cur_size)
            if x2_in_psram:
                size += ALIGN4(cur_c)
            if x1_in_psram:
                size += ALIGN4(cur_size)
            if y_in_psram:
                size += cur_size
            return size

        is_broadcast_h1w1 = (
            len(input1.shape) == 4 and
            len(input2.shape) == 4 and
            input1.shape[1] == input2.shape[1] and
            input2.shape[2] == 1 and
            input2.shape[3] == 1 and
            input2.shape[0] in (1, input1.shape[0])
        )

        if platform == "venusA":
            x1_in_psram = input1.mem_type != MemType.SHARE_MEM
            x2_in_psram = input2.mem_type != MemType.SHARE_MEM
            y_in_psram = output.mem_type != MemType.SHARE_MEM

            if is_broadcast_h1w1:
                hw = input1.shape[2] * input1.shape[3]
                full_workspace = broadcast_workspace_need(input1.shape[1], hw,
                                                          x1_in_psram, x2_in_psram, y_in_psram)
                min_workspace = broadcast_workspace_need(1, hw,
                                                         x1_in_psram, x2_in_psram, y_in_psram)
                workspace_size = split_workspace(full_workspace, min_workspace)
            else:
                input_bytes = input1.dtype.itemsize
                output_bytes = output.dtype.itemsize
                x2_need_workspace = False if len(input2.shape) == 0 else x2_in_psram
                full_workspace = workspace_need(output.size, input_bytes, output_bytes,
                                                x1_in_psram, x2_need_workspace, y_in_psram)
                min_workspace = workspace_need(1, input_bytes, output_bytes,
                                               x1_in_psram, x2_need_workspace, y_in_psram)
                workspace_size = split_workspace(full_workspace, min_workspace)
        elif platform == "arcs":
            x1_in_psram = input1.mem_type != MemType.SHARE_MEM
            x2_in_psram = input2.mem_type != MemType.SHARE_MEM
            y_in_psram = output.mem_type != MemType.SHARE_MEM

            if is_broadcast_h1w1:
                hw = input1.shape[2] * input1.shape[3]
                full_workspace = broadcast_workspace_need(input1.shape[1], hw,
                                                          x1_in_psram, x2_in_psram,
                                                          y_in_psram)
                min_workspace = broadcast_workspace_need(1, hw, x1_in_psram,
                                                         x2_in_psram, y_in_psram)
                workspace_size = split_workspace(full_workspace, min_workspace)
            else:
                x2_need_workspace = False if len(input2.shape) == 0 else x2_in_psram
                elem_bytes = input1.dtype.itemsize
                full_workspace = workspace_need(output.size, elem_bytes, elem_bytes,
                                                x1_in_psram, x2_need_workspace,
                                                y_in_psram)
                min_workspace = workspace_need(1, elem_bytes, elem_bytes,
                                               x1_in_psram, x2_need_workspace,
                                               y_in_psram)
                workspace_size = split_workspace(full_workspace, min_workspace)
        elif platform == "venus":
            x1_in_psram = input1.mem_type != MemType.SHARE_MEM
            x2_in_psram = input2.mem_type != MemType.SHARE_MEM
            y_in_psram = output.mem_type != MemType.SHARE_MEM
            if is_broadcast_h1w1:
                hw = input1.shape[2] * input1.shape[3]
                data_size = input1.shape[1] * hw
                workspace_size = ALIGN4(hw) + ALIGN4(data_size)
                if x2_in_psram:
                    workspace_size += ALIGN4(input1.shape[1])
                if x1_in_psram:
                    workspace_size += ALIGN4(data_size)
                if y_in_psram:
                    workspace_size += data_size
            else:
                if x1_in_psram:
                    workspace_size += ALIGN4(input1.nbytes)
                if len(input2.shape) != 0 and x2_in_psram:
                    workspace_size += ALIGN4(input2.nbytes)
                if y_in_psram:
                    workspace_size += output.nbytes

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

__all__ = ["iqMul"]
