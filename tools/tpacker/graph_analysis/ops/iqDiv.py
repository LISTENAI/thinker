import math
import numpy as np

from ...enum_defines import MemType
from .base import iqBinaryOperator, register_op

@register_op
class iqDiv(iqBinaryOperator):
    def infer_tensor(self, dynamic_shape):
        platform = self.attrs.get("platform", "venus")
        assert platform in {"venus", "arcs", "venusA"}, \
            "iqDiv platform must be venus, arcs, or venusA"
        lhs, rhs = self.inputs
        assert len(self.inputs) == 2, "iqDiv operator must have exactly two inputs"
        for name in ("scale_x", "scale_y", "scale_o"):
            value = self.attrs.get(name, 1.0)
            assert np.isfinite(value) and value > 0, \
                f"iqDiv {name} must be finite and positive"
            exponent = math.log(value, 2)
            assert abs(exponent - round(exponent)) < 0.000001, \
                f"iqDiv {name} must be a power of 2"
        assert lhs.zero == 0 and rhs.zero == 0, "iqDiv only supports zero point 0"
        rhs_exponent = int(round(math.log(self.attrs.get("scale_y", 1.0), 2)))
        if rhs.scale != -1:
            assert rhs.scale == rhs_exponent, "Input scale mismatch"
        super().infer_tensor(dynamic_shape)
        out = self.outputs[0]
        assert lhs.mem_type == MemType.SHARE_MEM and rhs.mem_type == MemType.SHARE_MEM, f"iqDiv on {platform} requires SHARE_MEM inputs"
        assert out.mem_type == MemType.SHARE_MEM, f"iqDiv on {platform} requires SHARE_MEM output"
        rhs_is_scalar = len(rhs.shape) == 0 or tuple(rhs.shape) == ()
        assert rhs_is_scalar or tuple(rhs.shape) == tuple(lhs.shape), f"iqDiv on {platform} does not support broadcasting"
        if rhs_is_scalar:
            assert lhs.dtype == rhs.dtype, f"iqDiv scalar path on {platform} requires matching input dtypes"
            supported = (np.int32,) if platform == "venusA" else ((np.int8, np.int32) if platform == "arcs" else (np.int8, np.int16, np.int32))
            assert lhs.dtype in supported, f"iqDiv scalar path on {platform} has unsupported input dtype"
            assert out.dtype in supported, f"iqDiv scalar path on {platform} has unsupported output dtype"
        else:
            assert lhs.dtype == rhs.dtype == out.dtype == np.int32, f"iqDiv vector path on {platform} only supports int32 tensors"
        assert out.zero == 0, "iqDiv output zero point must be 0"
        out.scale = int(out.scale)
        shift = int(out.scale) - (int(lhs.scale) - int(rhs.scale))
        assert 0 <= shift <= 63, f"iqDiv on {platform} requires Luna shift in [0, 63]"
        if rhs.has_data():
            assert np.all(rhs.data != 0), f"iqDiv on {platform} divisor must not contain zero"
            if rhs_is_scalar:
                scalar = int(rhs.data)
                assert scalar > 0 and scalar & (scalar - 1) == 0, f"iqDiv on {platform} scalar divisor must be a positive power of two"
                lshift = shift - (scalar.bit_length() - 1)
                max_lshift = {1: 6, 2: 14, 4: 30}[np.dtype(lhs.dtype).itemsize]
                assert -63 <= lshift <= max_lshift, f"iqDiv on {platform} scalar scale shift is unsupported"

__all__ = ["iqDiv"]
