import math
import numpy as np
from ...resource_packer._type._ctype import tffi
from ...graph import Tensor
from ...enum_defines import DevType, MemType, ALIGN4, ALIGN8
from ...xsympy import is_sympy
from .utils import QuantType, calc_expr
from .base import Operator, OperatorAttrs, register_op

class GRUIntAttrs(OperatorAttrs):
    def normalize(self) -> None:
        self.attrs["platform"] = self.attrs.get("platform", "venus")

    def checkparams(self) -> None:
        """Check if all required parameters are present."""
        required_attrs = ["scale_x", "scale_iw", "scale_hw", "scale_h", "scale_o", "batch_first", "hidden_size", "input_size", "go_forward"]
        for attr in required_attrs:
            assert attr in self.attrs, f"Missing required attribute: {attr}"
        assert self.attrs["platform"] in {"venus", "arcs", "venusA"}, \
            "GRUInt platform must be venus, arcs, or venusA"
        assert 0 < self.attrs["hidden_size"] <= 65535, \
            "GRUInt hidden_size must be in [1, 65535]"
        assert 0 < self.attrs["input_size"] <= 65535, \
            "GRUInt input_size must be in [1, 65535]"
        assert self.attrs["batch_first"] in (0, 1), \
            "GRUInt batch_first must be 0 or 1"
        assert self.attrs["go_forward"] in (0, 1), \
            "GRUInt go_forward must be 0 or 1"
        for name in ("scale_x", "scale_iw", "scale_hw", "scale_h", "scale_o"):
            value = self.attrs[name]
            assert np.isfinite(value) and value > 0, \
                f"GRUInt {name} must be finite and positive"
            exponent = math.log(value, 2)
            assert abs(exponent - round(exponent)) < 0.000001, \
                f"GRUInt {name} must be a power of 2"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the GRUInt operation."""
        attrs = tffi.new("GRUIntAttrs *")
        attrs.direction = self.attrs["go_forward"]
        attrs.hidden_size = self.attrs["hidden_size"]
        attrs.input_size = self.attrs["input_size"]
        attrs.layout = self.attrs.get("batch_first", 0)
        return bytes(tffi.buffer(attrs))

@register_op
class GRUInt(Operator):
    def __init__(self, attrs={}):
        """Initialize the GRUInt operator with given attributes."""
        self.attrs = GRUIntAttrs(attrs)
        self.weight_index = 1

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) in (5, 6), "GRUInt requires five or six inputs"
        # inputs ==6; input,hx iw,hw,ib,hb
        if len(inputs) == 6:
            self.weight_index = 2
        X = inputs[0]  # Input tensor
        i2h_w = inputs[self.weight_index]  # Input-to-hidden weights
        h2h_w = inputs[self.weight_index + 1]  # Hidden-to-hidden weights
        i2h_bias = inputs[self.weight_index + 2]
        h2h_bias = inputs[self.weight_index + 3]
        platform = self.attrs.get("platform", "venus")

        # Check data type consistency
        assert X.dtype == i2h_w.dtype == h2h_w.dtype, "Input and weight data types must be consistent"
        if platform in {"venus", "arcs", "venusA"}:
            assert X.dtype == np.int8, "GRUInt on venusA runtime only supports int8 input/weights"
        assert X.zero == i2h_w.zero == h2h_w.zero == 0, \
            "GRUInt only supports zero point 0"

        # Validate tensor dimensions
        assert len(X.shape) == 3
        assert len(i2h_w.shape) in (2, 3)
        assert len(h2h_w.shape) in (2, 3)
        if platform == "venus":
            assert len(inputs) in (5, 6), "GRUInt on venus expects optional history plus four parameters"
            assert self.attrs["hidden_size"] > 0 and self.attrs["input_size"] > 0, \
                "GRUInt on venus requires positive input and hidden sizes"
            assert self.attrs["batch_first"] in (0, 1), "GRUInt on venus layout must be 0 or 1"
            assert self.attrs["go_forward"] in (0, 1), "GRUInt on venus direction must be 0 or 1"
            assert i2h_w.shape == (self.attrs["hidden_size"] * 3, self.attrs["input_size"]), \
                "GRUInt on venus i2h weight shape mismatch"
            assert h2h_w.shape == (self.attrs["hidden_size"] * 3, self.attrs["hidden_size"]), \
                "GRUInt on venus h2h weight shape mismatch"
            assert i2h_bias.dtype == h2h_bias.dtype == np.int32, "GRUInt on venus requires int32 biases"
            assert i2h_bias.shape == h2h_bias.shape == (self.attrs["hidden_size"] * 3,), \
                "GRUInt on venus bias shape mismatch"
            assert X.shape[2] == self.attrs["input_size"], "GRUInt on venus input_size mismatch"
            batch = X.shape[0] if self.attrs["batch_first"] else X.shape[1]
            assert batch == 1, "GRUInt on venus only supports batch size 1"
            if len(inputs) == 6:
                history_h = inputs[1]
                if history_h.dtype != np.int8:
                    history_h.dtype = np.int8
                assert history_h.shape == (1, batch, self.attrs["hidden_size"]), "GRUInt on venus history shape mismatch"
        if platform == "arcs":
            assert len(inputs) == 6, "GRUInt on arcs requires input, history, two weights, and two biases"
            assert self.attrs["batch_first"] == 0, "GRUInt on arcs only supports batch_first=0"
            assert self.attrs["go_forward"] in (0, 1), "GRUInt on arcs direction must be 0 or 1"
            assert len(i2h_w.shape) == len(h2h_w.shape) == 2, "GRUInt on arcs requires rank-2 weights"
            assert i2h_w.shape == (self.attrs["hidden_size"] * 3, self.attrs["input_size"]), "GRUInt on arcs i2h weight shape mismatch"
            assert h2h_w.shape == (self.attrs["hidden_size"] * 3, self.attrs["hidden_size"]), "GRUInt on arcs h2h weight shape mismatch"
            assert i2h_bias.dtype == h2h_bias.dtype == np.int32, "GRUInt on arcs requires int32 biases"
            assert i2h_bias.shape == h2h_bias.shape == (self.attrs["hidden_size"] * 3,), "GRUInt on arcs bias shape mismatch"
            assert X.shape[2] == self.attrs["input_size"], "GRUInt on arcs input_size mismatch"
            assert X.shape[1] == 1, "GRUInt on arcs only supports batch size 1"
            history_h = inputs[1]
            if len(history_h.shape) != 0:
                assert history_h.dtype == np.int8, "GRUInt on arcs history must be int8"
                assert history_h.shape == (1, X.shape[1], self.attrs["hidden_size"]), "GRUInt on arcs history shape mismatch"
        if platform == "venusA":
            assert len(inputs) in (5, 6), "GRUInt on venusA requires input, optional history, two weights, and two biases"
            assert self.attrs["batch_first"] in (0, 1), "GRUInt on venusA layout must be 0 or 1"
            assert self.attrs["go_forward"] in (0, 1), "GRUInt on venusA direction must be 0 or 1"
            assert len(i2h_w.shape) == len(h2h_w.shape) == 2, "GRUInt on venusA requires rank-2 weights"
            assert i2h_w.shape == (self.attrs["hidden_size"] * 3, self.attrs["input_size"]), "GRUInt on venusA i2h weight shape mismatch"
            assert h2h_w.shape == (self.attrs["hidden_size"] * 3, self.attrs["hidden_size"]), "GRUInt on venusA h2h weight shape mismatch"
            assert i2h_bias.dtype == h2h_bias.dtype == np.int32, "GRUInt on venusA requires int32 biases"
            assert i2h_bias.shape == h2h_bias.shape == (self.attrs["hidden_size"] * 3,), "GRUInt on venusA bias shape mismatch"
            assert X.shape[2] == self.attrs["input_size"], "GRUInt on venusA input_size mismatch"
            history_h = inputs[1]
            if len(history_h.shape) != 0:
                if history_h.dtype != np.int8:
                    history_h.dtype = np.int8
                batch = X.shape[0] if self.attrs["batch_first"] else X.shape[1]
                assert history_h.shape == (1, batch, self.attrs["hidden_size"]), "GRUInt on venusA history shape mismatch"

        # Process scales
        scale_x = self.attrs["scale_x"]
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        assert self.inputs[0].scale == int(temp), "Input scale must match attribute scale_x"

        scale_iw = self.attrs["scale_iw"]
        temp = math.log(scale_iw, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        self.inputs[self.weight_index].scale = int(temp)

        scale_hw = self.attrs["scale_hw"]
        temp = math.log(scale_hw, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        self.inputs[self.weight_index + 1].scale = int(temp)

        scale_h = self.attrs["scale_h"]
        temp_h = math.log(scale_h, 2)
        assert abs(temp_h - int(temp_h)) < 0.000001, "Scale must be a power of 2"
        if len(inputs) == 6 and len(inputs[1].shape) != 0:
            assert inputs[1].zero == 0, "GRUInt history zero point must be 0"
            assert inputs[1].scale == int(temp_h) or inputs[1].scale == -1.0, \
                "GRUInt history scale must match scale_h"
            inputs[1].scale = int(temp_h)
        i2h_bias.scale = int(math.log(scale_x, 2)) + int(math.log(scale_iw, 2))
        h2h_bias.scale = int(temp_h) + int(math.log(scale_hw, 2))
        assert i2h_bias.zero == h2h_bias.zero == 0, \
            "GRUInt biases must use zero point 0"

        # Determine output shape based on layout
        layout = self.attrs.get("batch_first", 0)
        if layout == 0:
            T = X.shape[0]
            B = X.shape[1]
            yshape = [T, B, self.attrs["hidden_size"]]
        else:
            B = X.shape[0]
            T = X.shape[1]
            yshape = [B, T, self.attrs["hidden_size"]]

        # Process output scale
        scale_o = self.attrs["scale_o"]
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        if platform == "arcs":
            q_ib = int(math.log(scale_x, 2)) + int(math.log(scale_iw, 2))
            q_hb = int(temp_h) + int(math.log(scale_hw, 2))
            assert -63 <= 27 - q_ib <= 30, "GRUInt on arcs input gate shift is unsupported"
            assert -63 <= 27 - q_hb <= 30, "GRUInt on arcs hidden gate shift is unsupported"
            assert -63 <= int(temp_h) - 31 <= 30, "GRUInt on arcs hidden requant shift is unsupported"
            assert int(temp) == int(temp_h), "GRUInt on arcs output scale must equal hidden scale"
        if platform == "venusA":
            q_ib = int(math.log(scale_x, 2)) + int(math.log(scale_iw, 2))
            q_hb = int(temp_h) + int(math.log(scale_hw, 2))
            max_b_q = max(q_ib, q_hb)
            assert abs(q_ib - q_hb) <= 30, "GRUInt on venusA bias alignment left shift exceeds scalar range"
            if max_b_q < 27:
                assert 27 - max_b_q <= 30, "GRUInt on venusA activation left shift exceeds scalar range"
            else:
                assert max_b_q - 27 <= 63, "GRUInt on venusA activation right shift exceeds Luna limit"
            shift_h = int(temp_h) - 31
            assert -63 <= shift_h <= 30, "GRUInt on venusA hidden requant shift exceeds Luna/scalar limits"
            assert int(temp) == int(temp_h), \
                "GRUInt on venusA output scale must equal hidden scale"
        if platform == "venus":
            q_ib = int(math.log(scale_x, 2)) + int(math.log(scale_iw, 2))
            q_hb = int(temp_h) + int(math.log(scale_hw, 2))
            assert 0 <= q_ib - 11 <= 63, "GRUInt on venus input gate shift is unsupported"
            assert 0 <= q_hb - 11 <= 63, "GRUInt on venus hidden gate shift is unsupported"
            assert 0 <= 15 - int(temp_h) <= 63, "GRUInt on venus hidden requant shift is unsupported"
            assert int(temp) == int(temp_h), "GRUInt on venus output scale must equal hidden scale"

        # Create output tensors
        Y = Tensor(shape=yshape, dtype=X.dtype, scale=int(temp), zero=0)
        hshape = [1, B, self.attrs["hidden_size"]]
        hidden_o = Tensor(shape=hshape, dtype=np.int8, scale=int(temp_h), zero=0)
        self.outputs = [Y, hidden_o]

    def get_workspace(self):
        """Calculate the required workspace for the GRUInt operation."""
        X = self.inputs[0]
        hidden_size = self.attrs["hidden_size"]
        platform = self.attrs.get("platform", "venus")
        layout = self.attrs.get("batch_first", 0)
        if layout == 1:
            B,T,L = self.inputs[0].shape
        else:
            T,B,L = self.inputs[0].shape
        if platform == "venus":
            assert B == 1, "Venus platform requires batch size of 1"
            assert X.mem_type == MemType.SHARE_MEM, "GRUInt on venus requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, "GRUInt on venus requires SHARE_MEM output"
            assert self.outputs[1].mem_type == MemType.SHARE_MEM, "GRUInt on venus requires SHARE_MEM hidden output"
            assert ALIGN8(self.attrs["input_size"]) * ALIGN4(hidden_size * 3) <= 32768, \
                "GRUInt on venus input weight matrix exceeds capacity"
            assert ALIGN8(hidden_size) * ALIGN4(hidden_size * 3) <= 32768, "Hidden size must be aligned to 4 and 8"
            workspace_size = hidden_size * 4 * 3 * 2
            for parameter in self.inputs[self.weight_index:self.weight_index + 4]:
                assert parameter.mem_type == MemType.SHARE_MEM or parameter.has_data(), \
                    "GRUInt on venus requires local or DMA-relocatable constant parameters"
        elif platform == "venusA":
            assert X.mem_type == MemType.SHARE_MEM, "GRUInt on venusA requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, "GRUInt on venusA runtime requires SHARE_MEM output"
            assert self.outputs[1].mem_type == MemType.SHARE_MEM, "GRUInt on venusA runtime requires SHARE_MEM hidden output"
            # Keep a guard band beyond the theoretical int32 scratch usage.
            # The runtime kernels operate on tightly packed scratch buffers,
            # and packing the workspace to the exact lower bound can corrupt
            # the next share-memory slot for larger GRU cases.
            workspace_size = hidden_size * B * 4 * 7
            if layout == 1 and B != 1:
               assert ALIGN4(B) * ALIGN8(T * L) <= 65536, "GRUInt on venusA batch_first input transpose exceeds Luna limit"
               assert ALIGN4(T * hidden_size) * ALIGN8(B) <= 65536, "GRUInt on venusA output transpose exceeds Luna limit"
               assert ALIGN4(hidden_size) * ALIGN8(B) <= 65536, "GRUInt on venusA hidden transpose exceeds Luna limit"
               workspace_size += B * T * L
        else:
            assert X.mem_type == MemType.SHARE_MEM, "GRUInt on arcs requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, "GRUInt on arcs requires SHARE_MEM output"
            assert self.outputs[1].mem_type == MemType.SHARE_MEM, "GRUInt on arcs requires SHARE_MEM hidden output"
            workspace_size = hidden_size * B * 4 * 7
            if layout == 1 and B != 1:
               workspace_size += B * T * L
        if len(self.inputs) == 6 and len(self.inputs[1].shape) != 0:
            assert self.inputs[1].mem_type == MemType.SHARE_MEM, \
                f"GRUInt on {platform} requires SHARE_MEM history"
        for parameter in self.inputs[self.weight_index:self.weight_index + 4]:
            assert parameter.mem_type == MemType.SHARE_MEM or parameter.has_data(), \
                f"GRUInt on {platform} requires local or DMA-relocatable constant parameters"
        return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]

    def pack_params(self):
        """Pack the parameters for the GRUInt operation, handling weight transposition."""
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            weight_i = self.inputs[self.weight_index]
            weight_h = self.inputs[self.weight_index + 1]
            data_i = weight_i.data.transpose(1, 0)
            data_h = weight_h.data.transpose(1, 0)
            self.inputs[self.weight_index].update(data=data_i, shape=data_i.shape)
            self.inputs[self.weight_index + 1].update(data=data_h, shape=data_h.shape)

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the GRUInt operation."""
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

        # Determine dimensions based on layout
        layout = self.attrs.get("batch_first", 0)
        if layout == 0:
            T = xshape[0]
            B = xshape[1]
            F = xshape[2]
        else:
            B = xshape[0]
            T = xshape[1]
            F = xshape[2]

        hidden_size = self.attrs["hidden_size"]
        overall_flops = T * (hidden_size + F) * hidden_size * 3 * 2
        return int(overall_flops)

__all__ = ["GRUInt"]
