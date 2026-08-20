import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...xsympy import is_sympy
from ...resource_packer._type._ctype import tffi
from ...enum_defines import DevType, MemType
from .utils import QuantType, RoundMethod, calc_expr
from .base import Operator, OperatorAttrs, register_op


class LstmIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check if required parameters are present and valid."""
        assert "hidden_size" in self.attrs, "Missing required attribute: hidden_size"
        assert "input_size" in self.attrs, "Missing required attribute: input_size"
        assert "batch_first" in self.attrs, "Missing required attribute: batch_first"

        batch_first = self.attrs.get("batch_first", 0)
        self.attrs["layout"] = 1 if batch_first else 0

        go_forward = self.attrs.get("go_forward", 1)
        assert go_forward in (0, 1), "LSTMInt only supports a single forward or reverse direction"
        if go_forward == 0:
            self.attrs["direction"] = 1
        else:
            self.attrs["direction"] = 0

        platform = self.attrs.get("platform", "venus")
        if platform in {"arcs", "venusA"}:
            assert "quant_mode" in self.attrs, "Missing required attribute: quant_mode"
        else:
            if "quant_mode" in self.attrs:
                quant_mode = self.attrs.get("quant_mode")
                if quant_mode == "luna_quant":
                    quant_mode = "FLOOR_ADD"
            else:
                quant_mode = self.attrs.get("platform_quant")
                if quant_mode == "luna_quant":
                    quant_mode = "FLOOR_ADD"
            self.attrs['quant_mode'] = quant_mode

        assert "scale_x" in self.attrs, "Missing required attribute: scale_x"
        assert "scale_iw" in self.attrs, "Missing required attribute: scale_iw"
        assert "scale_hw" in self.attrs, "Missing required attribute: scale_hw"
        assert "scale_o" in self.attrs, "Missing required attribute: scale_o"
        assert "scale_h" in self.attrs, "Missing required attribute: scale_h"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the LstmInt operation."""
        attrs = tffi.new("LstmIntAttrs *")
        attrs.direction = self.attrs["direction"]
        attrs.hidden_size = self.attrs["hidden_size"]
        attrs.input_size = self.attrs["input_size"]
        attrs.layout = self.attrs["layout"]
        attrs.quant_type = RoundMethod.from_str(self.attrs["quant_mode"]).value
        return bytes(tffi.buffer(attrs))

@register_op
class LSTMInt(Operator):
    def __init__(self, attrs={}):
        """Initialize the LSTMInt operator with given attributes."""
        self.attrs = LstmIntAttrs(attrs)
        self.weight_index = 1

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) > 3, "LSTMInt operator must have more than 3 inputs"

        if len(inputs) == 5:
            self.weight_index = 1
        elif len(inputs) == 6:
            self.weight_index = 2
        elif len(inputs) == 7:
            self.weight_index = 3
        elif len(inputs) == 8:
            self.inputs[2].dtype = inputs[0].dtype
            self.weight_index = 4

        X = inputs[0]
        i2h_w = inputs[self.weight_index]
        h2h_w = inputs[self.weight_index + 1]

        # Process input scale
        scale_x = self.attrs.get("scale_x", 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Input scale must be a power of 2"
        if X.scale != -1:
            assert X.scale == temp, "Input scale must match attribute scale_x"
        else:
            X.scale = temp

        # Process weight scales
        scale_iw = self.attrs.get("scale_iw", 1.0)
        temp = math.log(scale_iw, 2)
        assert abs(temp - int(temp)) < 0.000001, "Input-to-hidden weight scale must be a power of 2"
        i2h_w.scale = temp

        scale_hw = self.attrs.get("scale_hw", 1.0)
        temp = math.log(scale_hw, 2)
        assert abs(temp - int(temp)) < 0.000001, "Hidden-to-hidden weight scale must be a power of 2"
        h2h_w.scale = temp

        # Process output scales
        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Output scale must be a power of 2"

        scale_h = self.attrs.get("scale_h", 1.0)
        temp1 = math.log(scale_h, 2)
        assert temp1 == int(temp1), "Hidden state scale must be a power of 2"

        scale_c = self.attrs.get("scale_c", 32768.0)
        temp2 = math.log(scale_c, 2)
        
        assert temp2 == int(temp2), "cell state scale must be a power of 2"

        platform = self.attrs.get("platform", "venus")
        q_ib = int(X.scale + i2h_w.scale)
        q_hb = int(int(temp1) + h2h_w.scale)
        active_q_in = 11 if platform == "venus" else 27
        for q in (q_ib, q_hb):
            if q < active_q_in:
                assert active_q_in - q <= 30, "LSTMInt activation left shift exceeds scalar range"
            else:
                assert q - active_q_in <= 63, "LSTMInt activation right shift exceeds Luna limit"
        assert int(temp2) == 15, "LSTMInt requires Q15 cell state scale"
        assert -33 <= int(temp1) <= 30, "LSTMInt hidden scale requires 30 - h_q in [0, 63]"
        assert int(temp) == int(temp1), "LSTMInt requires equal output and hidden scales"

        assert X.dtype == i2h_w.dtype == h2h_w.dtype, "Input, i2h_w, and h2h_w must have the same dtype"
        if platform in {"venus", "venusA"}:
            assert X.dtype == np.int8, "LSTMInt on venusA runtime only supports int8 input/weights"
        assert len(X.shape) == 3, "Input must be a 3D tensor"

        hidden_size = self.attrs["hidden_size"]
        if platform in ("venus", "arcs", "venusA"):
            assert self.attrs["layout"] in (0, 1), "LSTMInt layout must be 0 or 1"
            assert hidden_size > 0 and self.attrs["input_size"] > 0, \
                "LSTMInt requires positive input and hidden sizes"
            assert i2h_w.shape == (hidden_size * 4, self.attrs["input_size"]), \
                "LSTMInt i2h weight shape mismatch"
            assert h2h_w.shape == (hidden_size * 4, hidden_size), \
                "LSTMInt h2h weight shape mismatch"
            i2h_bias = inputs[self.weight_index + 2]
            h2h_bias = inputs[self.weight_index + 3]
            assert i2h_bias.dtype == h2h_bias.dtype == np.int32, "LSTMInt requires int32 biases"
            assert i2h_bias.shape == h2h_bias.shape == (hidden_size * 4,), \
                "LSTMInt bias shape mismatch"
            assert X.shape[2] == self.attrs["input_size"], "LSTMInt input_size mismatch"
        if self.weight_index >= 3:
            hidden_in = inputs[self.weight_index - 2]
            cell_in = inputs[self.weight_index - 1]
            assert hidden_in.dtype == np.int8 and hidden_in.scale == int(temp1), "LSTMInt hidden state must match output dtype and scale"
            assert cell_in.dtype == np.int32 and cell_in.scale == int(temp2), "LSTMInt cell state must match output dtype and scale"
        # Determine output shape based on layout
        if self.attrs["layout"] == 1:
            T = X.shape[1]
            B = X.shape[0]
            yshape = [B, T, hidden_size]
        else:
            T = X.shape[0]
            B = X.shape[1]
            yshape = [T, B, hidden_size]

        assert B == 1, "LSTMInt only supports batch size 1"

        Y = X.clone(shape=tuple(yshape), scale=int(temp))
        self.outputs = [Y]

        # Create hidden state tensors
        hshape = [1, 1, hidden_size]
        hidden_o = X.clone(shape=hshape, dtype=np.int8, bits=1, scale=int(temp1))
        self.outputs.append(hidden_o)

        cshape = [1, 1, hidden_size]
        hidden_c = X.clone(shape=cshape, dtype=np.int32, bits=4, scale=int(temp2))
        self.outputs.append(hidden_c)

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the LSTMInt operation."""
        i2h_weight = self.inputs[self.weight_index]
        h2h_weight = self.inputs[self.weight_index + 1]
        i2h_bias = self.inputs[self.weight_index + 2]
        h2h_bias = self.inputs[self.weight_index + 3]

        hidden_size = self.attrs["hidden_size"]
        workspace_size = hidden_size * 4 * 8
        platform = self.attrs.get("platform", "venus")

        if platform == "venus":
            def matrix_has_valid_split(rows, cols):
                return any(cols % split == 0 and
                           ((rows + 7) // 8 * 8) *
                           (((cols // split) + 3) // 4 * 4) <= 32768
                           for split in range(1, cols + 1))

            assert self.inputs[0].mem_type == MemType.SHARE_MEM, "LSTMInt on venus requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, "LSTMInt on venus requires SHARE_MEM output"
            assert self.outputs[1].mem_type == MemType.SHARE_MEM, "LSTMInt on venus requires SHARE_MEM hidden output"
            assert self.outputs[2].mem_type == MemType.SHARE_MEM, "LSTMInt on venus requires SHARE_MEM cell output"
            for state in self.inputs[1:self.weight_index]:
                assert state.mem_type == MemType.SHARE_MEM, "LSTMInt on venus requires SHARE_MEM state input"
            for parameter in self.inputs[self.weight_index:self.weight_index + 4]:
                assert parameter.mem_type == MemType.SHARE_MEM or parameter.has_data(), \
                    "LSTMInt on venus requires local or DMA-relocatable constant parameters"
            assert matrix_has_valid_split(self.attrs["input_size"], hidden_size * 4), \
                "LSTMInt on venus input weight matrix has no valid split"
            assert matrix_has_valid_split(hidden_size, hidden_size * 4), \
                "LSTMInt on venus recurrent weight matrix has no valid split"

        if platform in {"arcs", "venusA"}:
            workspace_size += h2h_weight.nbytes + i2h_weight.nbytes + i2h_bias.nbytes + h2h_bias.nbytes
            assert self.inputs[0].mem_type == MemType.SHARE_MEM, "LSTMInt on arcs/venusA requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, "LSTMInt on arcs/venusA requires SHARE_MEM output"
            assert self.outputs[1].mem_type == MemType.SHARE_MEM, "LSTMInt on arcs/venusA requires SHARE_MEM hidden output"
            assert self.outputs[2].mem_type == MemType.SHARE_MEM, "LSTMInt on arcs/venusA requires SHARE_MEM cell output"
            for state in self.inputs[1:self.weight_index]:
                assert state.mem_type == MemType.SHARE_MEM, "LSTMInt on arcs/venusA requires SHARE_MEM state input"

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def pack_params(self):
        """Pack the parameters for the LSTMInt operation, handling weight quantization."""
        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            weight_i = self.inputs[self.weight_index]
            weight_h = self.inputs[self.weight_index + 1]
            data_i = weight_i.data.transpose(1, 0)
            data_h = weight_h.data.transpose(1, 0)
            self.inputs[self.weight_index].update(data=data_i, shape=data_i.shape)
            self.inputs[self.weight_index + 1].update(data=data_h, shape=data_h.shape)

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the LSTMInt operation."""
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
        overall_flops = T * (hidden_size + F) * hidden_size * 4 * 2
        return int(overall_flops)

__all__ = ["LSTMInt"]
