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
        if go_forward == 0:
            self.attrs["direction"] = 1
        elif go_forward == 1:
            self.attrs["direction"] = 0
        else:
            self.attrs["direction"] = 2

        platform = self.attrs.get("platform", "venus")
        if platform in {"arcs", "venusA"}:
            quant_type = RoundMethod.from_str(self.attrs.get("quant_mode"))
        elif platform == "venus":
            quant_type = QuantType.from_str(self.attrs.get("platform_quant"))
        self.attrs["quant_mode"] = quant_type

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
        attrs.quant_type = self.attrs["quant_mode"].value
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

        assert X.dtype == i2h_w.dtype == h2h_w.dtype, "Input, i2h_w, and h2h_w must have the same dtype"
        assert len(X.shape) == 3, "Input must be a 3D tensor"

        # Determine output shape based on layout
        if self.attrs["layout"] == 0:
            T = X.shape[1]
        else:
            T = X.shape[2]

        hidden_size = self.attrs["hidden_size"]
        yshape = [1, T, hidden_size]
        Y = X.clone(shape=tuple(yshape), scale=int(temp))
        self.outputs = [Y]

        # Create hidden state tensors
        hshape = [1, 1, hidden_size]
        hidden_o = X.clone(shape=hshape, dtype=np.int8, bits=1, scale=int(temp))
        self.outputs.append(hidden_o)

        cshape = [1, 1, hidden_size]
        hidden_c = X.clone(shape=cshape, dtype=np.int32, bits=4, scale=int(temp1))
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

        if platform in {"arcs", "venusA"}:
            workspace_size += h2h_weight.nbytes + i2h_weight.nbytes + i2h_bias.nbytes + h2h_bias.nbytes

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