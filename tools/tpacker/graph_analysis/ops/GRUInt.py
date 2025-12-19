import math
import numpy as np
from ...resource_packer._type._ctype import tffi
from ...graph import Tensor
from ...enum_defines import DevType, MemType
from ...xsympy import is_sympy
from .utils import QuantType, calc_expr
from .base import Operator, OperatorAttrs, register_op

class GRUIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check if all required parameters are present."""
        required_attrs = ["scale_x", "scale_iw", "scale_hw", "scale_h", "scale_o", "batch_first", "hidden_size"]
        for attr in required_attrs:
            assert attr in self.attrs, f"Missing required attribute: {attr}"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the GRUInt operation."""
        attrs = tffi.new("GRUIntAttrs *")
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
        assert len(inputs) > 3, "GRUInt operator must have more than three inputs"

        X = inputs[0]  # Input tensor
        i2h_w = inputs[self.weight_index]  # Input-to-hidden weights
        h2h_w = inputs[self.weight_index + 1]  # Hidden-to-hidden weights

        # Check data type consistency
        assert X.dtype == i2h_w.dtype == h2h_w.dtype, "Input and weight data types must be consistent"

        # Validate tensor dimensions
        assert len(X.shape) == 3
        assert len(i2h_w.shape) in (2, 3)
        assert len(h2h_w.shape) in (2, 3)

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
        temp = math.log(scale_h, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

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

        # Create output tensors
        Y = Tensor(shape=yshape, dtype=X.dtype, scale=int(temp))
        hshape = [1, B, self.attrs["hidden_size"]]
        hidden_o = Tensor(shape=hshape, dtype=np.float32, scale=int(temp))
        self.outputs = [Y, hidden_o]

    def get_workspace(self):
        """Calculate the required workspace for the GRUInt operation."""
        h2h_weight = self.inputs[self.weight_index + 1]
        hidden_size = h2h_weight.shape[1]
        workspace_size = hidden_size * 4 * 3
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def pack_params(self):
        """Pack the parameters for the GRUInt operation, handling weight transposition."""
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