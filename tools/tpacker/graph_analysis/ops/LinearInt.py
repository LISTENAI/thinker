import math
import numpy as np
from typing import Any, Dict, Optional, List

from ...graph import Tensor
from ...xsympy import is_sympy
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op
from .utils import QuantType, calc_expr, combine4bit_8bit, RoundMethod
from ...enum_defines import DevType, Layout, MemType, ALIGN2, ALIGN4, ALIGN8

class LinearIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check if required parameters are present and valid."""
        required_attrs = ["scale_x", "scale_w", "scale_o", "data_bits", "o_bits", "parameter_bits"]
        for attr in required_attrs:
            assert attr in self.attrs, f"Missing required attribute: {attr}"
        
        platform = self.attrs.get("platform", "venus")
        if platform in {"arcs", "venusA"}:
            quant_type = RoundMethod.from_str(self.attrs.get("quant_mode"))
        elif platform == "venus":
            quant_type = QuantType.from_str(self.attrs.get("platform_quant"))
        self.attrs['quant_mode'] = quant_type

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the LinearInt operation."""
        attrs = tffi.new("LinearIntAttrs *")
        attrs.transA = 0
        attrs.transB = 0
        attrs.quant_type = self.attrs["quant_mode"].value
        return bytes(tffi.buffer(attrs))

@register_op
class LinearInt(Operator):
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        """Initialize the LinearInt operator with given attributes."""
        self.attrs = LinearIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        X = inputs[0]
        W = inputs[1]
        x_shape = list(X.shape)
        w_shape = list(W.shape)

        assert X.dtype in (np.int8,), "Input must be of type int8"
        assert W.dtype in (np.int8, np.int32), "Weight must be of type int8 or int32"
        assert len(inputs) in {2, 3}, "LinearInt operator must have 2 or 3 inputs"

        # Calculate input dimensions
        x_h = calc_expr(str(x_shape[-2]), dynamic_shape) if is_sympy(x_shape[-2]) else x_shape[-2]
        x_w = calc_expr(str(x_shape[-1]), dynamic_shape) if is_sympy(x_shape[-1]) else x_shape[-1]

        # Process input scale
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x[0], 2) if isinstance(scale_x, tuple) else math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Input scale must be a power of 2"
        if X.scale != -1:
            assert X.scale == int(temp), "Input scale must match attribute scale_x"
        else:
            X.scale = int(temp)

        # Process weight scale
        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w[0], 2) if isinstance(scale_w, tuple) else math.log(scale_w, 2)
        assert abs(temp - int(temp)) < 0.000001, "Weight scale must be a power of 2"
        W.scale = int(temp)

        # Process output scale
        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o[0], 2) if isinstance(scale_o, tuple) else math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Output scale must be a power of 2"
        self.outputs[0].scale = int(temp)

        # Check shape compatibility based on transpose flags
        assert x_w == w_shape[-1], f"x_w:{x_w} and w_shape[-1]:{w_shape[-1]}"

        # Determine output shape based on input dimensions
        if len(X.shape) == 1:
            shape = [w_shape[0]]
        elif len(X.shape) == 2:
            shape = [x_shape[0], w_shape[0]]
        elif len(X.shape) == 4:
            shape = [x_shape[0], x_shape[1], x_shape[2], w_shape[-1]]
        else:
            shape = [x_shape[0], x_shape[1], w_shape[0]]

        # Determine output data type and bits
        o_bits = self.attrs.get("o_bits", 8)
        if o_bits == 32:
            data_type = np.dtype("i4")
            bits = 4
        elif o_bits == 16:
            data_type = np.dtype("i2")
            bits = 2
        elif o_bits == 8:
            data_type = np.dtype("i1")
            bits = 1
        else:
            raise ValueError(f"Unsupported o_bits value: {o_bits}")

        # Create output tensor
        Y = X.clone(shape=shape, dtype=data_type, bits=bits, scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the LinearInt operation."""
        workspace_bytes = 0
        data = self.inputs[0]
        weight = self.inputs[1]
        out = self.outputs[0]
        platform = self.attrs.get("platform", "venus")
        weight_bits = self.attrs["parameter_bits"]

        if platform == "arcs":
            if weight.dtype == np.int8:
                workspace_bytes += self.inputs[0].nbytes
                workspace_bytes += self.outputs[0].nbytes
            else:
                workspace_bytes += self.inputs[0].nbytes * 4
                workspace_bytes += self.outputs[0].nbytes * 4
        elif platform == "venusA":
            M = int(np.prod(data.shape[:-1]))
            N = data.shape[-1]
            L = weight.shape[-1]

            if weight.dtype == np.int8:
                out_size = ALIGN4(M) * ALIGN8(L)
                if out_size > 65536:
                    workspace_bytes += max(data.nbytes * weight.dtype.itemsize, out.nbytes)
                else:
                    workspace_bytes += data.nbytes * weight.dtype.itemsize
            elif weight.dtype == np.int16:
                out_size = ALIGN4(M) * ALIGN4(L)
                if out_size > 65536:
                    workspace_bytes += max(data.nbytes * weight.dtype.itemsize, out.nbytes)
                else:
                    workspace_bytes += data.nbytes * weight.dtype.itemsize
            else:
                out_size = ALIGN2(M) * ALIGN4(L)
                if out_size > 32768:
                    workspace_bytes += max(data.nbytes * weight.dtype.itemsize, out.nbytes)
                else:
                    workspace_bytes += data.nbytes * weight.dtype.itemsize
        elif platform == "venus":
            if len(self.inputs) > 2:
                workspace_bytes = out.nbytes * self.inputs[2].dtype.itemsize

            M = int(np.prod(data.shape[:-1]))
            N = data.shape[-1]
            L = weight.shape[-1]
            assert data.dtype == np.int8 and weight.dtype == np.int8

            int8_condition_l = ALIGN4(M) * ALIGN8(N)
            int8_condition_r = ALIGN8(N) * ALIGN4(L)
            split_num = 1
            split_M = M

            if int8_condition_l > 65536:
                split_num = 2
                split_M = math.ceil(M / split_num)
                int8_condition_l_split = ALIGN4(split_M) * ALIGN8(N)
                while int8_condition_l_split > 65536:
                    split_num += 1
                    split_M = math.ceil(M / split_num)
                    int8_condition_l_split = ALIGN4(split_M) * ALIGN8(N)

            if data.mem_type != MemType.SHARE_MEM and out.mem_type != MemType.SHARE_MEM:
                workspace_bytes += split_M * max(N, L) + split_M * L * 4
            elif data.mem_type != MemType.SHARE_MEM:
                workspace_bytes += split_M * N
            elif out.mem_type != MemType.SHARE_MEM:
                workspace_bytes += split_M * L

        if workspace_bytes:
            return [Tensor.from_shape([workspace_bytes], np.int8, MemType.SHARE_MEM)]
        return []

    def pack_params(self):
        """Pack the parameters for the LinearInt operation, handling weight quantization."""
        weight_bits = self.attrs["parameter_bits"]
        weight_data = self.inputs[1].data
        layout = self.inputs[1].layout
        shape       = weight_data.shape
        platform = self.attrs.get("platform", "venus")

        new_weight_data = weight_data.transpose(1, 0)
        shape = new_weight_data.shape
        if platform in {"arcs", "venusA"}:
            if weight_bits == 4:
                new_weight_data = combine4bit_8bit(new_weight_data)
            self.inputs[1].update(data=new_weight_data, shape=shape, bits=np.float32(weight_bits / 8), layout=layout)
        elif platform == "venus":
            if layout == Layout.NCHW:
                self.inputs[1].update(data=new_weight_data, shape=shape, layout=Layout.NCWH)

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the LinearInt operation."""
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
        input_elements = int(np.prod(xshape))
        output_last_dim = yshape[-1]
        overall_conv_flops = input_elements * output_last_dim + (input_elements - 1) * output_last_dim
        bias_flops = 0

        if len(self.inputs) == 3:
            bias_flops = 2 * input_elements * output_last_dim

        total_flops = overall_conv_flops + bias_flops
        return int(total_flops)

__all__ = ["LinearInt"]