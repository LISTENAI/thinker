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

        transB = self.attrs.get("transB", 1)
        self.attrs['transB'] = transB

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the LinearInt operation."""
        attrs = tffi.new("LinearIntAttrs *")
        attrs.transA = 0
        attrs.transB = self.attrs['transB']
        attrs.quant_type = RoundMethod.from_str(self.attrs["quant_mode"]).value
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
        platform = self.attrs.get("platform", "venus")
        assert platform in {"venus", "arcs", "venusA"}, "Unsupported LinearInt platform"
        assert self.attrs["transB"] == 1, "LinearInt runtime only supports transB=1"
        assert len(inputs) in {2, 3}, "LinearInt operator must have 2 or 3 inputs"
        assert len(W.shape) == 2, "LinearInt weight must be rank 2"
        assert X.zero == 0 and W.zero == 0, "LinearInt only supports zero point 0"
        data_bits = self.attrs["data_bits"]
        parameter_bits = self.attrs["parameter_bits"]
        o_bits = self.attrs["o_bits"]
        assert data_bits == X.dtype.itemsize * 8, "LinearInt data_bits must match input dtype"
        if platform == "arcs":
            assert X.dtype == np.int8, "LinearInt on arcs only supports int8 input"
            assert W.dtype == np.int8, "LinearInt on arcs only supports int8 weight storage"
            assert parameter_bits in (4, 8), "LinearInt on arcs only supports int4/int8 weights"
            assert 1 <= len(X.shape) <= 4, "LinearInt on arcs only supports rank 1 to 4 input"
            if parameter_bits == 4:
                assert x_shape[-1] % 2 == 0, "LinearInt on arcs requires even input width for int4 weights"
        elif platform == "venusA":
            assert 1 <= len(X.shape) <= 4, "LinearInt on venusA only supports rank 1 to 4 input"
            supported = {
                (np.dtype(np.int8), np.dtype(np.int8), 4, 8),
                (np.dtype(np.int8), np.dtype(np.int8), 8, 8),
                (np.dtype(np.int8), np.dtype(np.int8), 8, 16),
                (np.dtype(np.int8), np.dtype(np.int8), 8, 32),
                (np.dtype(np.int8), np.dtype(np.int32), 32, 32),
                (np.dtype(np.int16), np.dtype(np.int16), 16, 8),
                (np.dtype(np.int16), np.dtype(np.int16), 16, 16),
                (np.dtype(np.int16), np.dtype(np.int16), 16, 32),
                (np.dtype(np.int32), np.dtype(np.int32), 32, 8),
                (np.dtype(np.int32), np.dtype(np.int32), 32, 32),
                (np.dtype(np.int32), np.dtype(np.int8), 8, 8),
            }
            assert (X.dtype, W.dtype, parameter_bits, o_bits) in supported, \
                "Unsupported LinearInt dtype/parameter/output combination on venusA"
            if parameter_bits == 4:
                assert X.shape[-1] % 2 == 0, "LinearInt int4 weights require even input width"
        else:
             assert X.dtype == np.int8, "Input must be of type int8"
             assert W.dtype == np.int8, "LinearInt on venus requires int8 weight storage"
             assert parameter_bits == 8, "LinearInt on venus only supports int8 weights"
             assert 1 <= len(X.shape) <= 3, "LinearInt on venus only supports rank 1, 2, or 3 input"
        if platform == "venus" and len(inputs) == 3:
            assert inputs[2].dtype in (np.int8, np.int16, np.int32), "LinearInt bias has unsupported dtype"
        if platform in {"arcs", "venusA"} and len(inputs) == 3:
            assert inputs[2].dtype == np.int32 and inputs[2].size == w_shape[0], \
                f"LinearInt on {platform} requires int32 bias matching output width"
        if len(inputs) == 3:
            assert inputs[2].size == w_shape[0], "LinearInt bias size must match output width"
            assert inputs[2].zero == 0, "LinearInt bias zero point must be 0"

        # Calculate input dimensions
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

        # Check shape compatibility based on transpose flags
        if self.attrs["transB"] == 0:
            assert x_w == w_shape[0], f"x_w:{x_w} and w_shape[0]:{w_shape[0]}"
            output_dim = w_shape[1]
        else:
            assert x_w == w_shape[-1], f"x_w:{x_w} and w_shape[-1]:{w_shape[-1]}"
            output_dim = w_shape[0]

        # Determine output shape based on input dimensions
        if len(X.shape) == 1:
            shape = [output_dim]
        elif len(X.shape) == 2:
            shape = [x_shape[0], output_dim]
        elif len(X.shape) == 4:
            shape = [x_shape[0], x_shape[1], x_shape[2], output_dim]
        else:
            shape = [x_shape[0], x_shape[1], output_dim]

        # Determine output data type and bits
        o_bits = self.attrs.get("o_bits", 8)
        if platform == "arcs":
            assert o_bits in (8, 32), "LinearInt on arcs only supports int8 or int32 output"
            if self.attrs["parameter_bits"] == 4:
                assert o_bits == 8, "LinearInt on arcs only supports int8 output for int4 weights"
            shift = X.scale + W.scale - int(temp)
            assert o_bits == 32 or shift >= 0, "LinearInt on arcs int8 output does not support left shift"
            assert shift <= 63, "LinearInt on arcs shift exceeds Luna limits"
            if o_bits == 32 and shift < 0:
                assert -shift <= 30, "LinearInt on arcs int32 left-shift compensation exceeds scalar limits"
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

        if platform == "venusA":
            shift = X.scale + W.scale - int(temp)
            min_shift = 0 if o_bits == 8 else (-14 if o_bits == 16 else -30)
            assert min_shift <= shift <= 63, "LinearInt on venusA shift exceeds Luna/scalar limits"
        elif platform == "venus":
            shift = X.scale + W.scale - int(temp)
            assert 0 <= shift <= 63, "LinearInt on venus shift exceeds runtime limits"

        # Create output tensor
        Y = X.clone(shape=shape, dtype=data_type, bits=bits, scale=int(temp))
        assert Y.zero == 0, "LinearInt output zero point must be 0"
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the LinearInt operation."""
        workspace_size = 0
        data = self.inputs[0]
        weight = self.inputs[1]
        out = self.outputs[0]
        platform = self.attrs.get("platform", "venus")
        weight_bits = self.attrs["parameter_bits"]

        if platform == "arcs":
            if weight.dtype == np.int8:
                workspace_size += self.inputs[0].nbytes
                workspace_size += self.outputs[0].nbytes
            else:
                workspace_size += self.inputs[0].nbytes * 4
                workspace_size += self.outputs[0].nbytes * 4
        elif platform == "venusA":
            assert self.attrs["transB"] == 1, "LinearInt on venusA runtime only supports transB=1"
            M = int(np.prod(data.shape[:-1]))
            N = data.shape[-1]
            if self.attrs['transB'] == 0:
                L = weight.shape[1]
                assert N == weight.shape[0], "N must be equal to weight shape 0 when transB =0"
            else:
                L = weight.shape[0]
                assert N == weight.shape[1], "N must be equal to weight shape 1 when transB =1"
            input_size = M * N
            output_size = M * L
            weight_size = L * N
            if data.dtype == np.int8:
                if out.dtype == np.int8:
                    out_trans_large = ALIGN4(L) * ALIGN8(M) > 65536
                    workspace_size = input_size
                    if data.mem_type != MemType.SHARE_MEM:
                        workspace_size = max(workspace_size, input_size * 2)
                    if out.mem_type != MemType.SHARE_MEM:
                        if out_trans_large:
                            workspace_size = max(workspace_size, max(input_size, output_size) + output_size)
                        else:
                            workspace_size = max(workspace_size, input_size + output_size)
                    elif out_trans_large:
                        workspace_size = max(workspace_size, input_size + output_size)
                elif out.dtype == np.int16:
                    if out.mem_type != MemType.SHARE_MEM:
                        if ALIGN4(L) * ALIGN4(M) <= 65536:
                            workspace_size = (input_size + output_size) * 2
                        else:
                            workspace_size = (output_size + max(input_size, output_size)) * 2
                    else:
                        if ALIGN4(L) * ALIGN4(M) <= 65536:
                            workspace_size = input_size * 2
                        else:
                            workspace_size = (input_size + output_size) * 2
                elif out.dtype == np.int32:  # int32 output
                    if weight.dtype == np.int32:
                        input_32_size = input_size * 4
                        input_16_size = input_size * 2
                    elif weight.dtype == np.int8:
                        input_32_size = input_size * 4
                        input_16_size = 0
                    else:
                        raise ValueError(f"Unsupported weight.dtype: {weight.dtype}\n")

                    if out.mem_type != MemType.SHARE_MEM:
                        if ALIGN2(L) * ALIGN4(M) <= 32768:
                            workspace_size = input_32_size + max(output_size * 4 , input_16_size)
                        else:
                            workspace_size = max(input_32_size, output_size * 4) + max(output_size * 4, input_16_size)
                    else:
                        if ALIGN2(L) * ALIGN4(M) <= 32768:
                            workspace_size = input_32_size + input_16_size
                        else:
                            workspace_size = input_32_size +  max(output_size * 4, input_16_size)
            elif data.dtype == np.int32:
                if out.dtype == np.int8:  # int32 output
                    #计算weight从int8到int32的workspace
                    if weight.dtype == np.int8:
                        weight_input_size = weight_size * 6
                    else:
                        weight_input_size = 0

                    if out.mem_type != MemType.SHARE_MEM:  # output in PSRAM (y_in_psram=1)
                        if ALIGN2(L) * ALIGN4(M) <= 65536:
                            workspace_size = input_size * 4 + output_size + weight_input_size
                        else:
                            workspace_size = max(input_size * 4, output_size) + output_size + weight_input_size
                    else:  # output in ShareRAM (y_in_psram=0)
                        if ALIGN2(L) * ALIGN4(M) <= 65536:
                            workspace_size = input_size * 4 + weight_input_size
                        else:
                            workspace_size = input_size * 4 + output_size + weight_input_size
                elif out.dtype == np.int32:  # int32 output
                    if out.mem_type != MemType.SHARE_MEM:  # output in PSRAM (y_in_psram=1)
                        if ALIGN2(L) * ALIGN4(M) <= 32768:
                            workspace_size = input_size * 4 + output_size * 4
                        else:
                            workspace_size = max(input_size, output_size) * 4 + output_size * 4
                    else:  # output in ShareRAM (y_in_psram=0)
                        if ALIGN2(L) * ALIGN4(M) <= 32768:
                            workspace_size = input_size * 4
                        else:
                            workspace_size = input_size * 4 + output_size * 4
            elif data.dtype == np.int16:
                if out.dtype == np.int8:
                    if ALIGN4(L) * ALIGN4(M) <= 65536:
                        workspace_size = input_size * 2 + max(input_size * 2, output_size)
                    else:
                        workspace_size = max(input_size * 2, output_size) * 2
                elif out.dtype == np.int16:
                    if ALIGN4(L) * ALIGN4(M) <= 65536:
                        workspace_size = input_size * 2 + max(input_size, output_size) * 2
                    else:
                        workspace_size = max(input_size, output_size) * 2 * 2
                elif out.dtype == np.int32:
                    if ALIGN2(L) * ALIGN4(M) <= 32768:
                        workspace_size = input_size * 2 + max(input_size * 2, output_size * 4)
                    else:
                        workspace_size = max(input_size * 2, output_size * 4) * 2
                else:
                    raise ValueError(f"Unsupported output dtype {out.dtype} for int16 input\n")
            if data.mem_type != MemType.SHARE_MEM:
                workspace_size = max(workspace_size, input_size * data.dtype.itemsize * 2)
        elif platform == "venus":
            M = int(np.prod(data.shape[:-1]))
            N = data.shape[-1]
            if self.attrs['transB'] == 0:
                L = weight.shape[1]
                assert N == weight.shape[0]
            else:
                L = weight.shape[0]
                assert N == weight.shape[1]
            assert data.dtype == np.int8 and weight.dtype == np.int8

            int8_condition_l = ALIGN4(M) * ALIGN8(N)
            int8_condition_r = ALIGN8(N) * ALIGN4(L)
            split_left = int8_condition_l > 65536
            if out.dtype != np.int8:
                assert not split_left, "LinearInt on venus cannot split rows for int16/int32 output"
                assert out.mem_type == MemType.SHARE_MEM, \
                    "LinearInt on venus requires SHARE_MEM int16/int32 output"
            split_num = 1
            split_M = M

            if int8_condition_l > 65536:
                assert ALIGN4(1) * ALIGN8(N) <= 65536, \
                    "LinearInt shared dimension exceeds the Venus row-split limit"
                split_num = 2
                split_M = math.ceil(M / split_num)
                int8_condition_l_split = ALIGN4(split_M) * ALIGN8(N)
                while int8_condition_l_split > 65536:
                    split_num += 1
                    split_M = math.ceil(M / split_num)
                    int8_condition_l_split = ALIGN4(split_M) * ALIGN8(N)

            if int8_condition_r > 32768:
                assert ALIGN8(N) * ALIGN4(1) <= 32768, \
                    "LinearInt shared dimension exceeds the Venus column-split limit"

            split_input_size = split_M * N
            split_output_size = split_M * L
            if len(self.inputs) > 2:
                if data.mem_type != MemType.SHARE_MEM:
                    workspace_size = ALIGN4(max(split_input_size, split_output_size)
                                            if out.mem_type != MemType.SHARE_MEM
                                            else split_input_size)
                elif out.mem_type != MemType.SHARE_MEM:
                    workspace_size = ALIGN4(split_output_size)
                workspace_size += split_output_size * 4
            elif data.mem_type != MemType.SHARE_MEM and out.mem_type != MemType.SHARE_MEM:
                workspace_size = ALIGN4(split_input_size) + split_output_size
            elif data.mem_type != MemType.SHARE_MEM:
                workspace_size = ALIGN4(split_input_size)
            elif out.mem_type != MemType.SHARE_MEM:
                workspace_size = split_output_size

        if workspace_size:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def pack_params(self):
        """Pack the parameters for the LinearInt operation, handling weight quantization."""
        weight_bits = self.attrs["parameter_bits"]
        weight_data = self.inputs[1].data
        layout = self.inputs[1].layout
        shape       = weight_data.shape
        platform = self.attrs.get("platform", "venus")

        if platform in {"arcs", "venusA"}:
            assert self.attrs['transB'] == 1, "Only support transB=1"
            if weight_bits == 4:
                new_weight_data = combine4bit_8bit(weight_data)
            else:
                new_weight_data = weight_data
            self.inputs[1].update(data=new_weight_data, shape=shape, bits=np.float32(weight_bits / 8), layout=layout)
        elif platform == "venus":
            if self.attrs["transB"] == 1:
                new_weight_data = weight_data.transpose(1, 0)
                self.attrs['transB'] = 0
            logical_shape = new_weight_data.shape
            if weight_bits == 4:
                new_weight_data = combine4bit_8bit(new_weight_data)
            self.inputs[1].update(data=new_weight_data, shape=logical_shape, bits=np.float32(weight_bits / 8.0))
            if layout == Layout.NCHW:
                self.inputs[1].update(layout=Layout.NCWH)

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
