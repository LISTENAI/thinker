import math
import numpy as np
from typing import List
from typing import Tuple
from ...graph import Tensor
from ...xsympy import is_sympy
from ...enum_defines import MemType, DevType, Layout, ALIGN2, ALIGN8, ALIGN16
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, ConvLayout, register_op
from .utils import (
    QuantType,
    RoundMethod,
    attr2tuple,
    calc_conv1d_output_shape,
    calc_expr,
)

class Conv1dIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check and validate the parameters for Conv1dInt operation."""
        platform = self.attrs.get("platform", "venus")
        quant_type = (
            RoundMethod.from_str(self.attrs.get("quant_mode"))
            if platform in ["arcs", "venusA"]
            else QuantType.from_str(self.attrs.get("platform_quant"))
        )
        self.attrs["quant_mode"] = quant_type

        # Check required attributes
        required_attrs = [
            "data_bits",
            "o_bits",
            "parameter_bits",
            "kernel_shape",
            "pads",
            "strides",
            "dilations",
            "group",
            "scale_x",
            "scale_w",
            "scale_o",
        ]
        for attr in required_attrs:
            assert attr in self.attrs, f"Missing required attribute: {attr}"

        # Validate kernel shape
        kernels = attr2tuple(self.attrs.get("kernel_shape"), (1,))
        assert len(kernels) == 1, "kernel of Conv1dInt must be one dim"
        kernel_size = kernels[0]
        if platform == "venus":
            assert 1 <= kernel_size <= 5, "kernel_w for Conv1dInt exceed limit"
        else:
            assert 1 <= kernel_size <= 12, "kernel_w for Conv1dInt exceed limit"

        # Validate pads
        pads = attr2tuple(self.attrs.get("pads"), (0, 0))
        assert len(pads) == 2, "pads of conv1dInt must be two dims"
        pad_left, pad_right = pads
        if platform == "venus":
            assert 0 <= pad_left <= 4, "pad_left for Conv1dInt exceed limit"
            assert 0 <= pad_right <= 4, "pad_right for Conv1dInt exceed limit"
        else:
            assert 0 <= pad_left <= 11, "pad_left for Conv1dInt exceed limit"
            assert 0 <= pad_right <= 11, "pad_right for Conv1dInt exceed limit"

        # Validate strides
        strides = attr2tuple(self.attrs.get("strides"), (1,))
        assert len(strides) == 1, "stride_w for Conv1dInt must be one dim"
        stride = strides[0]
        assert stride in (1, 2, 4), f"stride_h ({stride}) for Conv1dInt exceed limit"

        # Additional checks
        assert kernel_size >= stride, f"weight ({kernel_size}) and stride ({stride}) size of Conv1dInt do not match"
        assert pad_left <= kernel_size and pad_right <= kernel_size, f"pad_h ({pad_left}, {pad_right}) and weight_h ({kernel_size}) size of Conv1dInt do not match"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the Conv1dInt operation."""
        attrs = tffi.new("Conv1dIntAttrs *")
        attrs.kernel = self.attrs["kernel_shape"][0]
        attrs.pad = self.attrs["pads"]
        attrs.stride = self.attrs["strides"][0]
        attrs.group = self.attrs["group"]
        attrs.quant_type = self.attrs["quant_mode"].value
        attrs.act_type = self.attrs.get("act_type", 0)
        return bytes(tffi.buffer(attrs))

@register_op
class Conv1dInt(Operator, ConvLayout):
    def __init__(self, attrs={}):
        self.attrs = Conv1dIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        platform = self.attrs.get("platform", "venus")
        inputs = self.inputs
        assert len(inputs) in (2, 3)

        X = inputs[0]
        W = inputs[1]
        assert len(X.shape) == 3, "Conv1dInt just support 3D data"
        assert len(W.shape) == 3, "Conv1dInt just support 3D weight"
        assert X.dtype == np.int8, "input data type of Conv1dInt must be int8"
        assert W.dtype == np.int8, "weight data type of Conv1dInt must be int8"
        if len(inputs) == 3:
            assert inputs[2].dtype == np.int32, "bias data type of Conv1dInt must be int32"

        # Infer shape
        kernels = self.attrs.get("kernel_shape")
        strides = self.attrs.get("strides")
        pads = self.attrs.get("pads")
        group = self.attrs.get("group", 1)
        shape = calc_conv1d_output_shape(X.shape, W.shape, kernels, strides, (1, 1), pads, group)

        # Infer scale
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x[0], 2) if isinstance(scale_x, tuple) else math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        if X.scale != -1:
            assert X.scale == int(temp), "scale of tensor must be same with scale_x in attribute"
        else:
            self.inputs[0].scale = int(temp)

        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w[0], 2) if isinstance(scale_w, tuple) else math.log(scale_w, 2)
        assert abs(temp - int(temp)) < 0.000001
        self.inputs[1].scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o[0], 2) if isinstance(scale_o, tuple) else math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        # Infer type and return output
        data_bits = self.attrs.get("data_bits")
        parameter_bits = self.attrs.get("parameter_bits")
        output_bits = self.attrs.get("o_bits")

        if platform == "venus":
            assert X.dtype == np.int8
            assert data_bits == 8, f"data type:{X.dtype} must match data bits:{data_bits}"
            assert W.dtype in (np.int8, np.int16)
            if W.dtype == np.int8:
                assert parameter_bits == 8, f"weight type:{W.dtype} must match weight bits:{parameter_bits}"
            else:
                assert parameter_bits == 16, f"weight type:{W.dtype} must match weight bits:{parameter_bits}"
            assert output_bits in (8, 16, 32)
            dtype = (
                np.int8
                if output_bits == 8
                else np.int16
                if output_bits == 16
                else np.int32
            )
        elif platform == "arcs":
            assert X.dtype == np.int8
            assert data_bits == 8, f"data type:{X.dtype} must match data bits:{data_bits}"
            assert W.dtype == np.int8
            assert parameter_bits in (4, 8), f"weight type:{W.dtype} must match weight bits:{parameter_bits}"
            assert output_bits in (8, 32)
            dtype = np.int8 if output_bits == 8 else np.int32
        elif platform == "venusA":
            assert X.dtype in (np.int8, np.int16, np.int32)
            assert data_bits in (8, 16), f"data type:{X.dtype} must match data bits:{data_bits}"
            assert W.dtype in (np.int8, np.int16)
            assert parameter_bits in (4, 8, 16), f"weight type:{W.dtype} must match weight bits:{parameter_bits}"
            assert output_bits in (8, 16, 32)
            dtype = (
                np.int8
                if output_bits == 8
                else np.int16
                if output_bits == 16
                else np.int32
            )

        Y = X.clone(shape=tuple(shape), scale=int(temp), dtype=dtype, bits=int(output_bits / 8))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the Conv1dInt operation."""
        platform = self.attrs.get("platform", "venus")
        platform_module = __import__(
            f"tpacker.graph_analysis.ops.{platform}", fromlist=[""]
        )
        bias = self.inputs[2] if len(self.inputs) == 3 else None
        workspace_size = platform_module.get_Conv1dInt_workspace(
            self.inputs[0],
            self.inputs[1],
            bias,
            self.outputs[0],
            self.attrs["kernel_shape"],
            self.attrs["strides"],
            self.attrs["dilations"],
            self.attrs["pads"],
            self.attrs["group"],
        )
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def pack_params(self):
        """Pack the parameters for the Conv1dInt operation."""
        platform = self.attrs.get("platform", "venus")
        platform_module = __import__(
            f"tpacker.graph_analysis.ops.{platform}", fromlist=[""]
        )
        weight_bits = self.attrs["parameter_bits"]
        new_weight = platform_module.Conv1dInt_weight_rearrange(
            self.inputs[0],
            self.inputs[1],
            self.outputs[0],
            self.attrs["kernel_shape"],
            self.attrs["strides"],
            self.attrs["dilations"],
            self.attrs["pads"],
            self.attrs["group"],
            weight_bits,
        )
        self.inputs[1].update(
            shape=new_weight.shape,
            data=new_weight.data,
            bits=np.float32(weight_bits / 8),
            layout=new_weight.layout,
        )

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the Conv1dInt operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        y_out = Y.shape[1]
        y_h = Y.shape[2]

        xshape = list(X.shape)
        yshape = list(Y.shape)
        for i, s in enumerate(xshape):
            if is_sympy(s):
                xshape[i] = calc_expr(str(s), dynamic_shape)
        for i, s in enumerate(yshape):
            if is_sympy(s):
                yshape[i] = calc_expr(str(s), dynamic_shape)

        group = self.attrs["group"]
        weight = self.inputs[1]
        kernel_c = weight.shape[1]
        kernel_num = weight.shape[0]
        kernel_h = weight.shape[2]
        weight_dims = list(weight.shape[1:])
        output_dims = list(Y.shape[1:])
        weight_count = int(np.prod(weight_dims))
        active_elements_count = int(np.prod(output_dims))

        if 1 == kernel_c and group == kernel_num:
            overall_conv_flops = 2 * (kernel_h + y_out) * kernel_c * group * y_h
            overall_flops = overall_conv_flops
        elif group != 1:
            overall_conv_flops = (weight_count / group) * active_elements_count + (weight_count / group - 1) * active_elements_count
            bias_flops = active_elements_count if len(self.inputs) == 3 else 0
            overall_flops = overall_conv_flops + bias_flops
        else:
            overall_conv_flops = weight_count * active_elements_count + (weight_count - 1) * active_elements_count
            bias_flops = active_elements_count if len(self.inputs) == 3 else 0
            overall_flops = overall_conv_flops + bias_flops

        return int(overall_flops)

__all__ = ["Conv1dInt"]