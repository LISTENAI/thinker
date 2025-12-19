import math
import numpy as np
from typing import List, Tuple
from ...graph import Tensor
from ...xsympy import is_sympy
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op, ConvLayout
from ...enum_defines import DevType, MemType, Layout, ALIGN2, ALIGN8, ALIGN16
from .utils import (
    QuantType,
    RoundMethod,
    attr2tuple,
    calc_conv2d_output_shape,
    calc_expr,
)

class Conv2dIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check and validate the parameters for Conv2dInt operation."""
        platform = self.attrs.get("platform", "venus")
        quant_type = (
            RoundMethod.from_str(self.attrs.get("quant_mode"))
            if platform in ["arcs", "venusA"]
            else QuantType.from_str(self.attrs.get("platform_quant"))
        )
        self.attrs["quant_type"] = quant_type

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

        # Validate dilations
        dilations = attr2tuple(self.attrs.get("dilations"), (1, 1))
        if platform == "venus":
            assert dilations == (1, 1), "Dilations must be (1, 1) for venus platform"
        else:
            assert dilations[0] in {1, 2, 4, 8}, "Dilation width exceeds limit"
            assert dilations[1] in {1, 2, 4, 8}, "Dilation height exceeds limit"

        # Validate kernel shape
        kernels = attr2tuple(self.attrs.get("kernel_shape"), (1, 1))
        if platform == "venus":
            assert 1 <= kernels[0] <= 5, "Kernel width exceeds limit"
            assert 1 <= kernels[1] <= 5, "Kernel height exceeds limit"
        else:
            if dilations != (1, 1):
                assert 1 <= kernels[0] <= 12, "Kernel width exceeds limit"
                assert 1 <= kernels[1] <= 12, "Kernel height exceeds limit"
            else:
                assert 1 <= kernels[0] <= 5, "Kernel width exceeds limit"
                assert 1 <= kernels[1] <= 5, "Kernel height exceeds limit"

        # Validate pads
        pads = attr2tuple(self.attrs.get("pads"), (0, 0, 0, 0))
        if platform == "venus":
            for pad in pads:
                assert 0 <= pad <= 4, "Pad exceeds limit for venus platform"
        else:
            for pad in pads:
                assert 0 <= pad <= 11, "Pad exceeds limit for arcs/venusA platform"

        # Validate strides
        strides = attr2tuple(self.attrs.get("strides"), (1, 1))
        assert strides[0] in {1, 2, 4}, "Stride width exceeds limit"
        assert strides[1] in {1, 2, 4}, "Stride height exceeds limit"

        # Additional checks
        assert (
            kernels[0] >= strides[0] and kernels[1] >= strides[1]
        ), "Kernel size must be >= stride size"
        assert (
            pads[0] <= kernels[0] and pads[2] <= kernels[0]
        ), "Pad width exceeds kernel width"
        assert (
            pads[1] <= kernels[1] and pads[3] <= kernels[1]
        ), "Pad height exceeds kernel height"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the Conv2dInt operation."""
        attrs = tffi.new("Conv2dIntAttrs *")
        attrs.dilation = self.attrs["dilations"]
        attrs.kernel = self.attrs["kernel_shape"]
        attrs.pad = self.attrs["pads"]
        attrs.stride = self.attrs["strides"]
        attrs.group = self.attrs["group"]
        attrs.quant_type = self.attrs["quant_type"].value
        attrs.act_type = self.attrs.get("act_type", 0)
        return bytes(tffi.buffer(attrs))

@register_op
class Conv2dInt(Operator, ConvLayout):
    def __init__(self, attrs={}):
        self.attrs = Conv2dIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) in (2, 3)

        X = inputs[0]
        W = inputs[1]
        assert len(X.shape) == 4, "Input must be 4D"
        assert len(W.shape) == 4, "Weight must be 4D"
        assert X.dtype == np.int8, "Input data type must be int8"
        assert W.dtype == np.int8, "Weight data type must be int8"
        if len(inputs) == 3:
            assert inputs[2].dtype in (np.int16, np.int32), "Bias data type must be int16 or int32"

        # Infer shape
        shape = calc_conv2d_output_shape(
            X.shape,
            W.shape,
            self.attrs["kernel_shape"],
            self.attrs["strides"],
            self.attrs["dilations"],
            self.attrs["pads"],
            self.attrs["group"],
        )

        # Infer scale
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x[0], 2) if isinstance(scale_x, tuple) else math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        if X.scale != -1:
            assert X.scale == int(temp), "Scale mismatch"
        else:
            self.inputs[0].scale = int(temp)

        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w[0], 2) if isinstance(scale_w, tuple) else math.log(scale_w, 2)
        assert abs(temp - int(temp)) < 0.000001
        self.inputs[1].scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o[0], 2) if isinstance(scale_o, tuple) else math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        # Check shape compatibility
        group = self.attrs.get("group", 1)
        kernels = self.attrs.get("kernel_shape")
        pads = self.attrs.get("pads")
        x_c = calc_expr(str(X.shape[1]), dynamic_shape) if is_sympy(X.shape[1]) else X.shape[1]
        x_h = calc_expr(str(X.shape[2]), dynamic_shape) if is_sympy(X.shape[2]) else X.shape[2]
        x_w = calc_expr(str(X.shape[3]), dynamic_shape) if is_sympy(X.shape[3]) else X.shape[3]
        assert x_c == W.shape[1] * group, "Channel mismatch"
        assert kernels[0] == W.shape[2] and kernels[1] == W.shape[3], "Kernel size mismatch"
        if len(pads) == 4:
            assert (
                x_w + pads[0] + pads[2] >= W.shape[3]
                and x_h + pads[1] + pads[3] >= W.shape[2]
            ), "Input and weight size mismatch"
        elif len(pads) == 2:
            assert (
                x_w + pads[0] * 2 >= W.shape[3]
                and x_h + pads[1] * 2 >= W.shape[2]
            ), "Input and weight size mismatch"

        # Infer type and return output
        platform = self.attrs.get("platform", "venus")
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
            assert parameter_bits in {4, 8}, f"weight type:{W.dtype} must match weight bits:{parameter_bits}"
            assert output_bits in (8, 32)
            dtype = np.int8 if output_bits == 8 else np.int32
        elif platform == "venusA":
            assert X.dtype == np.int8
            assert data_bits == 8, f"data type:{X.dtype} must match data bits:{data_bits}"
            assert W.dtype in (np.int8, np.int16)
            assert parameter_bits in {4, 8, 16, 32}, f"weight type:{W.dtype} must match weight bits:{parameter_bits}"
            assert output_bits in (8, 16, 32)
            dtype = (
                np.int8
                if output_bits == 8
                else np.int16
                if output_bits == 16
                else np.int32
            )
        else:
            raise ValueError("Unsupported platform")

        Y = X.clone(shape=tuple(shape), scale=int(temp), dtype=dtype)
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the Conv2dInt operation."""
        platform = self.attrs.get("platform", "venus")
        platform_module = __import__(
            f"tpacker.graph_analysis.ops.{platform}", fromlist=[""]
        )
        bias = self.inputs[2] if len(self.inputs) == 3 else None
        workspace_size = platform_module.get_Conv2dInt_workspace(
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
        """Pack the parameters for the Conv2dInt operation."""
        platform = self.attrs.get("platform", "venus")
        platform_module = __import__(
            f"tpacker.graph_analysis.ops.{platform}", fromlist=[""]
        )
        weight_bits = self.attrs["parameter_bits"]
        new_weight = platform_module.Conv2dInt_weight_rearrange(
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
        """Calculate the number of floating-point operations (FLOPs) for the Conv2dInt operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        y_out = Y.shape[1]
        y_h = Y.shape[2]
        y_w = Y.shape[3]

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
        kernel_w = weight.shape[3]
        weight_dims = list(weight.shape[1:])
        output_dims = list(Y.shape[1:])
        weight_count = int(np.prod(weight_dims))
        active_elements_count = int(np.prod(output_dims))

        if 1 == kernel_c and group == kernel_num:
            overall_conv_flops = 2 * (kernel_h * kernel_w + y_out) * kernel_c * group * y_w * y_h
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

    def sub_layout_convert(self):
        """Convert the layout of input and weight tensors if necessary."""
        data = self.inputs[0]
        weight = self.inputs[1]
        if data.layout == Layout.NCWH and weight.layout == Layout.NCHW:
            new_weight = weight.clone()
            new_weight.data = weight.data.transpose(0, 1, 3, 2)
            new_weight.shape = new_weight.data.shape
            new_weight.layout = Layout.NCWH
            self.inputs[1].update(shape=new_weight.shape, data=new_weight.data, layout=new_weight.layout)

__all__ = ["Conv2dInt"]