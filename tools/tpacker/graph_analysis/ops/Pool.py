import math
import numpy as np
from typing import Any, Dict, List, Optional
from ...graph import Tensor
from ...xsympy import is_sympy
from ...resource_packer._type._ctype import tffi
from .base import Operator,OperatorAttrs, PoolLayout, register_op
from ...enum_defines import MemType, Layout, ALIGN4, ALIGN8, ALIGN16
from .utils import attr2tuple, calc_pool2d_output_shape, CeilMode, calc_expr


class PoolAttrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        """Initialize the Pool attributes."""
        super().__init__(attrs, "poolAttrs")

    def checkparams(self) -> None:
        """Check and validate the parameters for the Pool operation."""
        assert "kernel_shape" in self.attrs, "Missing required attribute: kernel_shape"
        assert "strides" in self.attrs, "Missing required attribute: strides"
        assert "pads" in self.attrs, "Missing required attribute: pads"

        ceil = CeilMode.from_int(self.attrs.get("ceil_mode", 0))

        kernels = attr2tuple(self.attrs["kernel_shape"], (1, 1))
        strides = attr2tuple(self.attrs["strides"], (1, 1))
        pads = attr2tuple(self.attrs["pads"], (0, 0, 0, 0))
        assert all(1 <= value <= 255 for value in kernels), "Invalid Pool kernel_shape"
        assert all(1 <= value <= 255 for value in strides), "Invalid Pool strides"
        assert all(0 <= value <= 255 for value in pads), "Invalid Pool pads"

        layout_value = self.attrs.get("layout", "NCHW")
        layout = layout_value if isinstance(layout_value, Layout) else \
            Layout.from_str(Layout, layout_value)
        assert layout in {Layout.NCHW, Layout.NHWC}, "Invalid layout for Pool operation"

        self.attrs["ceil"] = ceil
        self.attrs["layout"] = layout
        self.attrs["kernel_shape"] = kernels
        self.attrs["strides"] = strides
        self.attrs["pads"] = pads

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the Pool operation."""
        attrs = tffi.new("PoolAttrs *")
        attrs.ceil = self.attrs["ceil"].value
        attrs.kernel = self.attrs["kernel_shape"]
        attrs.stride = self.attrs["strides"]
        attrs.pad = self.attrs["pads"]
        attrs.layout = self.attrs["layout"].value
        return bytes(tffi.buffer(attrs))

@register_op
class MaxPool(Operator, PoolLayout):
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        """Initialize the MaxPool operator with given attributes."""
        self.attrs = PoolAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on input."""
        inputs = self.inputs
        assert len(inputs) == 1, "MaxPool operator must have exactly one input"
        X = inputs[0]
        assert len(X.shape) == 4, "Input must be a 4D tensor"
        assert X.dtype == np.int8, "Input type must be int8"

        kernels = self.attrs.get("kernel_shape")
        kernels = attr2tuple(kernels, (1, 1))
        strides = self.attrs.get("strides")
        strides = attr2tuple(strides, (1, 1))
        pads = self.attrs.get("pads")
        pads = attr2tuple(pads, (0, 0, 0, 0))
        ceil_mode = self.attrs.get("ceil_mode", 0)
        layout = self.attrs.get("layout", "NCHW")
        platform = self.attrs.get("platform", "venus")

        assert CeilMode.from_int(ceil_mode) == CeilMode.NO, \
            "MaxPool ceil_mode is not supported by the runtime"
        assert layout in ("NCHW", Layout.NCHW), \
            f"MaxPool on {platform} only supports NCHW"

        h_in = calc_expr(str(X.shape[2]), dynamic_shape) if is_sympy(X.shape[2]) else X.shape[2]
        w_in = calc_expr(str(X.shape[3]), dynamic_shape) if is_sympy(X.shape[3]) else X.shape[3]

        is_global_pool = (kernels[0] == h_in + pads[0] + pads[2] and
                          kernels[1] == w_in + pads[1] + pads[3])
        if not is_global_pool:
            if platform == "venus":
                assert 1 <= kernels[0] <= 5, "MaxPool kernel height exceeds Venus limit"
                assert 1 <= kernels[1] <= 5, "MaxPool kernel width exceeds Venus limit"
                assert all(0 <= pad <= 4 for pad in pads), "MaxPool padding exceeds Venus limit"
            elif platform in ("arcs", "venusA", "venusa"):
                assert 1 <= kernels[0] <= 7, "MaxPool kernel height exceeds platform limit"
                assert 1 <= kernels[1] <= 7, "MaxPool kernel width exceeds platform limit"
                assert all(0 <= pad <= 11 for pad in pads), "MaxPool padding exceeds platform limit"

        if platform in ("venusA", "venusa"):
            assert kernels[0] <= 7 and kernels[1] <= 7, \
                "MaxPool kernel exceeds VenusA runtime limit"
        assert strides[0] in {1, 2, 4} and strides[1] in {1, 2, 4}, \
            "MaxPool only supports stride 1, 2, or 4"
        assert kernels[0] >= strides[0] and kernels[1] >= strides[1], \
            "MaxPool kernel size must be >= stride size"
        assert pads[0] < kernels[0] and pads[2] < kernels[0], \
            "MaxPool height padding must be smaller than kernel height"
        assert pads[1] < kernels[1] and pads[3] < kernels[1], \
            "MaxPool width padding must be smaller than kernel width"


        shape = calc_pool2d_output_shape(X.shape, kernels, strides, (1, 1), pads, ceil_mode, layout)
        Y = X.clone(shape=tuple(shape), scale=X.scale)
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the MaxPool operation."""
        input_data = self.inputs[0]
        kernel_c, h_in, w_in = input_data.shape[1:4]
        kernel_h, kernel_w = self.attrs["kernel_shape"]
        stride_h, stride_w = self.attrs["strides"]
        out_size = int(np.prod(self.outputs[0].shape[1:])) * self.outputs[0].dtype.itemsize
        platform = self.attrs.get("platform", "venus")

        if platform in ("venusA", "venusa"):
            h_eff = h_in + (1 if self.attrs["pads"][0] != 0 else 0)
            data_size = ALIGN4(kernel_c) * h_eff * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
        else:
            data_size = ALIGN8(kernel_c) * h_in * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
        workspace_size = 0

        if platform == 'venus':
            if data_size > 65536:
                assert self.outputs[0].mem_type == MemType.SHARE_MEM, \
                    "Split MaxPool on venus requires SHARE_MEM output"
                data_size_withouth = ALIGN8(kernel_c) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * kernel_h
                assert data_size_withouth <= 65536, "only support H split"
                out_h = self.outputs[0].shape[2]
                split_num = 1
                split_h = h_in
                row_size = ALIGN8(kernel_c) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                while split_h * row_size > 65536 or out_h % split_num:
                    split_num += 1
                    split_h = out_h * stride_h // split_num + kernel_h - stride_h
                    assert split_num <= h_in and split_num <= out_h, "MaxPool only supports H split"
                workspace_size = max(kernel_c * split_h * w_in, out_size)
            else:
                input_size = int(np.prod(input_data.shape[1:])) * input_data.dtype.itemsize
                if input_data.mem_type != MemType.SHARE_MEM:
                    workspace_size += ALIGN4(input_size)
                if self.outputs[0].mem_type != MemType.SHARE_MEM:
                    workspace_size += out_size
        else:
            if platform == "arcs":
                assert self.outputs[0].mem_type == MemType.SHARE_MEM, \
                    "MaxPool on arcs requires SHARE_MEM output"
                input_size = int(np.prod(input_data.shape[1:])) * input_data.dtype.itemsize
                if input_data.mem_type != MemType.SHARE_MEM:
                    workspace_size = input_size
            if platform == "venusA" and self.outputs[0].mem_type == MemType.SHARE_MEM:
                assert data_size <= 32768, "MaxPool on venusA input exceeds runtime limit"
            if platform == "venusA" and self.outputs[0].mem_type != MemType.SHARE_MEM:
                assert ALIGN4(1) * h_in * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) <= 32768, \
                    "MaxPool on venusA has no feasible single-channel PSRAM split"
            if platform != "arcs" and self.outputs[0].mem_type != MemType.SHARE_MEM:
                out_channel_size = int(np.prod(self.outputs[0].shape[2:])) * self.outputs[0].dtype.itemsize
                assert out_channel_size <= 65536, \
                    "MaxPool PSRAM output channel exceeds workspace limit"
                workspace_size = min(out_size, 65536)

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the MaxPool operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        xshape = list(X.shape)
        yshape = list(Y.shape)

        for i, s in enumerate(xshape):
            if is_sympy(s):
                xshape[i] = calc_expr(str(s), dynamic_shape)
        for i, s in enumerate(yshape):
            if is_sympy(s):
                yshape[i] = calc_expr(str(s), dynamic_shape)

        return int(np.prod(yshape))

@register_op
class AvgPool2dInt(Operator, PoolLayout):
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        """Initialize the AvgPool2dInt operator with given attributes."""
        self.attrs = PoolAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on input."""
        inputs = self.inputs
        assert len(inputs) == 1, "AvgPool2dInt operator must have exactly one input"
        X = inputs[0]
        assert len(X.shape) == 4, "Input must be a 4D tensor"
        assert X.dtype == np.int8, "Input type must be int8"

        kernels = self.attrs.get("kernel_shape")
        kernels = attr2tuple(kernels, (1, 1))
        strides = self.attrs.get("strides")
        strides = attr2tuple(strides, (1, 1))
        pads = self.attrs.get("pads")
        pads = attr2tuple(pads, (0, 0, 0, 0))
        ceil_mode = self.attrs.get("ceil_mode", 0)
        assert CeilMode.from_int(ceil_mode) == CeilMode.NO, \
            "AvgPool2dInt ceil_mode is not supported by the runtime"
        layout = self.attrs.get("layout", "NCHW")

        # Infer scale
        scale_x = self.attrs.get("scale_x")
        assert scale_x is not None, "Missing required attribute: scale_x"
        temp = math.log(scale_x[0], 2) if isinstance(scale_x, tuple) else math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        if X.scale != -1:
            assert X.scale == int(temp), "Scale mismatch"
        else:
            self.inputs[0].scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        assert scale_o is not None, "Missing required attribute: scale_o"
        temp = math.log(scale_o[0], 2) if isinstance(scale_o, tuple) else math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        h_in = calc_expr(str(X.shape[2]), dynamic_shape) if is_sympy(X.shape[2]) else X.shape[2]
        w_in = calc_expr(str(X.shape[3]), dynamic_shape) if is_sympy(X.shape[3]) else X.shape[3]

        platform = self.attrs.get("platform", "venus")
        assert platform in ("venus", "arcs", "mars", "venusA", "venusa"), \
            f"Unsupported AvgPool2dInt platform: {platform}"

        assert layout in ("NCHW", Layout.NCHW), \
            f"AvgPool2dInt on {platform} only supports NCHW"

        batch_in = calc_expr(str(X.shape[0]), dynamic_shape) if is_sympy(X.shape[0]) else X.shape[0]
        assert batch_in == 1, f"AvgPool2dInt on {platform} only supports batch=1"

        kernel_size = kernels[0] * kernels[1]
        arcs_platform = platform in ("arcs", "mars")
        shift = int(temp) - X.scale if arcs_platform else X.scale - int(temp)
        if kernel_size & (kernel_size - 1):
            min_shift = 0 if arcs_platform else -30
            assert min_shift <= shift <= 63, "AvgPool2dInt div shift exceeds Luna limits"
        else:
            if platform == "venus":
                assert int(temp) == X.scale, \
                    "Power-of-two AvgPool2dInt on venus requires scale_x == scale_o"
            shift += int(math.log2(kernel_size))
            min_shift = 0 if arcs_platform else -30
            assert min_shift <= shift <= 63, "AvgPool2dInt scale shift exceeds Luna limits"

        if (kernels[0] != h_in + pads[0] + pads[-2]) or (kernels[1] != w_in + pads[1] + pads[-1]):
            if platform == "venus":
                assert 1 <= kernels[0] <= 5, "kernel_w for Conv2dInt exceed limit"
                assert 1 <= kernels[1] <= 5, "kernel_h for Conv2dInt exceed limit"
                assert 0 <= pads[0] <= 4, "pad_left for Conv2dInt exceed limit"
                assert 0 <= pads[1] <= 4, "pad_up for Conv2dInt exceed limit"
                assert 0 <= pads[2] <= 4, "pad_right for Conv2dInt exceed limit"
                assert 0 <= pads[3] <= 4, "pad_down for Conv2dInt exceed limit"
            elif platform in ("arcs", "mars", "venusA", "venusa"):
                assert 1 <= kernels[0] <= 7, "kernel_w for Pool exceed limit"
                assert 1 <= kernels[1] <= 7, "kernel_h for Pool exceed limit"
                assert 0 <= pads[0] <= 11, "pad_left for Conv2dInt exceed limit"
                assert 0 <= pads[1] <= 11, "pad_up for Conv2dInt exceed limit"
                assert 0 <= pads[2] <= 11, "pad_right for Conv2dInt exceed limit"
                assert 0 <= pads[3] <= 11, "pad_down for Conv2dInt exceed limit"

            if platform in ("venus", "arcs", "mars", "venusA", "venusa"):
                assert strides[0] in (1, 2, 4) and strides[1] in (1, 2, 4), \
                    f"AvgPool2dInt on {platform} only supports stride 1, 2, or 4"

            assert (kernels[0] >= strides[0] and kernels[1] >= strides[1]), "Kernel size must be >= stride size"
            assert (pads[0] < kernels[0] and pads[2] < kernels[0]), "Pad height must be smaller than kernel height"
            assert (pads[1] < kernels[1] and pads[3] < kernels[1]), "Pad width must be smaller than kernel width"
        elif arcs_platform:
            assert all(0 <= pad <= 11 for pad in pads), "AvgPool2dInt pad exceeds ARCS limit"
            assert kernels[0] >= strides[0] and kernels[1] >= strides[1], \
                "Kernel size must be >= stride size"
            assert pads[0] < kernels[0] and pads[2] < kernels[0], \
                "Pad height must be smaller than kernel height"
            assert pads[1] < kernels[1] and pads[3] < kernels[1], \
                "Pad width must be smaller than kernel width"

        shape = calc_pool2d_output_shape(X.shape, kernels, strides, (1, 1), pads, ceil_mode, layout)
        if not any(is_sympy(dim) for dim in shape[2:4]):
            assert shape[2] > 0 and shape[3] > 0, "AvgPool2dInt output shape must be positive"
        Y = X.clone(shape=tuple(shape), scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the AvgPool2dInt operation."""
        data = self.inputs[0]
        c_in, h_in, w_in = data.shape[1:4]
        kernel_h, kernel_w = self.attrs["kernel_shape"]
        kernel_size = kernel_h * kernel_w
        stride_h, stride_w = self.attrs["strides"]
        pads = self.attrs["pads"]
        ou_h, ou_w = self.outputs[0].shape[2:4]
        platform = self.attrs.get("platform", "venus")

        if platform == "venus":
            assert data.mem_type == MemType.SHARE_MEM, \
                "AvgPool2dInt on venus requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, \
                "AvgPool2dInt on venus requires SHARE_MEM output"

        if platform in ("venusA", "venusa"):
            h_eff = h_in + (1 if pads[0] != 0 else 0)
            align_c = ((c_in + 3) // 4) * 4
            align_w = ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
            split_ch = c_in if align_c * align_w * h_eff <= 32768 else min(8, c_in)
            assert ALIGN4(split_ch) * align_w * h_eff <= 32768, \
                "AvgPool2dInt has no feasible VenusA channel split"
        elif platform in ("arcs", "mars"):
            align_w = ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
            input_condition = ALIGN8(c_in) * h_in * align_w
            if input_condition <= 16384:
                split_ch = c_in
            else:
                split_num = (input_condition + 16383) // 16384
                while True:
                    raw_split = (c_in + split_num - 1) // split_num
                    split_ch = 8 if raw_split < 8 else ALIGN8(raw_split)
                    split_condition = split_ch * (kernel_h if raw_split < 8 else h_in) * align_w
                    if split_condition <= 16384:
                        break
                    split_num += 1
                    assert split_num <= c_in, "AvgPool2dInt has no valid ARCS channel split"
        else:
            align_w = ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
            input_condition = ALIGN8(c_in) * h_in * align_w
            split_ch = c_in if input_condition <= 65536 else min(8, c_in)
            assert ALIGN8(split_ch) * h_in * align_w <= 65536, \
                "AvgPool2dInt spatial size exceeds the Venus channel-split limit"

        is_global_pool = (kernel_h == h_in + pads[0] + pads[-2]) and (kernel_w == w_in + pads[1] + pads[-1])
        workspace_ch = split_ch
        if platform in ("arcs", "mars"):
            workspace_ch = c_in if is_global_pool else split_ch
            sum_bytes = workspace_ch * ou_h * ou_w * 4
            input_bytes = h_in * w_in if is_global_pool else (
                workspace_ch * h_in * w_in if data.mem_type != MemType.SHARE_MEM else 0)
            denominator_bytes = sum_bytes if kernel_size & (kernel_size - 1) else 0
            divided_bytes = denominator_bytes
            output_bytes = workspace_ch * ou_h * ou_w * self.outputs[0].dtype.itemsize \
                if self.outputs[0].mem_type != MemType.SHARE_MEM else 0
            workspace_size = sum_bytes + ALIGN4(input_bytes) + denominator_bytes + divided_bytes + output_bytes
        elif platform in ("venusA", "venusa"):
            sum_bytes = c_in * ou_h * ou_w * 4 if is_global_pool else split_ch * ou_h * ou_w * 4
            if is_global_pool:
                input_bytes = ALIGN4(h_in * w_in)
                workspace_size = input_bytes + sum_bytes if not kernel_size & (kernel_size - 1) \
                    else max(input_bytes, sum_bytes) + sum_bytes
            else:
                workspace_size = sum_bytes if not kernel_size & (kernel_size - 1) else sum_bytes * 2

            if self.outputs[0].mem_type != MemType.SHARE_MEM:
                workspace_size += c_in * ou_h * ou_w * self.outputs[0].dtype.itemsize
        elif is_global_pool:
            sum_bytes = workspace_ch * ou_h * ou_w * 4
            if kernel_size & (kernel_size - 1):
                if platform == "venus":
                    assert ALIGN4(workspace_ch) * ALIGN8(h_in * w_in) <= 65536, \
                        "Global AvgPool2dInt matrix exceeds the Venus channel-split limit"
                if split_ch == c_in:
                    workspace_size = max(c_in * h_in * w_in, sum_bytes) + sum_bytes
                else:
                    workspace_size = sum_bytes * 2
            elif split_ch == c_in:
                workspace_size = max(h_in * w_in, c_in * ou_h * ou_w * 2)
            else:
                workspace_size = max(h_in * w_in, sum_bytes)
        else:
            if kernel_size & (kernel_size - 1):
                workspace_size = split_ch * ou_h * ou_w * 8
            else:
                workspace_size = split_ch * ou_h * ou_w * (2 if platform == "venus" else 4)

        return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the AvgPool2dInt operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        xshape = list(X.shape)
        yshape = list(Y.shape)

        for i, s in enumerate(xshape):
            if is_sympy(s):
                xshape[i] = calc_expr(str(s), dynamic_shape)
        for i, s in enumerate(yshape):
            if is_sympy(s):
                yshape[i] = calc_expr(str(s), dynamic_shape)

        kernels = self.attrs["kernel_shape"]
        kernel_h = kernels[0]
        kernel_w = kernels[1]

        output_dims = list(Y.shape[1:])
        active_elements_count = int(np.prod(output_dims))
        overall_conv_flops = (kernel_h * kernel_w - 1 + 1) * active_elements_count
        return int(overall_conv_flops)

__all__ = ["MaxPool", "AvgPool2dInt"]
