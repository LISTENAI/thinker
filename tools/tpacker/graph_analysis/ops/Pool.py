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
        kernels = self.attrs.get("kernel_shape")
        kernels = attr2tuple(kernels, (1, 1))
        strides = self.attrs.get("strides")
        strides = attr2tuple(strides, (1, 1))
        pads = self.attrs.get("pads")
        pads = attr2tuple(pads, (0, 0, 0, 0))

        assert (kernels[-1] >= strides[-1] and kernels[-2] >= strides[-2]), "Kernel and stride sizes do not match"
        assert (pads[0] <= kernels[-2] and pads[2] <= kernels[-2]), "Pad height exceeds kernel height"
        assert (pads[1] <= kernels[-1] and pads[3] <= kernels[-1]), "Pad width exceeds kernel width"

        layout = Layout.from_str(Layout, self.attrs.get("layout", "NCHW"))
        assert layout in {Layout.NCHW, Layout.NHWC}, "Invalid layout for Pool operation"

        self.attrs["ceil"] = ceil
        self.attrs["layout"] = layout

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
class MaxPool(Operator):
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

        h_in = calc_expr(str(X.shape[2]), dynamic_shape) if is_sympy(X.shape[2]) else X.shape[2]
        w_in = calc_expr(str(X.shape[3]), dynamic_shape) if is_sympy(X.shape[3]) else X.shape[3]

        shape = calc_pool2d_output_shape(X.shape, kernels, strides, (1, 1), pads, ceil_mode, layout)
        Y = X.clone(shape=tuple(shape), scale=X.scale)
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the MaxPool operation."""
        input_data = self.inputs[0]
        kernel_c, h_in, w_in = input_data.shape[1:4]
        kernel_h, kernel_w = self.attrs["kernel_shape"]
        stride_h, stride_w = self.attrs["strides"]
        out_size = self.outputs[0].nbytes
        platform = self.attrs.get("platform", "venus")

        data_size = ALIGN8(kernel_c) * h_in * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
        workspace_size = 0

        if platform == 'venus':
            if data_size > 65536:
                data_size_withouth = ALIGN8(kernel_c) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * kernel_h
                assert data_size_withouth <= 65536, "only support H split"
                #workspace_size = out_size // h  * 2
                workspace_size = max(workspace_size, 60*1024)
                workspace_size = max(workspace_size, out_size)     
            elif self.outputs[0].mem_type != MemType.SHARE_MEM:
                workspace_size = out_size
        else:
            if self.outputs[0].mem_type != MemType.SHARE_MEM:
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
class AvgPool2dInt(Operator):
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
        ceil_mode = self.attrs["ceil_mode"]
        layout = self.attrs.get("layout", "NCHW")

        h_in = calc_expr(str(X.shape[2]), dynamic_shape) if is_sympy(X.shape[2]) else X.shape[2]
        w_in = calc_expr(str(X.shape[3]), dynamic_shape) if is_sympy(X.shape[3]) else X.shape[3]

        shape = calc_pool2d_output_shape(X.shape, kernels, strides, (1, 1), pads, ceil_mode, layout)
        Y = X.clone(shape=tuple(shape), scale=X.scale)
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
        out_size = self.outputs[0].nbytes

        if (kernel_h == h_in + pads[0] + pads[-2]) and (kernel_w == w_in + pads[1] + pads[-1]):
            split_num = 1
            split_ch = c_in
            c_last = 0
            input_condition = ALIGN8(c_in) * h_in * w_in
            if input_condition > 65536:
                split_ch = 8
                split_num = math.floor(c_in * 8)
                c_last = c_in - split_num * 8
            if kernel_size & (kernel_size - 1):
                workspace_size = max(h_in * w_in, split_ch * ou_h * ou_w * 4) + split_ch * ou_h * ou_w * 4
            else:
                workspace_size = max(h_in * w_in, split_ch * ou_h * ou_w * 4)
        else:
            split_num = 1
            split_ch = c_in
            c_last = 0
            input_condition = ALIGN8(c_in) * h_in * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
            if input_condition > 65536:
                split_ch = 8
                split_num = math.floor(c_in * 8)
                c_last = c_in - split_num * 8
            if kernel_size & (kernel_size - 1):
                workspace_size = split_ch * ou_h * ou_w * 8
            else:
                platform = self.attrs.get("platform", "venus")
                workspace_size = split_ch * ou_h * ou_w * 2 if platform == "venus" else split_ch * ou_h * ou_w * 4

        return [Tensor.from_shape([workspace_size], np.int8, self.inputs[0].mem_type)]

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