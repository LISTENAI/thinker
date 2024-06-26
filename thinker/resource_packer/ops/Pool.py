import math
import numpy as np
from typing import Any, Dict, List, Optional

from ...graph import Tensor
from .._type._ctype import tffi
from ...enum_defines import Layout, DevType, ALIGN16 , ALIGN8
from .base import Operator, OperatorAttrs, CPUPoolLayout, register_op
from .utils import QuantType, CeilMode, attr2tuple, calc_pool2d_output_shape

class PoolAttrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}) -> None:
        super().__init__(attrs, "poolAttrs")

    def checkparams(self) -> None:
        ceil = CeilMode.from_int(self.attrs.get("ceil_mode", 0))

        kernels = self.attrs.get("kernel_shape")
        kernels = attr2tuple(kernels, (1, 1))
        assert kernels[0] in (1, 2, 3, 4, 5), "kernel_w for maxpool:{} exceed limit".format(kernels[0])
        assert kernels[1] in (1, 2, 3, 4, 5), "kernel_h for maxpool:{} exceed limit".format(kernels[1])

        strides = self.attrs.get("strides")
        strides = attr2tuple(strides, (1, 1))
        assert strides[0] in (1, 2, 4), "stride_h for convolution:{} exceed limit".format(strides[0])
        assert strides[1] in (1, 2, 4), "stride_w for convolution:{} exceed limit".format(strides[1])

        pads = self.attrs.get("pads")
        pads = attr2tuple(pads, (0, 0, 0, 0))
        assert pads[0] in (0, 1, 2, 3, 4), "pad_left for convolution:{} exceed limit".format(pads[0])
        assert pads[1] in (0, 1, 2, 3, 4), "pad_right for convolution:{} exceed limit".format(pads[1])
        assert pads[2] in (0, 1, 2, 3, 4), "pad_up for convolution:{} exceed limit".format(pads[2])
        assert pads[3] in (0, 1, 2, 3, 4), "pad_down for convolution:{} exceed limit".format(pads[3])

        layout = Layout.from_str(self.attrs.get("layout", "NCHW"))
        assert layout in {Layout.NCHW, Layout.NHWC}
        quant_type = QuantType.from_str(
            self.attrs.get("platform_quant", "normal_quant")
        )

        self.attrs["ceil"] = ceil
        self.attrs["layout"] = layout
        self.attrs["quant_type"] = quant_type

    def serialize(self) -> bytes:
        attrs = tffi.new("PoolAttrs *")

        attrs.ceil = self.attrs["ceil"].value
        attrs.kernel = self.attrs["kernel_shape"]
        attrs.stride = self.attrs["strides"]
        attrs.pad = self.attrs["pads"]
        attrs.layout = self.attrs["layout"].value
        attrs.quant_type = self.attrs["quant_type"].value

        return bytes(tffi.buffer(attrs))


@register_op
class MaxPool(Operator, CPUPoolLayout):
    def __init__(self, attrs={}):
        self.attrs = PoolAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1
        X = inputs[0]
        assert len(X.shape) == 4
        assert X.dtype == np.int8, "input type of avgpool must be int8"
        shape = calc_pool2d_output_shape(
            input_shape=X.shape,
            kernels=self.attrs["kernel_shape"],
            strides=self.attrs["strides"],
            dilations=(1, 1),
            pads=self.attrs["pads"],
            ceil_mode=self.attrs.get("ceil_mode", 0),
            layout=self.attrs.get("layout", "NCHW"),
        )
        Y = X.clone(shape=tuple(shape),scale=X.scale)
        self.outputs = [Y]

    def get_workspace(self, dev_type: DevType) -> List[Tensor]:
        input_data = self.inputs[0]
        kernel_c = input_data.shape[1]
        h = input_data.shape[2]
        w = input_data.shape[3]

        strides = self.attrs["strides"]
        stride_w = strides[1]
        out_size = self.outputs[0].nbytes

        data_size = (
            ALIGN8(kernel_c)
            * ((w + 8 * stride_w - 1) // (8 * stride_w))
            * (8 * stride_w)
            * h
        )
        workspace_size = 0
        if data_size > 65536:
            data_size1 = (
                ALIGN8(kernel_c)
                * ((w + 8 * stride_w - 1) // (8 * stride_w))
                * (8 * stride_w)
            )
            assert data_size1 <= 65536, "only support H split"
           #workspace_size = out_size // h  * 2
            workspace_size = max(workspace_size, 60*1024)
            workspace_size = max(workspace_size, out_size)
            max_workspace = Tensor.from_shape(
                [workspace_size], np.int8, self.inputs[0].mem_type
            )       
        else:
            workspace_size = out_size
            max_workspace = Tensor.from_shape(
                [workspace_size], np.int8, self.inputs[0].mem_type
            )
        return [max_workspace]
     

@register_op
class AvgPool2dInt(Operator, CPUPoolLayout):
    def __init__(self, attrs={}):
        self.attrs = PoolAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1
        X = inputs[0]
        assert len(X.shape) == 4
        assert X.dtype == np.int8, "input type of avgpool must be int8"
        shape = calc_pool2d_output_shape(
            input_shape=X.shape,
            kernels=self.attrs["kernel_shape"],
            strides=self.attrs["strides"],
            dilations=(1, 1),
            pads=self.attrs["pads"],
            ceil_mode=self.attrs["ceil_mode"],
            layout=self.attrs.get("layout", "NCHW"),
        )
        Y = X.clone(shape=tuple(shape),scale=X.scale)
        self.outputs = [Y]

    def get_workspace(self, dev_type: DevType) -> List[Tensor]:
        input_data = self.inputs[0]
        kernel_c = input_data.shape[1]
        h = input_data.shape[2]
        w = input_data.shape[3]
        intput_size = h * w

        (kernel_h, kernel_w) = self.attrs['kernel_shape']
        kernel_size = kernel_h * kernel_w

        strides = self.attrs["strides"]
        stride_w = strides[1]

        pads = self.attrs['pads']

        ou_h = self.outputs[0].shape[2]
        ou_w = self.outputs[0].shape[3]
        ou_hw = ou_h * ou_w
        out_size = self.outputs[0].nbytes
        data_size = (
            ALIGN8(kernel_c)
            * ((w + 8 * stride_w - 1) // (8 * stride_w))
            * (8 * stride_w)
            * h
        )
        workspace_size = 0

        if data_size > 65536:
            data_size1 = (
                8
                * ((w + 8 * stride_w - 1) // (8 * stride_w))
                * (8 * stride_w)
                * h
            )
            assert data_size1 <= 65536, "only support channel split"
            kernel_c = 8
        if kernel_size & (kernel_size - 1):
            workspace_size = kernel_c * ou_hw * 8
        else:
            workspace_size = kernel_c * ou_hw * 2
        return [Tensor.from_shape([workspace_size], np.int8, self.inputs[0].mem_type)]


__all__ = ["MaxPool", "AvgPool2dInt"]
