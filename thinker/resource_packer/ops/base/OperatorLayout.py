import numpy as np
from typing import List

from ....graph import Tensor
from ....enum_defines import Layout, ALIGN16, ALIGN8, ALIGN2


class LayoutPerfData:
    def __init__(self, kernel: str = None, performance: int = 10):
        self.kernel = kernel
        self.performance = performance
        self.inputs_layout = []
        self.outputs_layout = []


class BaseLayout(object):
    def get_layout_perf_venus(self) -> List[LayoutPerfData]:
        perfs = []
        perf = LayoutPerfData()
        for tensor in self.inputs:
            perf.inputs_layout.append(Layout.NCHW)
        for tensor in self.outputs:
            perf.outputs_layout.append(Layout.NCHW)
        perfs.append(perf)
        return perfs

    def get_layout_perf(self) -> List[LayoutPerfData]:
        perfs = self.get_layout_perf_venus()
        return perfs


class CPUConvLayout(BaseLayout):
    def get_layout_perf_venus(self) -> List[LayoutPerfData]:
        perfs = []
        input = self.inputs[0]
        weight = self.inputs[1]

        kernel_num = weight.shape[0]
        kernel_c = weight.shape[1]

        group = self.attrs["group"]
        strides = self.attrs["strides"]
        stride_h = strides[0]
        if len(strides) == 1:
            stride_w = 1
        else:
            stride_w = strides[1]

        kernels = self.attrs["kernel_shape"]
        kernel_h = kernels[0]
        if len(kernels) == 1:
            kernel_w = 1
        else:
            kernel_w = kernels[1]

        h = input.shape[2]
        if len(input.shape) == 3:
            w = 1
        else:
            w = input.shape[3]

        data_size = (
            ALIGN8(kernel_c)
            * ((w + 8 * stride_w - 1) // (8 * stride_w))
            * (8 * stride_w)
            * h
        )
        data_size1 = (
            ALIGN8(kernel_c)
            * ((w + 8 * stride_w - 1) // (8 * stride_w))
            * (8 * stride_w)
        )
        data_size2 = ALIGN8(kernel_c) * h

        if group == kernel_c and group == kernel_num:  # depthwise conv
            num_input_align = ALIGN16(kernel_c)
            kernel_size = num_input_align * kernel_h * kernel_w
            assert (
                kernel_size * weight.dtype.itemsize <= 32768
            ), "kernel size of depthwise conv must less 32KB"
        elif group != 1:  # group conv
            assert kernel_num % group == 0
            num_input_align = ALIGN8(kernel_c)
            num_output_align = ((kernel_num // group + 1) // 2) * 2
            kernel_size = num_input_align * num_output_align * kernel_h * kernel_w
        else:  # common conv
            num_input_align = ALIGN8(kernel_c)
            num_output_align = ALIGN2(kernel_num)
            kernel_size = num_input_align * num_output_align * kernel_h * kernel_w

        perf = LayoutPerfData()
        if data_size > 65536 and kernel_size <= 32768:
            if data_size1 > 65536 and data_size2 <= 65536:
                for i in self.inputs:
                    perf.inputs_layout.append(Layout.NCWH)
                for i in self.outputs:
                    perf.outputs_layout.append(Layout.NCWH)
                perf.performance = 5
            elif data_size1 > 65536 and data_size2 > 65536:
                raise ValueError(f"data size exceed limit {data_size}")
            else:
                for i in self.inputs:
                    perf.inputs_layout.append(Layout.NCHW)
                for i in self.outputs:
                    perf.outputs_layout.append(Layout.NCHW)
        else:
            for i in self.inputs:
                perf.inputs_layout.append(Layout.NCHW)
            for i in self.outputs:
                perf.outputs_layout.append(Layout.NCHW)
        perfs.append(perf)

        return perfs


class CPUPoolLayout(BaseLayout):
    def get_layout_perf_venus(self) -> List[LayoutPerfData]:
        perfs = []
        input = self.inputs[0]

        kernel_num = input.shape[0]
        kernel_c = input.shape[1]

        strides = self.attrs["strides"]
        stride_h = strides[0]
        if len(strides) == 1:
            stride_w = 1
        else:
            stride_w = strides[1]

        h = input.shape[2]
        if len(input.shape) == 3:
            w = 1
        else:
            w = input.shape[3]

        data_size = (
            ALIGN8(kernel_c)
            * ((w + 8 * stride_w - 1) // (8 * stride_w))
            * (8 * stride_w)
            * h
        )
        data_size1 = (
            ALIGN8(kernel_c)
            * ((w + 8 * stride_w - 1) // (8 * stride_w))
            * (8 * stride_w)
        )
        data_size2 = ALIGN8(kernel_c) * h

        perf = LayoutPerfData()
        if data_size > 65536:
            if data_size1 > 65536 and data_size2 <= 65536:
                for i in self.inputs:
                    perf.inputs_layout.append(Layout.NCWH)
                for i in self.outputs:
                    perf.outputs_layout.append(Layout.NCWH)
                perf.performance = 5
            elif data_size1 > 65536 and data_size2 > 65536:
                raise ValueError(f"data size exceed limit {data_size}")
            else:
                for i in self.inputs:
                    perf.inputs_layout.append(Layout.NCHW)
                for i in self.outputs:
                    perf.outputs_layout.append(Layout.NCHW)
        else:
            for i in self.inputs:
                perf.inputs_layout.append(Layout.NCHW)
            for i in self.outputs:
                perf.outputs_layout.append(Layout.NCHW)
        perfs.append(perf)

        return perfs


__all__ = [
    "LayoutPerfData",
    "BaseLayout",
    "CPUConvLayout",
    "CPUPoolLayout",
]
