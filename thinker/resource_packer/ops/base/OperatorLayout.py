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
        # data_size = ((w + 8 * stride_w - 1)// (8 * stride_w)) * (8 * stride_w) * ((kernel_c + 7) // 8) * 8 * h
        data_size_h1 = ((w + 8 * stride_w - 1)// (8 * stride_w)) * (8 * stride_w) * ((kernel_c + 7) // 8) * 8 * 1

        if group == kernel_c and group == kernel_num: # depthwise conv
            num_input_align = ((kernel_c + 15) // 16) * 16
            num_output_align = ((kernel_num + 15) // 16) * 16
            kernel_size = num_input_align * kernel_h * kernel_w
            assert(kernel_size * weight.dtype.itemsize <= 32768), "kernel size of depthwise conv must less 32KB"
        elif group !=1: # group conv
            assert(kernel_num%group == 0)
            num_input_align = ((kernel_c + 7) // 8) * 8
            num_output_align = ((kernel_num//group + 1) // 2) * 2
            kernel_size = num_input_align * num_output_align * kernel_h * kernel_w
        else: # common conv
            num_input_align = ((kernel_c + 7) // 8) * 8
            num_output_align = ((kernel_num + 1) // 2) * 2
            kernel_size = num_input_align * num_output_align * kernel_h * kernel_w

        if data_size_h1 > 65536 and kernel_size <= 32768:
            perf = LayoutPerfData()
            for i in self.inputs:
                perf.inputs_layout.append(Layout.NCWH)
            for i in self.outputs:
                perf.outputs_layout.append(Layout.NCWH)
            perf.performance = 5
        else:
            perf = LayoutPerfData()
            for i in self.inputs:
                perf.inputs_layout.append(Layout.NCHW)
            for i in self.outputs:
                perf.outputs_layout.append(Layout.NCHW)
        perfs.append(perf)

        return perfs


class CPUPoolLayout(BaseLayout):
 def get_layout_perf_cpu(self, inputs:List[Tensor], outputs:List[Tensor]):
        perfs = []
        in_channel = inputs[0].shape[1]
        out_channel = outputs[0].shape[1]

        in_layouts = [Layout.NCHW]
        # only support 4
        if len(inputs[0].shape) == 4:
            if inputs[0].dtype == np.dtype('f4'):
                if (in_channel) % 4 == 0:
                    in_layouts.append(Layout.NC4HW4)
            elif inputs[0].dtype == np.dtype('i1'):
                if (in_channel) % 8 == 0:
                    in_layouts.append(Layout.NC8HW8)
            else:
                raise('not support')

        # only support 4
        output_layout = Layout.NCHW
        if len(outputs[0].shape) == 4:
            if outputs[0].dtype == np.dtype('f4'):
                if (out_channel) % 4 == 0:
                    output_layout = Layout.NC4HW4
            elif outputs[0].dtype == np.dtype('i1'):
                if (out_channel) % 8 == 0:
                    output_layout = Layout.NC8HW8
            else:
                raise('not support')

        for layout in in_layouts:
            perf = LayoutPerfData()
            for i in self.inputs:
                perf.inputs_layout.append(layout)
            for i in self.outputs:
                perf.outputs_layout.append(output_layout)
            if layout != Layout.NCHW:
                perf.performance = 5
            perfs.append(perf)
        return perfs

__all__ = [
    "LayoutPerfData",
    "BaseLayout",
    "CPUConvLayout",
    "CPUPoolLayout",
]
