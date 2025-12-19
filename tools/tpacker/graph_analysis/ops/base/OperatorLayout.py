import numpy as np
from typing import List

from ....graph import Tensor
from ....xsympy import is_sympy
from ..utils import calc_expr
from ....enum_defines import Layout, ALIGN2, ALIGN4, ALIGN8, ALIGN16


class LayoutPerfData:
    """Data structure to store layout performance information."""
    
    def __init__(self, kernel: str = None, performance: int = 10):
        self.kernel = kernel
        self.performance = performance
        self.inputs_layout = []
        self.outputs_layout = []


class BaseLayout:
    """Base class for layout optimization and performance analysis."""
    
    def get_layout_perf_venus(self, dynamic_shape) -> List[LayoutPerfData]:
        """Get layout performance data for Venus platform."""
        perf = LayoutPerfData()
        for tensor in self.inputs:
            perf.inputs_layout.append(Layout.NCHW)
        for tensor in self.outputs:
            perf.outputs_layout.append(Layout.NCHW)
        return [perf]

    def get_layout_perf(self, dynamic_shape) -> List[LayoutPerfData]:
        """Get layout performance data."""
        return self.get_layout_perf_venus(dynamic_shape)


class ConvLayout(BaseLayout):
    """Layout optimization for convolution operations."""
    
    def get_layout_perf_venus(self, dynamic_shape) -> List[LayoutPerfData]:
        """Get layout performance data for Venus platform."""
        perfs = []
        input_data = self.inputs[0]
        weight = self.inputs[1]
        channel_in = input_data.shape[1]
        kernel_num = weight.shape[0]
        kernel_c = weight.shape[1]
        strides = self.attrs["strides"]
        stride_h = strides[0]
        stride_w = strides[1] if len(strides) > 1 else 1
        kernels = self.attrs["kernel_shape"]
        kernel_h = kernels[0]
        kernel_w = kernels[1] if len(kernels) > 1 else 1
        group = self.attrs["group"]
        is_dw = (group == kernel_num) and (kernel_c == 1)
        platform = self.attrs.get("platform", "venus")

        h = input_data.shape[2]
        h = calc_expr(str(h), dynamic_shape) if is_sympy(h) else h
        w = input_data.shape[3] if len(input_data.shape) > 3 else 1
        w = calc_expr(str(w), dynamic_shape) if is_sympy(w) else w

        if platform == 'venusA':
            align_data_size_nchw = ALIGN4(channel_in) * ((w + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * kernel_h
            align_data_size_ncwh = ALIGN4(channel_in) * ((h + 8 * stride_h - 1) // (8 * stride_h)) * (8 * stride_h) * kernel_w
        else:
            align_data_size_nchw = ALIGN2(channel_in) * ((w + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * kernel_h
            align_data_size_ncwh = ALIGN2(channel_in) * ((h + 8 * stride_h - 1) // (8 * stride_h)) * (8 * stride_h) * kernel_w

        if is_sympy(align_data_size_nchw):
            align_data_size_nchw = calc_expr(str(align_data_size_nchw), dynamic_shape)
        if is_sympy(align_data_size_ncwh):
            align_data_size_ncwh = calc_expr(str(align_data_size_ncwh), dynamic_shape)

        perf = LayoutPerfData()
        if platform == "arcs":
            if align_data_size_nchw > 16384 and align_data_size_ncwh <= 16384:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCWH)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCWH)
                perf.performance = 5
            elif align_data_size_nchw <= 16384:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCHW)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCHW)
            else:
                raise ValueError("Unsupported data size configuration!")
        elif platform in {"venus", "venusA"}:
            threshold = 32768 if is_dw else 65536
            if align_data_size_nchw > threshold and align_data_size_ncwh <= threshold:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCWH)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCWH)
                perf.performance = 5
            elif align_data_size_nchw <= threshold:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCHW)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCHW)
            else:
                raise ValueError("Unsupported data size configuration!")
        perfs.append(perf)
        return perfs


class PoolLayout(BaseLayout):
    """Layout optimization for pooling operations."""
    
    def get_layout_perf_venus(self, dynamic_shape) -> List[LayoutPerfData]:
        """Get layout performance data for Venus platform."""
        perfs = []
        input_data = self.inputs[0]
        kernel_num = input_data.shape[0]
        kernel_c = input_data.shape[1]
        strides = self.attrs["strides"]
        stride_h = strides[0]
        stride_w = strides[1] if len(strides) > 1 else 1
        platform = self.attrs["platform"]
        kernels = self.attrs["kernel_shape"]
        kernel_h = kernels[0]
        kernel_w = kernels[1] if len(kernels) > 1 else 1

        h = input_data.shape[2]
        h = calc_expr(str(h), dynamic_shape) if is_sympy(h) else h
        w = input_data.shape[3] if len(input_data.shape) > 3 else 1
        w = calc_expr(str(w), dynamic_shape) if is_sympy(w) else w

        if platform == 'venusA':
            align_data_size_nchw = ALIGN4(kernel_c) * ((w + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * kernel_h
            align_data_size_ncwh = ALIGN4(kernel_c) * ((h + 8 * stride_h - 1) // (8 * stride_h)) * (8 * stride_h) * kernel_w
        else:
            align_data_size_nchw = ALIGN2(kernel_c) * ((w + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * kernel_h
            align_data_size_ncwh = ALIGN2(kernel_c) * ((h + 8 * stride_h - 1) // (8 * stride_h)) * (8 * stride_h) * kernel_w

        if is_sympy(align_data_size_nchw):
            align_data_size_nchw = calc_expr(str(align_data_size_nchw), dynamic_shape)
        if is_sympy(align_data_size_ncwh):
            align_data_size_ncwh = calc_expr(str(align_data_size_ncwh), dynamic_shape)

        perf = LayoutPerfData()
        if platform == "arcs":
            if align_data_size_nchw > 16384 and align_data_size_ncwh <= 16384:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCWH)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCWH)
                perf.performance = 5
            elif align_data_size_nchw <= 16384:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCHW)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCHW)
            elif kernel_h > 5 or kernel_w > 5:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCHW)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCHW)
            else:
                raise ValueError("Unsupported data size configuration!")
        elif platform in {"venus", "venusA"}:
            if align_data_size_nchw > 65536 and align_data_size_ncwh <= 65536:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCWH)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCWH)
                perf.performance = 5
            elif align_data_size_nchw <= 65536:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCHW)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCHW)
            elif kernel_h > 7 or kernel_w > 7:
                for tensor in self.inputs:
                    perf.inputs_layout.append(Layout.NCHW)
                for tensor in self.outputs:
                    perf.outputs_layout.append(Layout.NCHW)
            else:
                raise ValueError("Unsupported data size configuration!")
        perfs.append(perf)
        return perfs

    def get_layout_perf_cpu(self) -> List[LayoutPerfData]:
        """Get layout performance data for CPU platform."""
        perfs = []
        in_channel = self.inputs[0].shape[1]
        out_channel = self.outputs[0].shape[1]

        in_layouts = [Layout.NCHW]
        if len(self.inputs[0].shape) == 4:
            if self.inputs[0].dtype == np.float32:
                if in_channel % 4 == 0:
                    in_layouts.append(Layout.NC4HW4)
            elif self.inputs[0].dtype == np.int8:
                if in_channel % 8 == 0:
                    in_layouts.append(Layout.NC8HW8)
            else:
                raise ValueError("Unsupported data type!")

        output_layout = Layout.NCHW
        if len(self.outputs[0].shape) == 4:
            if self.outputs[0].dtype == np.float32:
                if out_channel % 4 == 0:
                    output_layout = Layout.NC4HW4
            elif self.outputs[0].dtype == np.int8:
                if out_channel % 8 == 0:
                    output_layout = Layout.NC8HW8
            else:
                raise ValueError("Unsupported data type!")

        for layout in in_layouts:
            perf = LayoutPerfData()
            for tensor in self.inputs:
                perf.inputs_layout.append(layout)
            for tensor in self.outputs:
                perf.outputs_layout.append(output_layout)
            if layout != Layout.NCWH:
                perf.performance = 5
            perfs.append(perf)
        return perfs


__all__ = ["LayoutPerfData", "BaseLayout", "ConvLayout", "PoolLayout"]