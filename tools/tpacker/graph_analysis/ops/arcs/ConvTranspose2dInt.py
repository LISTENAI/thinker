import math
import numpy as np

from ..utils import AutoPad, CeilMode
from ....enum_defines import Layout, MemType, ALIGN2, ALIGN4, ALIGN8, ALIGN16


def get_ConvTranspose2dInt_workspace(
    data, weight, bias, out, 
    kernels=None, strides=(1, 1), dilations=(1, 1),
    pads=(0, 0, 0, 0), groups=1, auto_pad=AutoPad.NOTSET
):
    """
    Calculate the workspace size for ConvTranspose2dInt operation.
    
    Args:
        data: Input tensor
        weight: Weight tensor
        bias: Bias tensor
        out: Output tensor
        kernels: Kernel size (optional)
        strides: Stride size (default: (1, 1))
        dilations: Dilation size (default: (1, 1))
        pads: Padding size (default: (0, 0, 0, 0))
        groups: Number of groups (default: 1)
        auto_pad: Auto padding mode (default: AutoPad.NOTSET)
        
    Returns:
        int: Workspace size
        
    Raises:
        AssertionError: If input or output memory type is not SHARE_MEM
    """
    assert data.mem_type == MemType.SHARE_MEM and out.mem_type == MemType.SHARE_MEM
    return 0


def ConvTranspose2dInt_weight_rearrange(
    data, weight, out, 
    kernels=None, strides=(1, 1), dilations=(1, 1),
    pads=(0, 0, 0, 0), group=1, weight_bits=8
):
    """
    Rearrange ConvTranspose2dInt weights for optimized computation.
    
    Args:
        data: Input tensor
        weight: Weight tensor
        out: Output tensor
        kernels: Kernel size (optional)
        strides: Stride size (default: (1, 1))
        dilations: Dilation size (default: (1, 1))
        pads: Padding size (default: (0, 0, 0, 0))
        group: Number of groups (default: 1)
        weight_bits: Weight bit width (default: 8)
        
    Returns:
        new_weight: Rearranged weight tensor
        
    Raises:
        AssertionError: If input layout is not supported
        AssertionError: If kernel size exceeds supported limit
    """
    kernel_h, kernel_w = kernels
    stride_h, stride_w = strides

    new_weight = weight.clone()
    new_weight.data = weight.data

    if data.layout == Layout.NCWH and weight.layout == Layout.NCHW:
        new_weight.data = weight.data.transpose(0, 1, 3, 2)
        new_weight.shape = new_weight.data.shape
        new_weight.dtype = Layout.NCWH

    if new_weight.layout in (Layout.NHWC8, Layout.NWHC8):
        return new_weight

    if kernel_h <= 12 and kernel_w <= 12:
        kernel_num = weight.shape[1]
        kernel_c = weight.shape[0]
        assert kernel_c * group == data.shape[1]
        assert kernel_h == weight.shape[2]
        assert kernel_w == weight.shape[3]

        h_in = data.shape[2]
        w_in = data.shape[3]
        data_size = ALIGN8(kernel_c) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * h_in
        assert group == 1, "deconv does not support depthwise!"

        num_input_align = ALIGN8(kernel_c)
        num_output_align = ALIGN2(kernel_num)

        weight_transpose = weight.data.transpose(1, 0, 2, 3)
        new_weight_transpose = np.zeros(
            (num_output_align, num_input_align, kernel_h, kernel_w),
            weight.dtype
        )

        for p in range(kernel_num):
            for q in range(kernel_c):
                new_weight_transpose[p, q, :, :] = weight_transpose[p, q, :, :]

        new_weight_transpose = np.flip(new_weight_transpose, (2, 3))
        new_weight_transpose = new_weight_transpose.transpose(0, 2, 3, 1)
        new_weight_transpose = new_weight_transpose.reshape(-1, 2, kernel_h, kernel_w, num_input_align)
        new_weight_transpose = new_weight_transpose.transpose(0, 2, 3, 1, 4)
        shape = new_weight_transpose.shape
        new_weight_transpose = new_weight_transpose.reshape(shape[0], shape[1], shape[2], shape[3], -1, 8)
        new_weight_transpose = new_weight_transpose.transpose(0, 1, 2, 4, 3, 5)

        new_weight = weight.clone()
        new_weight.data = new_weight_transpose
        new_weight.shape = new_weight_transpose.shape
        new_weight.bits = np.float32(weight_bits / 8.0)

        if data.layout == Layout.NCHW:
            new_weight.layout = Layout.NHWC8
        elif data.layout == Layout.NCWH:
            new_weight.layout = Layout.NWHC8

        return new_weight

    raise AssertionError(f"ConvTranspose2dInt does not support kernel size: {kernels}!")


__all__ = ["get_ConvTranspose2dInt_workspace", "ConvTranspose2dInt_weight_rearrange"]