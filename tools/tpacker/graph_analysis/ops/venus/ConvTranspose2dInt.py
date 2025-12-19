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
    c_in, h_in, w_in = data.shape[1:4]
    kernel_ch, kernel_num, kernel_h, kernel_w = weight.shape[0:4]
    stride_w = strides[1]

    data_size = ALIGN8(c_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * h_in * data.dtype.itemsize
    weight_size = ALIGN8(kernel_num) * ALIGN2(kernel_ch) * kernel_h * kernel_w * weight.dtype.itemsize

    assert data.mem_type == MemType.SHARE_MEM and weight.mem_type == MemType.SHARE_MEM and out.mem_type == MemType.SHARE_MEM
    assert data_size <= 65536 and weight_size <= 32768

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
    kernel_h = kernels[0]
    kernel_w = kernels[1]
    stride_h = strides[0]
    stride_w = strides[1]

    if data.layout == Layout.NCWH and weight.layout == Layout.NCHW:
        weight.data = weight.data.transpose(0, 1, 3, 2)
        weight.shape = weight.data.shape
        weight.layout = Layout.NCWH

    if weight.layout in (Layout.NHWC8, Layout.NWHC8):
        return weight

    kernel_num = weight.shape[1]
    kernel_c = weight.shape[0]
    assert kernel_c * group == data.shape[1]
    assert kernel_h == weight.shape[2]
    assert kernel_w == weight.shape[3]

    h_in = data.shape[2]
    w_in = data.shape[3]
    data_size = ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * ALIGN8(kernel_c) * h_in
    assert data_size * data.dtype.itemsize <= 65536, "Input size of deconv exceeds limit"
    assert group == 1, "Deconv does not support depthwise!"

    num_input_align = ALIGN8(kernel_c)
    num_output_align = ALIGN2(kernel_num)
    kernel_size = num_input_align * num_output_align * kernel_h * kernel_w
    assert kernel_size * weight.dtype.itemsize <= 32768

    weight_transpose = weight.data.transpose(1, 0, 2, 3)
    new_weight_transpose = np.zeros((num_output_align, num_input_align, kernel_h, kernel_w), weight.dtype)

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


__all__ = ["get_ConvTranspose2dInt_workspace", "ConvTranspose2dInt_weight_rearrange"]