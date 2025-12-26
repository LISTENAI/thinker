import math
import numpy as np

from ..utils import AutoPad, CeilMode, combine4bit_8bit
from ....enum_defines import Layout, MemType, ALIGN2, ALIGN8


def get_Conv1dInt_workspace(
    data, weight, bias, out, 
    kernels=None, strides=(1, 1), dilations=(1, 1),
    pads=(0, 0, 0, 0), group=1, auto_pad=AutoPad.NOTSET
):
    """
    Calculate the workspace size for Conv1dInt operation.
    
    Args:
        data: Input tensor
        weight: Weight tensor
        bias: Bias tensor
        out: Output tensor
        kernels: Kernel size (optional)
        strides: Stride size (default: (1, 1))
        dilations: Dilation size (default: (1, 1))
        pads: Padding size (default: (0, 0, 0, 0))
        group: Number of groups (default: 1)
        auto_pad: Auto padding mode (default: AutoPad.NOTSET)
        
    Returns:
        int: Workspace size (currently 0)
        
    Raises:
        AssertionError: If input memory type is not SHARE_MEM
    """
    assert data.mem_type == MemType.SHARE_MEM, "input of conv1d must be in share-memory"
    return 0


def Conv1dInt_weight_rearrange(
    data, weight, out, 
    kernels=None, strides=(1, 1), dilations=(1, 1),
    pads=(0, 0, 0, 0), group=1, weight_bits=8
):
    """
    Rearrange Conv1dInt weights for optimized computation.
    
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
    kernel_num = weight.shape[0]
    kernel_c = weight.shape[1]
    kernel_h = 1
    kernel_w = weight.shape[2]
    stride_h = 1
    stride_w = strides[0]
    
    if weight.layout not in {Layout.NCHW, Layout.NCWH}:
        return weight

    new_weight = weight.clone()
    if kernel_h <= 12:
        # Depthwise convolution case
        if group != 1:
            if group == kernel_num:
                assert kernel_c == 1
                num_output_align = ALIGN8(kernel_num)
                kernel_size = num_output_align * kernel_h * kernel_w
                group = num_output_align
                assert kernel_size * weight.dtype.itemsize <= 8192, "kernel size of Conv1dInt must be less than 8KB"

                new_weight_data = np.zeros((num_output_align, 1, kernel_w), weight.dtype)
                for p in range(kernel_num):
                    new_weight_data[p, :, :] = weight.data[p, :, :]

                new_weight_data = new_weight_data.reshape(-1, 8, 1, kernel_h)
                new_weight_data = new_weight_data.transpose(0, 2, 3, 1)
                new_weight.shape = new_weight_data.shape
                new_weight.layout = Layout.NHWC8

                if weight_bits == 4:
                    new_weight.data = combine4bit_8bit(new_weight_data)
                return new_weight
            else:
                raise AssertionError("group Conv1dInt should split in op_divide")
        
        # Common convolution case
        else:
            num_output_align = ALIGN2(kernel_num)
            num_input_align = ALIGN8(kernel_c)

            new_weight_data = np.zeros((num_output_align, num_input_align, kernel_w), weight.dtype)
            for p in range(kernel_num):
                for q in range(kernel_c):
                    new_weight_data[p, q, :] = weight.data[p, q, :]

            new_weight_data = new_weight_data.transpose(0, 2, 1)
            new_weight_data = new_weight_data.reshape(-1, 2, kernel_w, num_input_align)
            new_weight_data = new_weight_data.transpose(0, 2, 1, 3)
            shape = new_weight_data.shape
            new_weight_data = new_weight_data.reshape(shape[0], shape[1], shape[2], -1, 8)
            new_weight_data = new_weight_data.transpose(0, 1, 3, 2, 4)

            assert weight.layout in {Layout.NCHW, Layout.NCWH}
            if weight_bits == 4:
                new_weight.data = combine4bit_8bit(new_weight_data)
            else:
                new_weight.data = new_weight_data

            new_weight.shape = new_weight.data.shape
            new_weight.layout = Layout.NHWC8 if weight.layout == Layout.NCHW else Layout.NWHC8
            return new_weight
    else:
        raise AssertionError("Conv2dInt does not support kernel_size > 12")


__all__ = ["get_Conv1dInt_workspace", "Conv1dInt_weight_rearrange"]
