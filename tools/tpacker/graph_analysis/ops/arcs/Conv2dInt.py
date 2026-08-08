import math
import numpy as np

from ..utils import AutoPad, CeilMode, combine4bit_8bit
from ....enum_defines import Layout, MemType, ALIGN2, ALIGN4, ALIGN8, ALIGN16


def get_Conv2dInt_workspace(
    data, weight, bias, out, 
    kernels=None, strides=(1, 1), dilations=(1, 1),
    pads=(0, 0, 0, 0), group=1, auto_pad=AutoPad.NOTSET
):
    """
    Calculate the workspace size for Conv2dInt operation.
    
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
        int: Workspace size
        
    Raises:
        AssertionError: If input memory type is not supported
    """
    group_align = ALIGN16(group)
    stride_h, stride_w = strides
    kernel_h, kernel_w = kernels
    pad_up, pad_left, pad_down, pad_right = pads
    c_in, h_in, w_in = data.shape[1:4]
    ou_c, ou_h, ou_w = out.shape[1:4]
    kernel_extent_h = (kernel_h - 1) * dilations[0] + 1
    out_bytes = out.dtype.itemsize

    log2n_stride_w = stride_w >> 1
    align_w_shift = 2 + log2n_stride_w
    align_w = ((w_in + (1 << align_w_shift) - 1) // (1 << align_w_shift)) << align_w_shift
    data_size_withouth = align_w * ALIGN8(c_in)
    assert data_size_withouth * kernel_extent_h <= 16384, "input size of conv2dInt on acrs must less than 16KB"

    def split_workspace_size():
        target_elements = 16384 // data_size_withouth
        split_ou_h = max((target_elements - kernel_extent_h + stride_h) // stride_h, 1)
        split_ou_h = min(split_ou_h, ou_h)
        split_in_h = (split_ou_h - 1) * stride_h + kernel_extent_h
        input_size = ALIGN4(c_in * split_in_h * w_in)
        output_size = ou_c * split_ou_h * ou_w * out_bytes
        return input_size + output_size

    is_depthwise = group == c_in and group == ou_c
    if (group == 1 or is_depthwise) and out.mem_type != MemType.SHARE_MEM:
        return min(out.nbytes, split_workspace_size())

    return 0


def Conv2dInt_weight_rearrange(
    data, weight, out, 
    kernels=None, strides=(1, 1), dilations=(1, 1),
    pads=(0, 0, 0, 0), group=1, weight_bits=8
):
    """
    Rearrange Conv2dInt weights for optimized computation.
    
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
    kernel_h, kernel_w = kernels[0:2]
    stride_h, stride_w = strides[0:2]

    new_weight = weight.clone()
    new_weight.data = weight.data

    if new_weight.layout not in {Layout.NCHW, Layout.NCWH}:
        return new_weight

    kernel_c = weight.shape[1]
    kernel_num = weight.shape[0]
    assert kernel_c * group == data.shape[1]
    assert kernel_h == weight.shape[2]
    assert kernel_w == weight.shape[3]

    if data.dtype == np.int8 and weight_bits in (4, 8):
        if group != 1:
            if group == kernel_num:
                assert kernel_c == 1
                num_input_align = kernel_c
                num_output_align = ALIGN8(kernel_num)
                kernel_size = num_output_align * kernel_h * kernel_w
                group = num_output_align
                assert kernel_size * weight.dtype.itemsize <= 8192, "kernel size of Conv2dInt must be less than 8KB"

                new_weight_data = np.zeros(
                    (num_output_align, num_input_align, kernel_h, kernel_w),
                    weight.dtype
                )
                for p in range(kernel_num):
                    for q in range(kernel_c):
                        new_weight_data[p, q, :, :] = weight.data[p, q, :, :]

                new_weight_data = new_weight_data.reshape(-1, 8, num_input_align, kernel_h, kernel_w)
                new_weight_data = new_weight_data.transpose(0, 2, 3, 4, 1)
                new_weight.data = new_weight_data
                new_weight.shape = new_weight_data.shape
                new_weight.layout = Layout.NHWC8

                if weight_bits == 4:
                    new_weight.data = combine4bit_8bit(new_weight_data)
                return new_weight
            else:
                raise AssertionError(f"Conv2dInt does not support this type: {kernels}!")
        else:
            num_input_align = ALIGN8(kernel_c)
            num_output_align = ALIGN2(kernel_num)
            kernel_size = num_input_align * num_output_align * kernel_h * kernel_w

            if kernel_size * weight.dtype.itemsize > 8192:
                temp_size = num_input_align * kernel_h * kernel_w
                max_size = 8192 // temp_size
                assert max_size > 0, "kernel size of Conv2dInt must be less than 8KB"
                if max_size % 2 != 0:
                    max_size -= 1
                split_num = (kernel_num + max_size - 1) // max_size
                out_size = (kernel_num + split_num - 1) // split_num
                if out_size % 2 != 0:
                    out_size += 1
                num_output_align = out_size * split_num

            new_weight_data = np.zeros(
                (num_output_align, num_input_align, kernel_h, kernel_w),
                weight.dtype
            )
            for p in range(kernel_num):
                for q in range(kernel_c):
                    new_weight_data[p, q, :, :] = weight.data[p, q, :, :]

            new_weight_data = new_weight_data.transpose(0, 2, 3, 1)
            new_weight_data = new_weight_data.reshape(-1, 2, kernel_h, kernel_w, num_input_align)
            new_weight_data = new_weight_data.transpose(0, 2, 3, 1, 4)
            shape = new_weight_data.shape
            new_weight_data = new_weight_data.reshape(shape[0], shape[1] * shape[2], shape[3], -1, 8)
            new_weight_data = new_weight_data.transpose(0, 1, 3, 2, 4)

            assert weight.layout in {Layout.NCHW, Layout.NCWH}
            new_weight.data = new_weight_data
            new_weight.shape = new_weight.data.shape
            if weight_bits == 4:
                new_weight.data = combine4bit_8bit(new_weight_data)
            new_weight.bits = np.float32(weight_bits / 8.0)

            if data.layout == Layout.NCHW:
                new_weight.layout = Layout.NHWC8
            elif data.layout == Layout.NCWH:
                new_weight.layout = Layout.NWHC8
            return new_weight
    elif data.dtype == np.int16 and weight_bits == 16:
        new_weight.data = weight.data.transpose(1, 2, 3, 0)
        new_weight.shape = new_weight.data.shape
        return new_weight
    else:
        raise AssertionError(f"Conv2dInt does not support this type: {kernels}!")


__all__ = ["get_Conv2dInt_workspace", "Conv2dInt_weight_rearrange"]
