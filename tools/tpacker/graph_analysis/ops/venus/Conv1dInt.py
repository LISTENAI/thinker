import math
import numpy as np

from ..utils import AutoPad, CeilMode, combine4bit_8bit
from ....enum_defines import Layout, MemType, ALIGN2, ALIGN8, ALIGN16


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
        int: Workspace size
        
    Raises:
        AssertionError: If input memory type is not SHARE_MEM
    """
    assert data.mem_type == MemType.SHARE_MEM, "input of conv1d must be in share-memory"

    workspace_size = out.nbytes if out.mem_type != MemType.SHARE_MEM else 0
    stride_w = strides[0]
    kernel_w = kernels[0]

    c_in = data.shape[1]
    w_in = data.shape[2]
    data_size = ALIGN8(c_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
    c_ou = out.shape[1]

    if kernel_w < 6:
        weight_size = ALIGN2(c_ou) * ALIGN8(c_in) * kernel_w
        assert data_size <= 65536, "Input size of Conv1dInt exceeds the Venus runtime limit"
        assert weight_size <= 32768 or data_size <= 65536
    else:
        raise AssertionError("Conv1dInt on venus only supports kernels up to 5")

    return workspace_size


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

    if kernel_w > 5:
        new_weight = weight.clone()
        new_weight.data = weight.data.transpose(2, 1, 0)
        new_weight.shape = new_weight.data.shape
        new_weight.layout = Layout.WHCN
        return new_weight
    else:
        h_in = 1
        w_in = data.shape[2]
        data_size = ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * ALIGN8(kernel_c) * h_in
        assert data_size * data.dtype.itemsize <= 65536, "input size of Conv1dInt exceed limit"

        weight.data = weight.data.reshape(kernel_num, kernel_c, kernel_h, kernel_w)

        if group != 1:
            if group == kernel_num:
                assert kernel_c == 1
                num_output_align = ALIGN16(kernel_num)
                kernel_size = num_output_align * kernel_h * kernel_w
                group = num_output_align
                assert kernel_size * weight.dtype.itemsize <= 32768

                new_weight_data = np.zeros((num_output_align, 1, kernel_h, kernel_w), weight.dtype)
                for p in range(kernel_num):
                    for q in range(kernel_c):
                        new_weight_data[p, q, :, :] = weight.data[p, q, :, :]

                new_weight_data = new_weight_data.reshape(-1, 16, kernel_h, kernel_w)
                new_weight_data = new_weight_data.transpose(0, 2, 3, 1)

                new_weight = weight.clone()
                if weight_bits == 4:
                    new_weight.data = combine4bit_8bit(new_weight_data)
                else:
                    new_weight.data = new_weight_data
                new_weight.shape = new_weight_data.shape
                new_weight.layout = Layout.NHWC16
                return new_weight
            else:
                raise AssertionError("group Conv1dInt should split in op_divide")
        else:
            num_output_align = ALIGN2(kernel_num)
            num_input_align = ALIGN8(kernel_c)
            kernel_size = num_input_align * num_output_align * kernel_h * kernel_w

            if (kernel_size * weight.dtype.itemsize > 32768) and (data_size * data.dtype.itemsize <= 65536):
                temp_size = num_input_align * kernel_h * kernel_w
                max_size = 32768 // temp_size
                if max_size % 2 != 0:
                    max_size -= 1
                split_num = (kernel_num + max_size - 1) // max_size
                out_size = (kernel_num + split_num - 1) // split_num
                if out_size % 2 != 0:
                    out_size += 1
                num_output_align = out_size * split_num

            new_weight_data = np.zeros((num_output_align, num_input_align, kernel_h, kernel_w), weight.dtype)
            for p in range(kernel_num):
                for q in range(kernel_c):
                    new_weight_data[p, q, :, :] = weight.data[p, q, :, :]

            new_weight_data = new_weight_data.transpose(0, 2, 3, 1)
            new_weight_data = new_weight_data.reshape(-1, 2, kernel_h, kernel_w, num_input_align)
            new_weight_data = new_weight_data.transpose(0, 2, 3, 1, 4)
            shape = new_weight_data.shape
            new_weight_data = new_weight_data.reshape(shape[0], shape[1] * shape[2], shape[3], -1, 8)

            new_weight = weight.clone()
            new_weight_data = new_weight_data.transpose(0, 1, 3, 2, 4)
            if weight_bits == 4:
                new_weight.data = combine4bit_8bit(new_weight_data)
            else:
                new_weight.data = new_weight_data
            new_weight.shape = new_weight_data.shape
            assert data.layout in {Layout.NCHW, Layout.NCWH}
            if data.layout == Layout.NCHW:
                new_weight.layout = Layout.NHWC8
            else:
                new_weight.layout = Layout.NWHC8

            return new_weight


__all__ = ["get_Conv1dInt_workspace", "Conv1dInt_weight_rearrange"]
