import math
import numpy as np

from ..utils import AutoPad, CeilMode
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
        AssertionError: If input or output memory type is not supported
    """
    workspace_size = 0
    group_align = ALIGN16(group)
    stride_h, stride_w = strides[0:2]
    kernel_h, kernel_w = kernels[0:2]
    pad_l, pad_r, pad_t, pad_b = pads[0:4]
    c_in, h_in, w_in = data.shape[1:4]
    c_ou, h_ou, w_ou = out.shape[1:4]

    data_size = ALIGN8(c_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * h_in
    data_size_withouth = ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * ALIGN8(c_in)
    out_size = out.nbytes

    if data.dtype == np.int8 and weight.dtype == np.int8:
        if dilations[0] > 1:
            split_h_in = h_in - ((kernel_h - 1) * dilations[0] + 1) // dilations[0] + kernel_h
            split_h_ou = (split_h_in - kernel_h) // stride_h + 1
            split_input_condition = data_size_withouth * split_h_in

            if group != 1:
                if group == c_ou:
                    weight_size = ALIGN16(c_ou) * kernel_h * kernel_w
                else:
                    raise AssertionError("Group convolution must split in op-divide module!")
            else:
                weight_size = ALIGN2(c_ou) * ALIGN8(c_in) * kernel_h * kernel_w
            assert split_input_condition <= 65536 and weight_size <= 32768, "Conv2dInt does not support this type!"
            workspace_size  = c_in * h_in * w_in + max(c_in * w_in * split_h_in, c_ou * w_ou * split_h_ou)
        elif dilations == (1, 1):
            if group == 1:
                weight_size = ALIGN8(c_in) * ALIGN2(c_ou) * kernel_h * kernel_w # commcon conv
                if (data_size <= 65536) and (weight_size <= 32768):
                    workspace_size = (c_in * h_in * w_in) * (data.mem_type != MemType.SHARE_MEM) + (c_ou * h_ou * w_ou * out.dtype.itemsize) *(out.mem_type != MemType.SHARE_MEM)
                elif (data_size <= 65536) and (weight_size > 32768):
                    c_ou_max = 32768 // (ALIGN8(c_in) * kernel_h * kernel_w)
                    split_num = (c_ou + c_ou_max - 1) // c_ou_max
                    split_c_ou = (c_ou + 2 * split_num - 1) // split_num
                    out_size = split_c_ou * h_ou * w_ou * out.dtype.itemsize
                    workspace_size = (c_in * h_in * w_in) * (data.mem_type != MemType.SHARE_MEM) + (split_c_ou * h_ou * w_ou * out.dtype.itemsize) * (out.mem_type != MemType.SHARE_MEM)
                elif (data_size > 65536) and (weight_size <= 32768):
                    split_num   = 1
                    split_h     = h_in
                    while (split_h * data_size_withouth > 65536) or (h_ou % split_num) != 0:
                        split_num   += 1
                        split_h     = (h_ou * stride_h) // split_num + kernel_h - stride_h
                        if split_num > h_in or split_num > h_ou:
                            break
                    if h_in * w_in >= 65536:
                        workspace_size = max(c_in * split_h * w_in, c_ou * h_ou * w_ou * out.dtype.itemsize)
                else:
                    pass
            elif group == c_ou:
                weight_size = ALIGN16(c_in) * kernel_h * kernel_w # depthwise conv
                if (data_size <= 65536) and (weight_size <= 32768):
                    workspace_size = c_in * h_in * w_in * (data.mem_type != MemType.SHARE_MEM) + c_ou * h_ou * w_ou * (out.mem_type != MemType.SHARE_MEM)
                elif (data_size > 65536) and (weight_size <= 32768):
                    split_num = 1
                    split_h_in = h_in
                    while (split_h_in * data_size_withouth > 65536) or (h_ou % split_num) != 0:
                        split_num = split_num + 1
                        split_h_in = (h_ou * stride_h) // split_num + kernel_h - stride_h
                        if split_num > h_in or split_num > h_ou:
                            break
                    workspace_size = c_in * split_h_in * w_in + c_ou * w_ou * h_ou
                else:
                    raise AssertionError("Do not support this type!")
            else:
                raise AssertionError("Do not support this type!")
    elif data.dtype == np.int16 and weight.dtype == np.int16:
        assert dilations == (1, 1) and group == 1  and \
        data.mem_type == MemType.SHARE_MEM and out.mem_type == MemType.SHARE_MEM
        data_size = ALIGN4(h_ou * w_ou) * ALIGN2(c_in * kernel_h * kernel_w) * 2
        weight_size = ALIGN2(c_in * kernel_h * kernel_w) * ALIGN4(c_ou) * 2
        assert data_size < 65536 and weight_size < 32768
        workspace_size = h_ou * w_ou * c_in * kernel_h * kernel_w * 2 + c_ou * h_ou * w_ou * 4
    else:
        raise AssertionError("Do not support this type!")

    return workspace_size


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
    kernel_h = kernels[0]
    kernel_w = kernels[1]
    stride_h = strides[0]
    stride_w = strides[1]

    new_weight = weight.clone()
    new_weight.data = weight.data

    if data.layout == Layout.NCWH and weight.layout == Layout.NCHW:
        new_weight.data = weight.data.transpose(0, 1, 3, 2)
        new_weight.shape = new_weight.data.shape
        new_weight.layout = Layout.NCWH

    if new_weight.layout not in {Layout.NCHW, Layout.NCWH}:
        return new_weight

    if kernel_h <= 5 and kernel_w <= 5:
        kernel_num = new_weight.shape[0]
        kernel_c = new_weight.shape[1]

        h_in = data.shape[2]
        w_in = data.shape[3]
        data_size = ALIGN8(kernel_c) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w) * h_in
        # depthwise/group conv
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
                        new_weight_data[p, q, :, :] = new_weight.data[p, q, :, :]

                new_weight_data = new_weight_data.reshape(-1, 16, kernel_h, kernel_w)
                new_weight.data = new_weight_data.transpose(0, 2, 3, 1)
                new_weight.shape = new_weight.data.shape
                new_weight.layout = Layout.NHWC16
            else:
                raise AssertionError("Group Conv2dInt should split in op_divide")
        # common conv
        else:
            num_input_align = ALIGN8(kernel_c)
            num_output_align = ALIGN2(kernel_num)
            kernel_size = num_input_align * num_output_align * kernel_h * kernel_w
            assert kernel_size * weight.dtype.itemsize <= 32768 or data_size <= 65536, "kernel size exceed 32KB and data size exceed 64KB in conv2d!"

            if (kernel_size * weight.dtype.itemsize > 32768) and \
                data_size * data.dtype.itemsize <= 65536:
                temp_size = num_input_align * kernel_h * kernel_w
                max_size = 32768 // temp_size
                if max_size % 2:
                    max_size -= 1
                split_num = (kernel_num + max_size - 1) // max_size

                out_size = (kernel_num + split_num - 1) // split_num
                if out_size % 2:
                    out_size += 1
                num_output_align = out_size * split_num

            new_weight_data = np.zeros((num_output_align, num_input_align, kernel_h, kernel_w), weight.dtype,)

            for p in range(kernel_num):
                for q in range(kernel_c):
                    new_weight_data[p, q, :, :] = new_weight.data[p, q, :, :]

            new_weight_data = new_weight_data.transpose(0, 2, 3, 1)
            new_weight_data = new_weight_data.reshape(-1, 2, kernel_h, kernel_w, num_input_align)
            new_weight_data = new_weight_data.transpose(0, 2, 3, 1, 4)
            shape = new_weight_data.shape
            new_weight_data = new_weight_data.reshape(shape[0], shape[1] * shape[2], shape[3], -1, 8)
            new_weight.data = new_weight_data.transpose(0, 1, 3, 2, 4)
            new_weight.shape = new_weight.data.shape

            if data.layout == Layout.NCHW:
                new_weight.layout = Layout.NHWC8
            else:
                new_weight.layout = Layout.NWHC8
    else:
        raise AssertionError(f"Conv2dInt does not support this type: {kernels}!")

    return new_weight

__all__ = ["get_Conv2dInt_workspace", "Conv2dInt_weight_rearrange"]