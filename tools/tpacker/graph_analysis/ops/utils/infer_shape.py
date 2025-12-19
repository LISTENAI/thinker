import math
from typing import Tuple, Dict

from .utils import AutoPad, CeilMode
from ....enum_defines import Layout

def handle_conv_auto_pad(in_dim: int, stride: int, kernel: int, dilation: int, pad_type: AutoPad, pad_head: int, pad_tail: int) -> Tuple[int, int]:
    """Handle auto-padding for convolution operations.
    
    Args:
        in_dim (int): Input dimension size.
        stride (int): Stride size.
        kernel (int): Kernel size.
        dilation (int): Dilation size.
        pad_type (AutoPad): Type of auto-padding.
        pad_head (int): Padding at the beginning.
        pad_tail (int): Padding at the end.
        
    Returns:
        Tuple[int, int]: Adjusted padding values (pad_head, pad_tail).
    """
    if pad_type == AutoPad.NOTSET:
        return (pad_head, pad_tail)
    elif pad_type == AutoPad.VALID:
        return (0, 0)
    elif pad_type in {AutoPad.SAME_LOWER, AutoPad.SAME_UPPER}:
        target_size = (in_dim + stride - 1) // stride
        pad_needed = (target_size - 1) * stride + kernel - in_dim
        if pad_type == AutoPad.SAME_LOWER:
            pad_head = (pad_needed + 1) // 2
        else:
            pad_head = pad_needed // 2
        pad_tail = pad_needed - pad_head
        return (pad_head, pad_tail)
    else:
        raise ValueError(f"Pad type {pad_type} not supported.")

def compute_conv_output_size(in_dim: int, stride: int, kernel: int, dilation: int, pad_head: int, pad_tail: int) -> int:
    """Compute the output size of a convolution operation.
    
    Args:
        in_dim (int): Input dimension size.
        stride (int): Stride size.
        kernel (int): Kernel size.
        dilation (int): Dilation size.
        pad_head (int): Padding at the beginning.
        pad_tail (int): Padding at the end.
        
    Returns:
        int: Output dimension size.
    """
    dkernel = dilation * (kernel - 1) + 1
    output_size = (in_dim + pad_head + pad_tail - dkernel) // stride + 1
    return output_size

def calc_conv_output_dim(input_size: int, kernel: int, stride: int, dilation: int, pad_head: int, pad_tail: int, auto_pad: AutoPad) -> int:
    """Calculate the output dimension of a convolution operation.
    
    Args:
        input_size (int): Input dimension size.
        kernel (int): Kernel size.
        stride (int): Stride size.
        dilation (int): Dilation size.
        pad_head (int): Padding at the beginning.
        pad_tail (int): Padding at the end.
        auto_pad (AutoPad): Type of auto-padding.
        
    Returns:
        int: Output dimension size.
    """
    pad_head, pad_tail = handle_conv_auto_pad(input_size, stride, kernel, dilation, auto_pad, pad_head, pad_tail)
    output_size = compute_conv_output_size(input_size, stride, kernel, dilation, pad_head, pad_tail)
    return output_size

def calc_conv1d_output_shape(input_shape: Tuple[int, int, int], weight_shape: Tuple[int, int, int], kernels: Tuple[int], strides: Tuple[int] = (1,),
                            dilations: Tuple[int] = (1,), pads: Tuple[int, int] = (0, 0), groups: int = 1, auto_pad: AutoPad = AutoPad.NOTSET, layout: Layout = Layout.NCHW) -> Tuple[int, int, int]:
    """Calculate the output shape of a 1D convolution operation.
    
    Args:
        input_shape (Tuple[int, int, int]): Input tensor shape (batch, channels, height).
        weight_shape (Tuple[int, int, int]): Weight tensor shape (output_channels, input_channels, kernel_size).
        kernels (Tuple[int]): Kernel size.
        strides (Tuple[int], optional): Stride size. Defaults to (1,).
        dilations (Tuple[int], optional): Dilation size. Defaults to (1,).
        pads (Tuple[int, int], optional): Padding values. Defaults to (0, 0).
        groups (int, optional): Number of groups. Defaults to 1.
        auto_pad (AutoPad, optional): Type of auto-padding. Defaults to AutoPad.NOTSET.
        layout (Layout, optional): Data layout. Defaults to Layout.NCHW.
        
    Returns:
        Tuple[int, int, int]: Output tensor shape.
    """
    b, c, h = input_shape
    out_c, in_c, k_h = weight_shape
    if layout == Layout.NHWC:
        b, h, c = input_shape
        out_c, k_h, in_c = weight_shape
    assert groups * in_c == c, "Check fail: groups * channel(weight) != channel(input)"
    assert k_h == kernels[0], "Check fail: kernel_h != kernels[0]"
    out_h = calc_conv_output_dim(h, kernels[0], strides[0], dilations[0], pads[0], pads[1], auto_pad)
    output_shape = (b, out_c, out_h)
    if layout == Layout.NHWC:
        output_shape = (b, out_h, out_c)
    return output_shape

def calc_conv2d_output_shape(input_shape: Tuple[int, int, int, int], weight_shape: Tuple[int, int, int, int], kernels: Tuple[int, int], strides: Tuple[int, int] = (1, 1),
                            dilations: Tuple[int, int] = (1, 1), pads: Tuple[int, int, int, int] = (0, 0, 0, 0), groups: int = 1, auto_pad: AutoPad = AutoPad.NOTSET) -> Tuple[int, int, int, int]:
    """Calculate the output shape of a 2D convolution operation.
    
    Args:
        input_shape (Tuple[int, int, int, int]): Input tensor shape (batch, channels, height, width).
        weight_shape (Tuple[int, int, int, int]): Weight tensor shape (output_channels, input_channels, kernel_height, kernel_width).
        kernels (Tuple[int, int]): Kernel sizes.
        strides (Tuple[int, int], optional): Stride sizes. Defaults to (1, 1).
        dilations (Tuple[int, int], optional): Dilation sizes. Defaults to (1, 1).
        pads (Tuple[int, int, int, int], optional): Padding values. Defaults to (0, 0, 0, 0).
        groups (int, optional): Number of groups. Defaults to 1.
        auto_pad (AutoPad, optional): Type of auto-padding. Defaults to AutoPad.NOTSET.
        
    Returns:
        Tuple[int, int, int, int]: Output tensor shape.
    """
    b, c, h, w = input_shape
    out_c, in_c, k_h, k_w = weight_shape
    assert groups * in_c == c, "Check fail: groups * channel(weight) != channel(input)"
    assert (k_h == kernels[0] and k_w == kernels[1]), "Check fail: kernel_h != kernels[0] && kernel_w != kernels[1]"
    out_h = calc_conv_output_dim(h, kernels[0], strides[0], dilations[0], pads[0], pads[2], auto_pad)
    out_w = calc_conv_output_dim(w, kernels[1], strides[1], dilations[1], pads[1], pads[3], auto_pad)
    return (b, out_c, out_h, out_w)

def compute_deconv_total_pad(input_size: int, stride: int, output_pad: int, kernel: int, dilation: int, output_size: int) -> int:
    """Compute the total padding required for a deconvolution operation.
    
    Args:
        input_size (int): Input dimension size.
        stride (int): Stride size.
        output_pad (int): Output padding size.
        kernel (int): Kernel size.
        dilation (int): Dilation size.
        output_size (int): Output dimension size.
        
    Returns:
        int: Total padding required.
    """
    return max(0, (input_size - 1) * stride + output_pad + (kernel - 1) * dilation + 1 - output_size)

def handle_deconv_auto_pad(auto_pad: AutoPad, total_pad: int) -> Tuple[int, int]:
    """Handle auto-padding for deconvolution operations.
    
    Args:
        auto_pad (AutoPad): Type of auto-padding.
        total_pad (int): Total padding required.
        
    Returns:
        Tuple[int, int]: Adjusted padding values (pad_head, pad_tail).
    """
    if auto_pad == AutoPad.SAME_UPPER:
        pad_head = total_pad - total_pad // 2
        pad_tail = total_pad // 2
    else:
        pad_head = total_pad // 2
        pad_tail = total_pad - pad_head
    return (pad_head, pad_tail)

def calc_deconv_output_dim(input_size: int, kernel: int, stride: int, dilation: int, pad_head: int, pad_tail: int, auto_pad: AutoPad, output_pad: int, output_size: int = None) -> int:
    """Calculate the output dimension of a deconvolution operation.
    
    Args:
        input_size (int): Input dimension size.
        kernel (int): Kernel size.
        stride (int): Stride size.
        dilation (int): Dilation size.
        pad_head (int): Padding at the beginning.
        pad_tail (int): Padding at the end.
        auto_pad (AutoPad): Type of auto-padding.
        output_pad (int): Output padding size.
        output_size (int, optional): Output dimension size. Defaults to None.
        
    Returns:
        int: Output dimension size.
    """
    if output_size != 0:
        assert output_size >= 0
        total_pad = compute_deconv_total_pad(input_size, stride, output_pad, kernel, dilation, output_size)
        pad_head, pad_tail = handle_deconv_auto_pad(auto_pad, total_pad)
        return output_size
    if auto_pad in {AutoPad.SAME_LOWER, AutoPad.SAME_UPPER}:
        total_pad = compute_deconv_total_pad(input_size, stride, output_pad, kernel, dilation, input_size * stride)
        pad_head, pad_tail = handle_deconv_auto_pad(auto_pad, total_pad)
    output_size = ((input_size - 1) * stride + output_pad + (kernel - 1) * dilation + 1 - pad_head - pad_tail)
    return output_size

def calc_deconv2d_output_shape(input_shape: Tuple[int, int, int, int], weight_shape: Tuple[int, int, int, int], kernels: Tuple[int, int] = None, strides: Tuple[int, int] = (1, 1),
                              dilations: Tuple[int, int] = (1, 1), pads: Tuple[int, int, int, int] = (0, 0, 0, 0), groups: int = 1, auto_pad: AutoPad = AutoPad.NOTSET,
                              output_pads: Tuple[int, int] = (0, 0), output_shape: Tuple[int, int] = (0, 0), layout: Layout = Layout.NCHW) -> Tuple[int, int, int, int]:
    """Calculate the output shape of a 2D deconvolution operation.
    
    Args:
        input_shape (Tuple[int, int, int, int]): Input tensor shape (batch, channels, height, width).
        weight_shape (Tuple[int, int, int, int]): Weight tensor shape (input_channels, output_channels, kernel_height, kernel_width).
        kernels (Tuple[int, int], optional): Kernel sizes. Defaults to None.
        strides (Tuple[int, int], optional): Stride sizes. Defaults to (1, 1).
        dilations (Tuple[int, int], optional): Dilation sizes. Defaults to (1, 1).
        pads (Tuple[int, int, int, int], optional): Padding values. Defaults to (0, 0, 0, 0).
        groups (int, optional): Number of groups. Defaults to 1.
        auto_pad (AutoPad, optional): Type of auto-padding. Defaults to AutoPad.NOTSET.
        output_pads (Tuple[int, int], optional): Output padding values. Defaults to (0, 0).
        output_shape (Tuple[int, int], optional): Output shape. Defaults to (0, 0).
        layout (Layout, optional): Data layout. Defaults to Layout.NCHW.
        
    Returns:
        Tuple[int, int, int, int]: Output tensor shape.
    """
    b, c, h, w = input_shape
    in_c, out_c, k_h, k_w = weight_shape
    if layout == Layout.NHWC:
        b, h, w, c = input_shape
        in_c, k_h, k_w, out_c = weight_shape
    if kernels is None:
        kernels = (k_h, k_w)
    assert in_c == c, "Check fail: groups * channel(weight) != channel(input)"
    assert (k_h == kernels[0] and k_w == kernels[1]), "Check fail: kernel_h != kernels[0] && kernel_w != kernels[1]"
    out_h = calc_deconv_output_dim(h, kernels[0], strides[0], dilations[0], pads[0], pads[2], auto_pad, output_pads[0], output_shape[0])
    out_w = calc_deconv_output_dim(w, kernels[1], strides[1], dilations[1], pads[1], pads[3], auto_pad, output_pads[1], output_shape[1])
    final_output_shape = (b, groups * out_c, out_h, out_w)
    if layout == Layout.NHWC:
        final_output_shape = (b, groups * out_h, out_w, out_c)
    return final_output_shape

def compute_pool_output_size(in_dim: int, stride: int, kernel: int, dilation: int, pad_head: int, pad_tail: int, ceil_mode: CeilMode) -> int:
    """Compute the output size of a pooling operation.
    
    Args:
        in_dim (int): Input dimension size.
        stride (int): Stride size.
        kernel (int): Kernel size.
        dilation (int): Dilation size.
        pad_head (int): Padding at the beginning.
        pad_tail (int): Padding at the end.
        ceil_mode (CeilMode): Ceiling mode for output size calculation.
        
    Returns:
        int: Output dimension size.
    """
    dkernel = dilation * (kernel - 1) + 1
    output_size = (in_dim + pad_head + pad_tail - dkernel) // stride + 1
    return output_size

def calc_pool_output_dim(input_size: int, kernel: int, stride: int, dilation: int, pad_head: int, pad_tail: int, auto_pad: AutoPad, ceil_mode: CeilMode) -> int:
    """Calculate the output dimension of a pooling operation.
    
    Args:
        input_size (int): Input dimension size.
        kernel (int): Kernel size.
        stride (int): Stride size.
        dilation (int): Dilation size.
        pad_head (int): Padding at the beginning.
        pad_tail (int): Padding at the end.
        auto_pad (AutoPad): Type of auto-padding.
        ceil_mode (CeilMode): Ceiling mode for output size calculation.
        
    Returns:
        int: Output dimension size.
    """
    pad_head, pad_tail = handle_conv_auto_pad(input_size, stride, kernel, dilation, auto_pad, pad_head, pad_tail)
    output_size = compute_pool_output_size(input_size, stride, kernel, dilation, pad_head, pad_tail, ceil_mode)
    return output_size

def calc_pool2d_output_shape(input_shape: Tuple[int, int, int, int], kernels: Tuple[int, int], strides: Tuple[int, int] = (1, 1),
                            dilations: Tuple[int, int] = (1, 1), pads: Tuple[int, int, int, int] = (0, 0, 0, 0), ceil_mode: CeilMode = CeilMode.NO,
                            layout: Layout = Layout.NCHW, auto_pad: AutoPad = AutoPad.NOTSET) -> Tuple[int, int, int, int]:
    """Calculate the output shape of a 2D pooling operation.
    
    Args:
        input_shape (Tuple[int, int, int, int]): Input tensor shape (batch, channels, height, width).
        kernels (Tuple[int, int]): Kernel sizes.
        strides (Tuple[int, int], optional): Stride sizes. Defaults to (1, 1).
        dilations (Tuple[int, int], optional): Dilation sizes. Defaults to (1, 1).
        pads (Tuple[int, int, int, int], optional): Padding values. Defaults to (0, 0, 0, 0).
        ceil_mode (CeilMode, optional): Ceiling mode for output size calculation. Defaults to CeilMode.NO.
        layout (Layout, optional): Data layout. Defaults to Layout.NCHW.
        auto_pad (AutoPad, optional): Type of auto-padding. Defaults to AutoPad.NOTSET.
        
    Returns:
        Tuple[int, int, int, int]: Output tensor shape.
    """
    b, c, h, w = input_shape
    if layout == Layout.NHWC:
        b, h, w, c = input_shape
    out_h = calc_pool_output_dim(h, kernels[0], strides[0], dilations[0], pads[0], pads[2], auto_pad, ceil_mode)
    out_w = calc_pool_output_dim(w, kernels[1], strides[1], dilations[1], pads[1], pads[3], auto_pad, ceil_mode)
    output_shape = (b, c, out_h, out_w)
    if layout == Layout.NHWC:
        output_shape = (b, out_h, out_w, c)
    return output_shape

__all__ = ["calc_conv1d_output_shape", "calc_conv2d_output_shape", "calc_deconv2d_output_shape", "calc_pool2d_output_shape"]