import math

from .utils import AutoPad, CeilMode
from ....enum_defines import Layout


def handle_conv_auto_pad(
    in_dim, stride, kernel, dilation, pad_type, pad_head, pad_tail
):
    if pad_type == AutoPad.NOTSET:
        return pad_head, pad_tail
    elif pad_type == AutoPad.VALID:
        return 0, 0
    elif pad_type in {AutoPad.SAME_LOWER, AutoPad.SAME_UPPER}:
        target_size = (in_dim + stride - 1) // stride
        pad_needed = (target_size - 1) * stride + kernel - in_dim
        if pad_type == AutoPad.SAME_LOWER:
            pad_head = (pad_needed + 1) // 2
        else:
            pad_head = pad_needed // 2
        pad_tail = pad_needed - pad_head
        return pad_head, pad_tail
    else:
        raise (f"Pad type {pad_type} not supported.")


def compute_conv_output_size(in_dim, stride, kernel, dilation, pad_head, pad_tail):
    dkernel = dilation * (kernel - 1) + 1
    output_size = (in_dim + pad_head + pad_tail - dkernel) // stride + 1
    return output_size


def calc_conv_output_dim(
    input_size, kernel, stride, dilation, pad_head, pad_tail, auto_pad
):
    pad_head, pad_tail = handle_conv_auto_pad(
        input_size, stride, kernel, dilation, auto_pad, pad_head, pad_tail
    )
    output_size = compute_conv_output_size(
        input_size, stride, kernel, dilation, pad_head, pad_tail
    )
    return output_size


def calc_conv1d_output_shape(
    input_shape,
    weight_shape,
    kernels=None,
    strides=(1),
    dilations=(1),
    pads=(0, 0),
    groups=1,
    auto_pad=AutoPad.NOTSET,
    layout=Layout.NCHW,
):
    b, c, h = input_shape
    out_c, in_c, k_h = weight_shape
    if layout == Layout.NHWC:
        b, h, c = input_shape
        out_c, k_h, in_c = weight_shape
    assert groups * in_c == c, "Check fail: groups * channel(weight) != channel(input)"
    assert k_h == kernels[0], "Check fail: kernel_h != kernels[0]"
    out_h = calc_conv_output_dim(
        h, kernels[0], strides[0], dilations[0], pads[0], pads[1], auto_pad
    )
    output_shape = (b, out_c, out_h)
    if layout == Layout.NHWC:
        output_shape = (b, out_h, out_c)
    return output_shape


def calc_conv2d_output_shape(
    input_shape,
    weight_shape,
    kernels=None,
    strides=(1, 1),
    dilations=(1, 1),
    pads=(0, 0, 0, 0),
    groups=1,
    auto_pad=AutoPad.NOTSET,
):
    b, c, h, w = input_shape
    out_c, in_c, k_h, k_w = weight_shape
    assert groups * in_c == c, "Check fail: groups * channel(weight) != channel(input)"
    assert (
        k_h == kernels[0] and k_w == kernels[1]
    ), "Check fail: kernel_h != kernels[0] && kernel_w != kernels[1]"
    out_h = calc_conv_output_dim(
        h, kernels[0], strides[0], dilations[0], pads[0], pads[2], auto_pad
    )
    out_w = calc_conv_output_dim(
        w, kernels[1], strides[1], dilations[1], pads[1], pads[3], auto_pad
    )
    output_shape = (b, out_c, out_h, out_w)
    return output_shape


def compute_deconv_total_pad(
    input_size, stride, output_pad, kernel, dilation, output_size
):
    return max(
        0,
        (input_size - 1) * stride
        + output_pad
        + (kernel - 1) * dilation
        + 1
        - output_size,
    )


def handle_deconv_auto_pad(auto_pad, total_pad):
    if auto_pad == AutoPad.SAME_UPPER:
        pad_head = total_pad - total_pad / 2
        pad_tail = total_pad / 2
    else:  # for auto_pad is NOTSET, SAME_LOWER or VALID
        pad_head = total_pad / 2
        pad_tail = total_pad - total_pad / 2
    return pad_head, pad_tail


def calc_deconv_output_dim(
    input_size,
    kernel,
    stride,
    dilation,
    pad_head,
    pad_tail,
    auto_pad,
    output_pad,
    output_size=None,
):
    if output_size != 0:
        assert output_size >= 0
        total_pad = compute_deconv_total_pad(
            input_size, stride, output_pad, kernel, dilation, output_size
        )
        pad_head, pad_tail = handle_deconv_auto_pad(auto_pad, total_pad)
        return output_size
    if auto_pad in {AutoPad.SAME_LOWER, AutoPad.SAME_UPPER}:
        total_pad = compute_deconv_total_pad(
            input_size, stride, output_pad, kernel, dilation, input_size * stride
        )
        pad_head, pad_tail = handle_deconv_auto_pad(auto_pad, total_pad)
    output_size = (
        (input_size - 1) * stride
        + output_pad
        + (kernel - 1) * dilation
        + 1
        - pad_head
        - pad_tail
    )
    return output_size


def calc_deconv2d_output_shape(
    input_shape,
    weight_shape,
    kernels=None,
    strides=(1, 1),
    dilations=(1, 1),
    pads=(0, 0, 0, 0),
    groups=1,
    auto_pad=AutoPad.NOTSET,
    output_pads=(0, 0),
    output_shape=(0, 0),
    layout=Layout.NCHW,
):
    b, c, h, w = input_shape
    in_c, out_c, k_h, k_w = weight_shape
    if layout == Layout.NHWC:
        b, h, w, c = input_shape
        in_c, k_h, k_w, out_c = weight_shape
    if kernels is None:
        kernels = (k_h, k_w)
    assert in_c == c, "Check fail: groups * channel(weight) != channel(input)"
    assert (
        k_h == kernels[0] and k_w == kernels[1]
    ), "Check fail: kernel_h != kernels[0] && kernel_w != kernels[1]"
    out_h = calc_deconv_output_dim(
        h,
        kernels[0],
        strides[0],
        dilations[0],
        pads[0],
        pads[2],
        auto_pad,
        output_pads[0],
        output_shape[0],
    )
    out_w = calc_deconv_output_dim(
        w,
        kernels[1],
        strides[1],
        dilations[1],
        pads[1],
        pads[3],
        auto_pad,
        output_pads[1],
        output_shape[1],
    )
    final_output_shape = (b, groups * out_c, out_h, out_w)
    if layout == Layout.NHWC:
        final_output_shape = (b, groups * out_h, out_w, out_c)
    return final_output_shape


def compute_pool_output_size(
    in_dim, stride, kernel, dilation, pad_head, pad_tail, ceil_mode
):
    dkernel = dilation * (kernel - 1) + 1
    output_size = (in_dim + pad_head + pad_tail - dkernel) / stride + 1
    if ceil_mode == CeilMode.CEIL:
        return math.ceil(output_size)
    return math.floor(output_size)


def calc_pool_output_dim(
    input_size, kernel, stride, dilation, pad_head, pad_tail, auto_pad, ceil_mode
):
    pad_head, pad_tail = handle_conv_auto_pad(
        input_size, stride, kernel, dilation, auto_pad, pad_head, pad_tail
    )
    output_size = compute_pool_output_size(
        input_size, stride, kernel, dilation, pad_head, pad_tail, ceil_mode
    )
    return output_size


def calc_pool2d_output_shape(
    input_shape,
    kernels,
    strides=(1, 1),
    dilations=(1, 1),
    pads=(0, 0, 0, 0),
    auto_pad=AutoPad.NOTSET,
    ceil_mode=CeilMode.NO,
    layout=Layout.NCHW,
):
    b, c, h, w = input_shape
    if layout == Layout.NHWC:
        b, h, w, c = input_shape
    out_h = calc_pool_output_dim(
        h, kernels[0], strides[0], dilations[0], pads[0], pads[2], auto_pad, ceil_mode
    )
    out_w = calc_pool_output_dim(
        w, kernels[1], strides[1], dilations[1], pads[1], pads[3], auto_pad, ceil_mode
    )
    output_shape = (b, c, out_h, out_w)
    if layout == Layout.NHWC:
        output_shape = (b, out_h, out_w, c)
    return output_shape


__all__ = [
    "calc_conv1d_output_shape",
    "calc_conv2d_output_shape",
    "calc_deconv2d_output_shape",
    "calc_pool2d_output_shape",
]
