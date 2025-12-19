import math
import os
import logging
import numpy as np

"""LUNA basic parameter definitions"""

SHARE_SIZE = 384 * 1024
SHARE_IN_BW = 64
SHARE_OUT_BW = 128
PSRAM_IN_BW = 280
PSRAM_OUT_BW = 300
FLASH_IN_BW = 60
FLASH_OUT_BW = 0
MATRIX_LEFT_SIZE_LIMIT = 8 * 1024
MATRIX_RIGHT_SIZE_LIMIT = 16 * 1024
CONV2D_WIEGHT_SIZE_LIMIT = 8 * 1024
CONV2D_INPUT_SIZE_LIMIT = 16 * 1024
CONV2D_INPUT_C_ALIGNMENT = 8
CONV2D_OUTPUT_C_ALIGNMENT = 2
DEVICE_SHARE = 0
DEVICE_PSRAM = 1
DEVICE_FLASH = 2

"""LUNA internal performance model"""

import functools
import inspect
import json
from datetime import datetime


def record_est_params(include_return=True, log_file=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            call_time = datetime.now().isoformat()
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            arguments = {}
            for name, value in bound_args.arguments.items():
                try:
                    json.dumps(value)
                    arguments[name] = value
                except (TypeError, ValueError):
                    arguments[name] = str(value)
            try:
                result = func(*args, **kwargs)
                log_info = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "call_time": call_time,
                    "arguments": arguments,
                    "result": result,
                }
                if log_file:
                    logging.debug("record_est_params, " + str(log_info))
                    # for luna perf test.
                    if func.__name__ in [
                        "est_nn_conv2d_cycles",
                        "est_nn_depthwise2d_cycles",
                        "est_nn_deconv2d_cycles",
                    ]:
                        pass
                        s = f"static uint32_t input_w_array[] = {{ { arguments['input_w'] } }} ;\n"
                        s += f"static uint32_t input_h_array[] = {{ { arguments['input_h'] } }} ;\n"
                        s += f"static uint32_t input_c_array[] = {{ { arguments['input_c'] } }} ;\n"
                        s += f"static uint32_t weight_w_array[] = {{ { arguments['kernel_w'] } }} ;\n"
                        s += f"static uint32_t weight_h_array[] = {{ { arguments['kernel_h'] } }} ;\n"
                        s += f"static uint32_t stride_w_array[] = {{ { arguments['stride_w'] } }} ;\n"
                        s += f"static uint32_t stride_h_array[] = {{ { arguments['stride_h'] } }} ;\n"
                        s += f"static uint32_t output_c_array[] = {{ { arguments['output_c'] } }} ;\n"
                        logging.debug("record_luna_perf, \n" + str(s))
                return result
            except Exception as e:
                print(f"exception: {e}")
                raise

        return wrapper

    return decorator


def _l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res


def _l_sum(in_list):
    res = 0
    for _ in in_list:
        res += _
    return res


def _l_align_n(size: int, alignment: int = 8) -> int:
    return (size + (alignment - 1)) // alignment * alignment


def _l_polyfit(x: np.ndarray, y: np.ndarray, degree: int = 4):
    assert x.shape == y.shape, "x and y must have the same size"
    assert len(x.shape) == 1, "x and y must be 1-D array"
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    return polynomial


def _cnn_performance_v5(
    iw=5, ih=128, ic=16, kw=3, kh=3, oc=32, oh=126, lp=0, rp=0, up=0, bp=0, sw=1, sh=1
):

    # cnn parameter
    # iw = 5
    # ih = 20
    # ic = 64

    # kw = 3
    # kh = 3
    # oc = 8

    # oh = 10

    # lp = 1
    # rp = 1
    # up = 1
    # bp = 1

    # sw = 2
    # sh = 2

    # cnn - newly added preprocessing calculation
    # if(rp>0)
    #     rp = sw*floor((iw+lp+rp-(kw-1)+(sw-1))/sw) - iw - lp ;
    # end
    if rp > 0:
        rp = sw * math.floor((iw + lp + rp - (kw - 1) + (sw - 1)) / sw) - iw - lp

    # if(bp>0)
    #     bp = sh*floor((ih+up+bp-(kh-1)+(sh-1))/sh) - ih - up ;
    # end
    if bp > 0:
        bp = sh * math.floor((ih + up + bp - (kh - 1) + (sh - 1)) / sh) - ih - up

    # if(rp<0)
    #     rp=0;
    # end
    if rp < 0:
        rp = 0

    # if(bp<0)
    #     bp=0;
    # end
    if bp < 0:
        bp = 0

    up_h = math.ceil(up / sh)
    bp_h = math.ceil(bp / sh)

    # h0 = kh*(ih-(kh-1));
    h0 = kh * (oh - up_h - bp_h)
    h1 = 0
    h2 = 0

    uh = 0
    bh = 0

    # if(up~=0)
    #     for n=1:1:up
    #         h1=h1+(kh-n);
    #     end
    # end
    if up != 0:
        # for n=up:(0-sh):1
        # MATLAB's up:(0-sh):1 means starting from up, decrementing to 1 with step size -sh
        n = up
        while n >= 1:
            h1 = h1 + (kh - n)
            uh = uh + n
            n = n - sh

    # if(bp~=0)
    #     for n=1:1:bp
    #         h2=h2+(kh-n);
    #     end
    # end
    if bp != 0:
        n = bp
        while n >= 1:
            h2 = h2 + (kh - n)
            bh = bh + n
            n = n - sh

    h = h0 + h1 + h2

    if lp % sw == 0:
        oft_sw = 0
    else:
        oft_sw = sw - (lp % sw)

    kw = kw + oft_sw

    left_hold = 0
    left_bypass = 0
    left_cal = 0

    # if(lp>0)
    #     if(kw > 4*sw*ceil(lp/(4*sw)))
    #         bbb=0;
    #     else
    #         bbb=1;
    #     end
    #     left_hold = ceil(lp/(4*sw))-bbb;
    #     for n=1:1:left_hold
    #         left_bypass = left_bypass + n;
    #         left_cal    = left_cal + (ceil(kw/(4*sw))-n);
    #     end
    # end
    if lp > 0:
        if kw > 4 * sw * math.ceil(lp / (4 * sw)):
            bbb = 0
        else:
            bbb = 1
        left_hold = math.ceil(lp / (4 * sw)) - bbb
        for n in range(1, left_hold + 1):
            left_bypass = left_bypass + n
            left_cal = left_cal + (math.ceil(kw / (4 * sw)) - n)

    right_hold = 0
    right_bypass = 0
    right_cal = 0

    # if(mod(iw,4*sw)==0)
    #     oft_iw =0;
    # else
    #     oft_iw = 4*sw-mod(iw,4*sw);
    # end
    if iw % (4 * sw) == 0:
        oft_iw = 0
    else:
        oft_iw = 4 * sw - (iw % (4 * sw))

    # if(mod(kw,4*sw)==0)
    #     oft_kw =0;
    # else
    #     oft_kw = 4*sw-mod(kw,4*sw);
    # end
    if kw % (4 * sw) == 0:
        oft_kw = 0
    else:
        oft_kw = 4 * sw - (kw % (4 * sw))

    rp = rp - oft_iw + oft_kw

    # if(rp>0)
    #     right_hold = ceil(rp/(4*sw));
    #     for n=1:1:ceil(rp/(4*sw))
    #         right_bypass = right_bypass + n;
    #         right_cal    = right_cal + (ceil(kw/(4*sw))-n);
    #     end
    # end
    if rp > 0:
        right_hold = math.ceil(rp / (4 * sw))
        for n in range(1, right_hold + 1):
            right_bypass = right_bypass + n
            right_cal = right_cal + (math.ceil(kw / (4 * sw)) - n)

    # middle_hold = ceil(iw/(4*sw))-floor((kw-1)/(4*sw));
    middle_hold = math.ceil(iw / (4 * sw)) - math.floor((kw - 1) / (4 * sw))

    middle_cal = kw * middle_hold
    # w_cal = middle_cal + 4*sw*(left_cal + right_cal) - oft_kw*left_hold;
    w_cal = middle_cal + 4 * sw * (left_cal + right_cal) - oft_kw * left_hold

    hold_value = (kw - 1 + 3) * (left_hold + right_hold + middle_hold)
    # bypass = 4*sw*(left_bypass + right_bypass) - oft_kw*right_hold;
    bypass = 4 * sw * (left_bypass + right_bypass) - oft_kw * right_hold

    # stride padding bypass
    # aaa0 = oft_sw*ceil(iw/(4*sw))*ceil(oc/2)*(oh-uh-bh)*(ceil(ic/8)*kh-1);
    aaa0 = (
        oft_sw
        * math.ceil(iw / (4 * sw))
        * math.ceil(oc / 2)
        * (oh - uh - bh)
        * (math.ceil(ic / 8) * kh - 1)
    )

    aaa1 = 0
    aaa2 = 0

    # if(up~=0)
    #     for n=up:(0-sh):1
    #         aaa1 = aaa1 + oft_sw*ceil(iw/(4*sw))*ceil(oc/2)*(ceil(ic/8)*(kh-n)+n-1);
    #     end
    # end
    if up != 0:
        n = up
        while n >= 1:
            aaa1 = aaa1 + oft_sw * math.ceil(iw / (4 * sw)) * math.ceil(oc / 2) * (
                math.ceil(ic / 8) * (kh - n) + n - 1
            )
            n = n - sh

    # if(bp~=0)
    #     for n=bp:(0-sh):1
    #         aaa2 = aaa2 + oft_sw*ceil(iw/(4*sw))*ceil(oc/2)*(ceil(ic/8)*(kh-n)+n-1);
    #     end
    # end
    if bp != 0:
        n = bp
        while n >= 1:
            aaa2 = aaa2 + oft_sw * math.ceil(iw / (4 * sw)) * math.ceil(oc / 2) * (
                math.ceil(ic / 8) * (kh - n) + n - 1
            )
            n = n - sh

    aaa = aaa0 + aaa1 + aaa2

    ##################################################

    cal_cnt = (
        math.ceil(oc / 2) * math.ceil(ic / 8) * h * w_cal
        + math.ceil(oc / 2) * (uh + bh) * w_cal
    )
    cnt0 = oh * math.ceil(oc / 2) * (hold_value + bypass)
    cnt1 = 13 * math.ceil(oc / 2)

    # cnn_cnt = ceil(oc/2)*ceil(ic/8)*h*w_cal + oh*ceil(oc/2)*(hold+bypass) + 13*ceil(oc/2) - aaa;
    cnn_cnt = cal_cnt + cnt0 + cnt1 - aaa

    return cnn_cnt


def _dwconv2d_performance_v0(
    iw=5, ih=128, ic=16, kw=3, kh=3, oc=32, oh=126, lp=0, rp=0, up=0, bp=0, sw=1, sh=1
):
    # CNN 参数
    # iw = 4
    # ih = 4
    # ic = 16

    # kw = 3
    # kh = 3
    # oc = ic

    # oh = 2

    # lp = 0
    # rp = 0
    # up = 0
    # bp = 0

    # sw = 1
    # sh = 1

    # CNN
    if rp > 0:
        rp = sw * math.floor((iw + lp + rp - (kw - 1) + (sw - 1)) / sw) - iw - lp

    if bp > 0:
        bp = sh * math.floor((ih + up + bp - (kh - 1) + (sh - 1)) / sh) - ih - up

    if rp < 0:
        rp = 0
    if bp < 0:
        bp = 0

    up_h = math.ceil(up / sh)
    bp_h = math.ceil(bp / sh)

    h0 = kh * (oh - up_h - bp_h)
    h1 = 0
    h2 = 0

    uh = 0
    bh = 0

    if up != 0:
        n = up
        while n >= 1:
            h1 += kh - n
            uh += n
            n -= sh

    if bp != 0:
        n = bp
        while n >= 1:
            h2 += kh - n
            bh += n
            n -= sh

    h = h0 + h1 + h2

    if lp % sw == 0:
        oft_sw = 0
    else:
        oft_sw = sw - (lp % sw)
    kw = kw + oft_sw

    left_hold = 0
    left_bypass = 0
    left_cal = 0
    if lp > 0:
        if kw > 4 * sw * math.ceil(lp / (4 * sw)):
            bbb = 0
        else:
            bbb = 1
        left_hold = math.ceil(lp / (4 * sw)) - bbb
        for n in range(1, left_hold + 1):
            left_bypass += n
            left_cal += math.ceil(kw / (4 * sw)) - n

    right_hold = 0
    right_bypass = 0
    right_cal = 0

    if iw % (4 * sw) == 0:
        oft_iw = 0
    else:
        oft_iw = 4 * sw - (iw % (4 * sw))

    if kw % (4 * sw) == 0:
        oft_kw = 0
    else:
        oft_kw = 4 * sw - (kw % (4 * sw))

    rp = rp - oft_iw + oft_kw

    if rp > 0:
        right_hold = math.ceil(rp / (4 * sw))
        for n in range(1, math.ceil(rp / (4 * sw)) + 1):
            right_bypass += n
            right_cal += math.ceil(kw / (4 * sw)) - n

    middle_hold = math.ceil(iw / (4 * sw)) - math.floor((kw - 1) / (4 * sw))
    middle_cal = kw * middle_hold
    w_cal = middle_cal + 4 * sw * (left_cal + right_cal) - oft_kw * left_hold

    hold = (kw - 1 + 3) * (left_hold + right_hold + middle_hold)
    bypass = 4 * sw * (left_bypass + right_bypass) - oft_kw * right_hold

    # stride padding bypass
    aaa0 = oft_sw * math.ceil(iw / (4 * sw)) * math.ceil(oc / 2) * (oh - uh - bh)
    aaa1 = 0
    aaa2 = 0
    if up != 0:
        n = up
        while n >= 1:
            aaa1 += oft_sw * math.ceil(iw / (4 * sw)) * math.ceil(oc / 2)
            n -= sh

    if bp != 0:
        n = bp
        while n >= 1:
            aaa2 += oft_sw * math.ceil(iw / (4 * sw)) * math.ceil(oc / 2)
            n -= sh

    aaa = aaa0 + aaa1 + aaa2

    # cal_cnt = math.ceil(oc / 2) * math.ceil(ic / 8) * h * w_cal + math.ceil(oc / 2) * (uh + bh) * w_cal
    cal_cnt = math.ceil(oc / 2) * h * w_cal + math.ceil(oc / 2) * (uh + bh) * w_cal
    cnt0 = oh * math.ceil(oc / 2) * (hold + bypass)
    cnt1 = 13 * math.ceil(oc / 2)

    cnn_cnt = cal_cnt + cnt0 + cnt1 - aaa
    # print('cnn_cnt =', cnn_cnt)

    tmp = cnn_cnt - 16
    # print('tmp =', tmp)

    return tmp


@record_est_params()
def _deconv2d_performance_v0(
    iw=5, ih=128, ic=16, kw=3, kh=3, oc=32, oh=126, lp=0, rp=0, up=0, bp=0, sw=1, sh=1
):

    # # ----------- CNN 参数 -----------
    # iw = 10
    # ih = 8
    # ic = 16

    # kw = 5
    # kh = 5
    # oc = 8

    # oh = 11

    # lp = 0
    # rp = 0
    # up = 0
    # bp = 0

    # sw = 2
    # sh = 2

    # ----------- CNN 计算 -----------

    left_hold = 0
    left_bypass = 0
    left_cal = 0
    if lp > 0:
        if kw > 4:
            bbb = 0
        else:
            bbb = 1
        left_hold = math.ceil(lp / 4) - bbb
        for n in range(1, left_hold + 1):
            left_bypass += n
            left_cal += math.ceil(kw / 4) - n

    right_hold = 0
    right_bypass = 0
    right_cal = 0

    if (iw * sw) % 4 == 0:
        oft_iw = 0
    else:
        oft_iw = 4 - (iw * sw) % 4

    if kw % 4 == 0:
        oft_kw = 0
    else:
        oft_kw = 4 - kw % 4

    rp = rp - oft_iw + oft_kw

    if rp > 0:
        right_hold = math.ceil(rp / 4)
        for n in range(1, math.ceil(rp / 4) + 1):
            right_bypass += n
            right_cal += math.ceil(kw / 4) - n

    middle_hold = math.ceil(iw * sw / 4) - math.floor((kw - 1) / 4)
    middle_cal = kw * middle_hold
    w_cal = middle_cal + 4 * (left_cal + right_cal) - oft_kw * left_hold

    hold = (kw - 1 + 3) * (left_hold + right_hold + middle_hold)
    bypass = 4 * (left_bypass + right_bypass) - oft_kw * right_hold

    h = 0
    for n in range(1, ih + 1):
        if kh > up + (n - 1) * sh:
            if kh > bp + (ih - n) * sh + 1:
                h_tmp = up + bp + 2 - kh + (ih - 1) * sh
            else:
                h_tmp = up + (n - 1) * sh + 1
        else:
            if kh > bp + (ih - n) * sh + 1:
                h_tmp = bp + (ih - n) * sh + 1
            else:
                h_tmp = kh
        h += h_tmp

    cal_cnt = (
        math.ceil(oc / 2) * math.ceil(ic / 8) * h * w_cal
        + math.ceil(oc / 2) * (oh * kh - h) * w_cal
    )
    cnt0 = oh * math.ceil(oc / 2) * (hold + bypass)
    cnt1 = 13 * math.ceil(oc / 2)

    cnn_cnt = cal_cnt + cnt0 + cnt1
    # print(cnn_cnt)
    return cnn_cnt


def _mat_trans2d_v1(row: int, col: int, precision: int):
    """
    读取耗时：
    1. ceil(row/2)*ceil(col*precision/64) 一次读取2行
    2. 3*ceil(row/2) 一次读取2行,每个口下次读取需要跳1行
    写入耗时:
    1. ceil(col/2)*ceil(row*precision/64) 一次写2行
    2. 4*ceil(col/2) 一次写2行,每个口下次写入需要跳1行
    注意：
    1.如果读取psram，读取部分根据带宽计算。
    """
    cnt = (
        148
        + (
            3 * math.ceil(row / 2)
            + math.ceil(row / 2) * math.ceil(col * precision / 64)
        )
        + (
            4 * math.ceil(col / 2)
            + math.ceil(col / 2) * math.ceil(row * precision / 64)
        )
    )
    return cnt


def _est_matrix_mul_cycles(
    M: int,
    K: int,
    N: int,
    i_bits_0: int,
    i_bits_1: int,
    o_bits: int,
    i_device_0: int,
    i_device_1: int,
    o_device: int,
) -> int:
    """Calculate the total time consumption of matrix multiplication, unit is cycle"""
    assert i_device_0 in (0, 1, 2), "Only support share memory, psram and flash now"
    assert i_device_1 in (0, 1, 2), "Only support share memory, psram and flash now"
    assert o_device in (0,), "Only support share memory now"
    assert i_bits_0 in (4, 8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"

    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 2536
    macs = 64
    align_8 = (2, 8, 4)
    align_32 = (2, 1, 2)
    bias_en = 0

    if i_bits_0 == 8 and o_bits == 8:
        macs = 64
        config_cycles = 2155 + 100
    elif i_bits_0 == 16 and o_bits == 16:
        macs = 16
        config_cycles = 2155 + 100
    elif i_bits_0 == 32 and i_bits_1 == 32:
        macs = 4
        config_cycles = 2155 + 100

    if i_bits_0 == 8 and i_bits_1 == 8:
        M = math.ceil(M / align_8[0]) * align_8[0]
        K = math.ceil(K / align_8[1]) * align_8[1]
        N = math.ceil(N / align_8[2]) * align_8[2]
    elif i_bits_0 == 32 and i_bits_1 == 32:
        M = math.ceil(M / align_32[0]) * align_32[0]
        K = math.ceil(K / align_32[1]) * align_32[1]
        N = math.ceil(N / align_32[2]) * align_32[2]

    left_size = M * K * (i_bits_0 / 8)
    right_size = K * N * (i_bits_1 / 8)
    output_size = M * N * (o_bits / 8)
    calc_macs = M * N * K

    load_cycles_L = left_size / (share_in_bw / 8)
    load_cycles_R = right_size / (share_in_bw / 8)
    save_cycles = output_size / (share_out_bw / 8)
    if bias_en:
        calc_cycles = calc_macs / macs + math.ceil(M / 2) * (16)
    else:
        calc_cycles = calc_macs / macs + math.ceil(M / 2) * (16)
    # floor(b/32)!=floor(a/32) or floor(a/32)+1
    if 0:
        a, b = 0, 0
        extra_cycles = 0
        for mm in range(M - 1):
            b = a
            b += K * i_bits_0 / 8
            if math.floor(b / 32) not in [math.floor(a / 32), math.floor(a / 32) + 1]:
                extra_cycles += 4
            a = b
        print(extra_cycles)
        load_cycles_L += extra_cycles
        a, b = 0, 0
        extra_cycles = 0
        for mm in range(K - 1):
            b = a
            b += N * i_bits_1 / 8
            if math.floor(b / 32) not in [math.floor(a / 32), math.floor(a / 32) + 1]:
                extra_cycles += 4
            a = b
        print(extra_cycles)
        load_cycles_R += extra_cycles
    logging.debug(
        f"left_size: {left_size}, right_size: {right_size}, output_size: {output_size}, calc_macs: {calc_macs}"
    )
    logging.debug(
        f"config_cycles: {config_cycles}, load_cycles_L: {load_cycles_L}, load_cycles_R: {load_cycles_R}, save_cycles: {save_cycles}, calc_cycles: {calc_cycles}"
    )
    total_cycles = (
        gap_cycles
        + config_cycles
        + load_cycles_L
        + load_cycles_R
        + max(save_cycles, calc_cycles)
    )
    # return total_cycles, (gap_cycles, config_cycles, load_cycles_L, load_cycles_R, calc_cycles, save_cycles)
    return total_cycles


def _calc_conv2d_load_kernel_cycles(oc, ic, kh, kw):
    return (oc * ic * kh * kw) / (64 / 8)


def _calc_conv2d_load_input_cycles(ib, ic, ih, iw):
    return (ib * ic * ih * iw) / (64 / 8)


def _calc_conv2d_save_output_cycles(ob, oc, oh, ow):
    return (ob * oc * oh * ow) / (128 / 8)


def _calc_conv2d_cycles(
    input_c,
    input_h,
    input_w,
    output_c,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    padding_h_0,
    padding_h_1,
    padding_w_0,
    padding_w_1,
    groups,
    est_load_input=True,
    est_load_weight=True,
    est_calc=True,
    est_save=True,
):
    """Calculate the total time consumption of convolution, unit is cycle"""
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    weight_size_limit = CONV2D_WIEGHT_SIZE_LIMIT
    input_size_limit = CONV2D_INPUT_SIZE_LIMIT
    output_c_alignment = CONV2D_OUTPUT_C_ALIGNMENT
    input_c_alignment = CONV2D_INPUT_C_ALIGNMENT
    gap_cycles = 0
    config_cycles = 1350
    macs = 64
    load_cycles_input = 0
    load_cycles_weight = 0
    save_cycles = 0
    calc_cycles = 0

    input_w_a = _l_align_n(input_w, 4 * stride_w)
    input_c_a = _l_align_n(input_c, input_c_alignment)
    output_c_a = _l_align_n(output_c, output_c_alignment)
    output_h = (
        input_h + padding_h_0 + padding_h_1 - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w_a + padding_w_0 + padding_w_1 - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1

    input_size = 1 * input_c_a * input_h * input_w_a
    input_size_bw64 = (
        1 * input_c_a * input_h * _l_align_n(input_w_a, share_in_bw / 8)
    )  # fix: input need align 64bit
    weight_size = output_c_a * input_c_a * kernel_h * kernel_w
    output_size = 1 * output_c * output_h * output_w
    if input_size > input_size_limit:
        logging.warning(
            f"conv2d split_input, input_size({input_size}) > input_size_limit({input_size_limit})."
        )
    if weight_size > weight_size_limit:
        logging.warning(
            f"conv2d split_weight, weight_size({weight_size}) > weight_size_limit({weight_size_limit})."
        )

    if est_load_input:
        load_cycles_input = input_size_bw64 / (share_in_bw / 8)
    if est_load_weight:
        load_cycles_weight = weight_size / (share_in_bw / 8)
    if est_save:
        save_cycles = output_size / (share_out_bw / 8)
    if est_calc:
        calc_cycles = _cnn_performance_v5(
            input_w,
            input_h,
            input_c,
            kernel_w,
            kernel_h,
            output_c,
            output_h,
            padding_w_0,
            padding_w_1,
            padding_h_0,
            padding_h_1,
            stride_w,
            stride_h,
        )

    total_cycles = (
        gap_cycles
        + config_cycles
        + load_cycles_input
        + load_cycles_weight
        + max(save_cycles, calc_cycles)
    )

    logging.debug(
        f"est conv2d, config_cycles:{config_cycles}, load_cycles_input:{load_cycles_input}, load_cycles_weight:{load_cycles_weight}, calc_cycles:{calc_cycles}, save_cycles:{save_cycles}"
    )

    return total_cycles


def _calc_conv2d_cycles_for_split_weight(
    input_c,
    input_h,
    input_w,
    output_c,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    padding_h_0,
    padding_h_1,
    padding_w_0,
    padding_w_1,
    groups,
):
    """Calculate the total time consumption of convolution, unit is cycle"""
    assert groups == 1, "group must be 1"

    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    weight_size_limit = CONV2D_WIEGHT_SIZE_LIMIT
    input_size_limit = CONV2D_INPUT_SIZE_LIMIT
    output_c_alignment = CONV2D_OUTPUT_C_ALIGNMENT
    input_c_alignment = CONV2D_INPUT_C_ALIGNMENT
    gap_cycles = 0
    config_cycles = 1350
    startup_cycles = 300  # TODO: fix luna start cycles.
    macs = 64
    calc_macs = 0
    total_cycles = 0

    input_w_a = _l_align_n(input_w, 4 * stride_w)
    input_c_a = _l_align_n(input_c, input_c_alignment)
    output_c_a = _l_align_n(output_c, output_c_alignment)
    output_h = (
        input_h + padding_h_0 + padding_h_1 - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w_a + padding_w_0 + padding_w_1 - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    input_size = 1 * input_c_a * input_h * input_w_a
    weight_size = output_c_a * input_c_a * kernel_h * kernel_w
    weight_chw_size = input_c_a * kernel_h * kernel_w
    assert (
        output_c_alignment * weight_chw_size <= weight_size_limit
    ), f"output_c_alignment({output_c_alignment}) * weight_chw_size({weight_chw_size}) must less than weight_size_limit({weight_size_limit})."

    # split weight output_c
    output_c_max = int(
        weight_size_limit // weight_chw_size // output_c_alignment * output_c_alignment
    )
    output_c_remain = int(output_c % output_c_max)
    output_c_split_num = int(output_c // output_c_max)
    if output_c_remain > 0:
        output_c_split_num += 1

    logging.debug(
        f"conv2d split_weight, input_size: {input_size}, weight_size: {weight_size}, output_c_split_num: {output_c_split_num}, output_c_max: {output_c_max}, output_c_remain: {output_c_remain}"
    )
    if input_size > input_size_limit:
        logging.warning(
            f"conv2d split_input, input_size({input_size}) > input_size_limit({input_size_limit})."
        )

    input_cw_size = input_c_a * input_w_a
    assert (
        input_cw_size <= input_size_limit
    ), f"input_cw_size({input_cw_size}) must less than input_size_limit({input_size_limit})."

    max_h = int((input_size_limit / (1 * input_cw_size)) * 1)
    overlap_h = kernel_h - stride_h

    input_hs_first = min(max_h, input_h)
    input_hs_mid = max_h - overlap_h
    input_hs_last = max(input_h - max_h, 0) % (max_h - overlap_h)
    input_hn_fist = 1
    input_hn_mid = max(input_h - max_h, 0) // (max_h - overlap_h)
    input_hn_last = 1 if input_hs_last > 0 else 0

    logging.debug(
        f"conv2d split_input, input_hs_first: {input_hs_first}, input_hs_mid: {input_hs_mid}, input_hs_last: {input_hs_last}"
    )
    logging.debug(
        f"conv2d split_input, input_hn_fist: {input_hn_fist}, input_hn_mid: {input_hn_mid}, input_hn_last: {input_hn_last}"
    )

    for split_idx in range(output_c_split_num):
        if split_idx == output_c_split_num - 1 and output_c_remain > 0:
            output_c_split = output_c_remain
        else:
            output_c_split = output_c_max

        if True:
            split_cycles = _calc_conv2d_cycles(
                input_c,
                input_h,
                input_w,
                output_c_split,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                padding_h_0,
                padding_h_1,
                padding_w_0,
                padding_w_1,
                groups,
            )
            output_h_split_num = math.ceil(input_h / max_h)
            if output_h_split_num > 1:
                # fix: param config.
                split_cycles += (config_cycles - startup_cycles) * output_h_split_num
                # fix: channel interval.
                split_cycles += 3 * input_c
        else:
            start = 0
            split_cycles = 0
            input_hn = input_hn_fist + input_hn_mid + input_hn_last
            if input_hn > 1:  # fix: inv
                split_cycles += input_c * 3 * input_hn
            for i in range(input_hn):
                if i == 0:
                    end = min(start + max_h, input_h)
                    padded_start = max(0, start)
                    padded_end = min(input_h, end)
                    padded_h_p0 = padding_h_0
                    padded_h_p1 = 0
                    padded_h = input_hs_first
                elif i == input_hn - 1:
                    end = min(start + (max_h - overlap_h), input_h)
                    padded_start = max(0, start - overlap_h)  # expand up overlap_h
                    padded_end = min(input_h, end)
                    padded_h_p0 = 0
                    padded_h_p1 = padding_h_1
                    padded_h = input_hs_last
                else:
                    end = min(start + (max_h - overlap_h), input_h)
                    padded_start = max(0, start - overlap_h)  # expand up overlap_h
                    padded_end = min(input_h, end)
                    padded_h_p0 = 0
                    padded_h_p1 = 0
                    padded_h = input_hs_mid
                start = end
                est_load_weight = False
                if i == 0:
                    est_load_weight = True
                print(i, est_load_weight)
                split_cycles += _calc_conv2d_cycles(
                    input_c,
                    padded_h,
                    input_w,
                    output_c_split,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    padded_h_p0,
                    padded_h_p1,
                    padding_w_0,
                    padding_w_1,
                    groups,
                    est_load_weight=est_load_weight,
                )

        if split_idx > 0:
            total_cycles -= startup_cycles

        total_cycles += split_cycles

    return total_cycles


def _est_matrix_transponse_cycles(
    M: int, N: int, i_bits: int, o_bits: int, i_device: int, o_device: int
) -> int:
    """estimate trans op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    assert i_bits == o_bits, "Only support same bits input and output"

    gap_cycles = 890
    calc_cycles = _mat_trans2d_v1(M, N, i_bits)
    total_cycles = gap_cycles + calc_cycles

    return total_cycles


def _calc_dwconv2d_load_kernel_cycles(oc, ic, kh, kw):
    return (oc * ic * kh * kw) / (64 / 8)


def _calc_dwconv2d_load_input_cycles(ib, ic, ih, iw):
    return (ib * ic * ih * iw) / (64 / 8)


def _calc_dwconv2d_save_output_cycles(ob, oc, oh, ow):
    return (ob * oc * oh * ow) / (128 / 8)


def _calc_dwconv2d_cycles(
    input_c,
    input_h,
    input_w,
    output_c,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    padding_h_0,
    padding_h_1,
    padding_w_0,
    padding_w_1,
    groups,
    est_load_input=True,
    est_load_weight=True,
    est_calc=True,
    est_save=True,
):
    """Calculate the total time consumption of convolution, unit is cycle"""
    assert (
        input_c == output_c
    ), f"input_c{input_c} and output_c{output_c} must be equal."
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    weight_size_limit = CONV2D_WIEGHT_SIZE_LIMIT
    input_size_limit = CONV2D_INPUT_SIZE_LIMIT
    output_c_alignment = CONV2D_OUTPUT_C_ALIGNMENT
    input_c_alignment = CONV2D_INPUT_C_ALIGNMENT
    gap_cycles = 0
    config_cycles = 1350 - 70
    macs = 64
    input_w_a = _l_align_n(input_w, 4 * stride_w)
    input_c_a = _l_align_n(input_c, output_c_alignment)
    output_c_a = _l_align_n(output_c, output_c_alignment)
    output_h = (
        input_h + padding_h_0 + padding_h_1 - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w_a + padding_w_0 + padding_w_1 - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    load_cycles_input = 0
    load_cycles_weight = 0
    calc_cycles = 0

    if est_load_input:
        load_cycles_input = _calc_dwconv2d_load_input_cycles(
            1, input_c_a, input_h, input_w_a
        )
    if est_load_weight:
        load_cycles_weight = _calc_dwconv2d_load_kernel_cycles(
            1, input_c_a, kernel_h, kernel_w
        )
    if est_save:
        save_cycles = _calc_dwconv2d_save_output_cycles(1, output_c, output_h, output_w)
    if est_calc:
        calc_cycles = _dwconv2d_performance_v0(
            input_w,
            input_h,
            input_c,
            kernel_w,
            kernel_h,
            output_c,
            output_h,
            padding_w_0,
            padding_w_1,
            padding_h_0,
            padding_h_1,
            stride_w,
            stride_h,
        )

    total_cycles = (
        gap_cycles
        + config_cycles
        + load_cycles_input
        + load_cycles_weight
        + max(save_cycles, calc_cycles)
    )

    # print("\nweight_size:", kernel_size, "input_size:", input_size, "output_size:", output_size, "calc macs:", calc_macs)
    # print("config_cycles:", config_cycles , "load_cycles_weight:", load_cycles_weight , "load_cycles_input:", load_cycles_input , "calc_cycles:",  calc_cycles, "total_cycles:", total_cycles)
    return total_cycles


def _calc_deconv2d_load_kernel_cycles(oc, ic, kh, kw):
    return (oc * ic * kh * kw) / (64 / 8)


def _calc_deconv2d_load_input_cycles(ib, ic, ih, iw):
    return (ib * ic * ih * iw) / (64 / 8)


def _calc_deconv2d_save_output_cycles(ob, oc, oh, ow):
    return (ob * oc * oh * ow) / (128 / 8)


def _calc_deconv2d_cycles(
    input_c,
    input_h,
    input_w,
    output_c,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    padding_h_0,
    padding_h_1,
    padding_w_0,
    padding_w_1,
    groups,
    convert_torch2luna=False,
):
    """Calculate the total time consumption of deconvolution, unit is cycle"""
    assert groups == 1, "only support normal deconv2d."
    assert padding_h_0 == padding_h_1
    assert padding_w_0 == padding_w_1
    # logging.debug(f"before deconv convert param, input_c:{input_c}, input_h:{input_h}, input_w:{input_w}, output_c:{output_c}, kernel_h:{kernel_h}, kernel_w:{kernel_w}, stride_h:{stride_h}, stride_w:{stride_w}, \
    #                         dilation_h:{dilation_h}, dilation_w:{dilation_w}, padding_h_0:{padding_h_0}, padding_h_1:{padding_h_1}, padding_w_0:{padding_w_0}, padding_w_1:{padding_w_1}, groups:{groups}")
    # convert deconv param.
    if convert_torch2luna:
        out_pads_h = 0
        out_pads_w = 0
        # torch
        # import pdb; pdb.set_trace()
        input_h_new = (input_h - 1) * stride_h + 1
        input_w_new = (input_w - 1) * stride_w + 1
        pads_h_new = (kernel_h - 1) - padding_h_0 + out_pads_h
        pads_w_new = (kernel_w - 1) - padding_w_0 + out_pads_w
        # luna right += stride_w - 1
        padding_h_0 = pads_h_new
        padding_h_1 = pads_h_new
        padding_w_0 = pads_w_new
        padding_w_1 = pads_w_new - (stride_w - 1)  # input_w * stride_w
        assert padding_w_1 >= 0

        output_h_new = (input_h_new + 2 * pads_h_new - kernel_h) + 1
        output_w_new = (input_w_new + 2 * pads_w_new - kernel_w) + 1

        # print(input_h_new, input_w_new, pads_h_new, pads_w_new, output_h_new, output_w_new)

        output_h = output_h_new
        output_w = output_w_new
    else:
        input_h_new = (input_h - 1) * stride_h + 1
        input_w_new = (input_w - 1) * stride_w + (stride_w - 1)
        output_h_new = (input_h_new + padding_h_0 + padding_h_1 - kernel_h) + 1
        output_w_new = (input_w_new + padding_w_0 + padding_w_1 - kernel_w) + 1
        output_h = output_h_new
        output_w = output_w_new

    # logging.debug(f"after deconv convert param, input_c:{input_c}, input_h:{input_h}, input_w:{input_w}, output_c:{output_c}, kernel_h:{kernel_h}, kernel_w:{kernel_w}, stride_h:{stride_h}, stride_w:{stride_w}, \
    #     dilation_h:{dilation_h}, dilation_w:{dilation_w}, padding_h_0:{padding_h_0}, padding_h_1:{padding_h_1}, padding_w_0:{padding_w_0}, padding_w_1:{padding_w_1}, groups:{groups}, output_h:{output_h}, output_w:{output_w}")

    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    weight_size_limit = CONV2D_WIEGHT_SIZE_LIMIT
    input_size_limit = CONV2D_INPUT_SIZE_LIMIT
    output_c_alignment = CONV2D_OUTPUT_C_ALIGNMENT
    input_c_alignment = CONV2D_INPUT_C_ALIGNMENT
    gap_cycles = 0
    config_cycles = 1350
    macs = 64
    input_w_a = _l_align_n(input_w, 4 * stride_w)
    input_c_a = _l_align_n(input_c, input_c_alignment)
    output_c_a = _l_align_n(output_c, output_c_alignment)

    load_cycles_input = _calc_deconv2d_load_input_cycles(
        1, input_c_a, input_h, input_w_a
    )
    load_cycles_weight = _calc_deconv2d_load_kernel_cycles(
        output_c_a, input_c_a, kernel_h, kernel_w
    )
    save_cycles = _calc_deconv2d_save_output_cycles(1, output_c, output_h, output_w)

    calc_cycles = _deconv2d_performance_v0(
        input_w,
        input_h,
        input_c,
        kernel_w,
        kernel_h,
        output_c,
        output_h,
        padding_w_0,
        padding_w_1,
        padding_h_0,
        padding_h_1,
        stride_w,
        stride_h,
    )

    logging.debug(
        f"est deconv2d, config_cycles:{config_cycles}, load_cycles_input:{load_cycles_input}, load_cycles_weight:{load_cycles_weight}, calc_cycles:{calc_cycles}, save_cycles:{save_cycles}"
    )

    total_cycles = (
        gap_cycles
        + config_cycles
        + load_cycles_input
        + load_cycles_weight
        + max(save_cycles, calc_cycles)
    )

    return total_cycles


"""LUNA public performance model"""

# estimate vector op cycles


@record_est_params()
def est_vector_add_cycles(
    size: int,
    i_bits_0: int = 8,
    i_bits_1: int = 8,
    o_bits: int = 8,
    i_device_0: int = 0,
    i_device_1: int = 0,
    o_device: int = 0,
) -> int:
    """estimate add op cycles"""
    assert (
        i_device_0 == 0 and i_device_1 == 0 and o_device == 0
    ), "Only support share memory now"
    assert i_bits_0 in (8, 32), "Only support 8 or 32 bits input"
    assert i_bits_1 in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    assert i_bits_1 == i_bits_0, "Only support same bits input"
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0
    if i_bits_0 == 8:
        config_cycles = 534
    elif i_bits_0 == 32:
        config_cycles = 476
    load_cycles = (size * i_bits_0) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)
    # TOD: bank comflict error.
    return gap_cycles + config_cycles + max(load_cycles, save_cycles)


@record_est_params()
def est_vector_sub_cycles(
    size: int,
    i_bits_0: int = 8,
    i_bits_1: int = 8,
    o_bits: int = 8,
    i_device_0: int = 0,
    i_device_1: int = 0,
    o_device: int = 0,
) -> int:
    """estimate sub op cycles"""
    assert (
        i_device_0 == 0 and i_device_1 == 0 and o_device == 0
    ), "Only support share memory now"
    assert i_bits_0 in (8, 32), "Only support 8 or 32 bits input"
    assert i_bits_1 in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    assert i_bits_1 == i_bits_0, "Only support same bits input"
    return est_vector_add_cycles(
        size, i_bits_0, i_bits_1, o_bits, i_device_0, i_device_1, o_device
    )


@record_est_params()
def est_vector_mul_cycles(
    size: int,
    i_bits_0: int = 8,
    i_bits_1: int = 8,
    o_bits: int = 8,
    i_device_0: int = 0,
    i_device_1: int = 0,
    o_device: int = 0,
) -> int:
    """estimate mul op cycles"""
    assert (
        i_device_0 == 0 and i_device_1 == 0 and o_device == 0
    ), "Only support share memory now"
    assert i_bits_0 in (8, 32), "Only support 8 or 32 bits input"
    assert i_bits_1 in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 511

    if i_bits_0 == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW
    load_cycles = (size * i_bits_0) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)
    # TOD: same bank read/write will comflict, 128bit x 8bank = 1024bit.
    if i_bits_0 == 8 and o_bits == 8:
        config_cycles = 511 + 10
    elif i_bits_0 == 8 and o_bits == 32:
        config_cycles = 530
    elif i_bits_0 == 32 and o_bits == 8:
        config_cycles = 511 + 40
    elif i_bits_0 == 32 and o_bits == 32:
        config_cycles = 511 + 20

    return gap_cycles + config_cycles + max(load_cycles, save_cycles)


@record_est_params()
def est_vector_div_cycles(
    size: int,
    i_bits_0: int = 8,
    i_bits_1: int = 8,
    o_bits: int = 8,
    i_device_0: int = 0,
    i_device_1: int = 0,
    o_device: int = 0,
) -> int:
    """estimate div op cycles"""
    assert (
        i_device_0 == 0 and i_device_1 == 0 and o_device == 0
    ), "Only support share memory now"
    assert i_bits_0 in (32,), "Only support 32 bits input"
    assert i_bits_1 in (32,), "Only support 32 bits input"
    assert o_bits in (32,), "Only support 32 bits output"
    assert i_bits_1 == i_bits_0, "Only support same bits input"
    logging.warning("LUNA div op not supported yet, return 0")
    # luna wait x 8
    config_cycles = 900
    return config_cycles + 5 * size


@record_est_params()
def est_vector_scale_cycles(
    size: int,
    i_bits_0: int = 8,
    i_bits_1: int = 8,
    o_bits: int = 8,
    i_device_0: int = 0,
    i_device_1: int = 0,
    o_device: int = 0,
) -> int:
    """estimate scale op cycles"""
    assert (
        i_device_0 == 0 and i_device_1 == 0 and o_device == 0
    ), "Only support share memory now"
    assert i_bits_0 in (8, 32), "Only support 8 or 32 bits input"
    assert i_bits_1 in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    assert i_bits_1 == i_bits_0, "Only support same bits input"

    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 10
    config_cycles = 516

    if i_bits_0 == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW

    load_cycles = (size * i_bits_0) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    if i_bits_0 == 8 and o_bits == 8:
        config_cycles = 516 + 10
    elif i_bits_0 == 8 and o_bits == 32:
        config_cycles = 550 - 20
    elif i_bits_0 == 32 and o_bits == 8:
        config_cycles = 509 + 35
    elif i_bits_0 == 32 and o_bits == 32:
        config_cycles = 516

    return gap_cycles + config_cycles + max(load_cycles, save_cycles)


@record_est_params()
def est_vector_offset_cycles(
    size: int,
    i_bits_0: int = 8,
    i_bits_1: int = 8,
    o_bits: int = 8,
    i_device_0: int = 0,
    i_device_1: int = 0,
    o_device: int = 0,
) -> int:
    """estimate offset op cycles"""
    assert (
        i_device_0 == 0 and i_device_1 == 0 and o_device == 0
    ), "Only support share memory now"
    assert i_bits_0 in (8, 32), "Only support 8 or 32 bits input"
    assert i_bits_1 in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    assert i_bits_1 == i_bits_0, "Only support same bits input"

    return est_vector_add_cycles(
        size, i_bits_0, i_bits_1, o_bits, i_device_0, i_device_1, o_device
    )


@record_est_params()
def est_vector_cmp_vv_cycles(
    size: int,
    i_bits_0: int = 8,
    i_bits_1: int = 8,
    o_bits: int = 8,
    i_device_0: int = 0,
    i_device_1: int = 0,
    o_device: int = 0,
) -> int:
    """estimate cmp vector and vector cycles"""
    assert (
        i_device_0 == 0 and i_device_1 == 0 and o_device == 0
    ), "Only support share memory now"
    assert i_bits_0 in (8, 32), "Only support 8 or 32 bits input"
    assert i_bits_1 in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    assert i_bits_1 == i_bits_0, "Only support same bits input"

    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits_0 == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW

    if i_bits_0 == 8:
        config_cycles = 534 - 50
    elif i_bits_0 == 32:
        config_cycles = 476

    load_cycles = (size * i_bits_0) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    return gap_cycles + config_cycles + max(save_cycles, load_cycles)


@record_est_params()
def est_vector_cmp_vs_cycles(
    size: int,
    i_bits_0: int = 8,
    i_bits_1: int = 8,
    o_bits: int = 8,
    i_device_0: int = 0,
    i_device_1: int = 0,
    o_device: int = 0,
) -> int:
    """estimate cmp vector and scalar op cycles"""
    assert (
        i_device_0 == 0 and i_device_1 == 0 and o_device == 0
    ), "Only support share memory now"
    assert i_bits_0 in (8, 32), "Only support 8 or 32 bits input"
    assert i_bits_1 in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    assert i_bits_1 == i_bits_0, "Only support same bits input"

    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits_0 == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW

    if i_bits_0 == 8:
        config_cycles = 534 - 50
    elif i_bits_0 == 32:
        config_cycles = 476

    load_cycles = (size * i_bits_0) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    return gap_cycles + config_cycles + max(save_cycles, load_cycles)


@record_est_params()
def est_vector_max_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate max op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"

    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits == 8:
        config_cycles = 534 - 60
    elif i_bits == 32:
        config_cycles = 476

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    return gap_cycles + config_cycles + max(save_cycles, load_cycles)


@record_est_params()
def est_vector_min_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate min op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"

    return est_vector_max_cycles(size, i_bits, o_bits, i_device, o_device)


@record_est_params()
def est_vector_sum_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate sum op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 10
    config_cycles = 516

    if i_bits == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    if i_bits == 8 and o_bits == 8:
        config_cycles = 516 + 10
    elif i_bits == 8 and o_bits == 32:
        config_cycles = 550 - 20
    elif i_bits == 8 and o_bits == 64:
        config_cycles = 550 - 20
    elif i_bits == 32 and o_bits == 8:
        config_cycles = 509 + 35
    elif i_bits == 32 and o_bits == 32:
        config_cycles = 516 + 30
    elif i_bits == 32 and o_bits == 64:
        config_cycles = 516

    return gap_cycles + config_cycles + max(load_cycles, save_cycles)


@record_est_params()
def est_vector_dot_prod_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate dot_prod op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32, 64), "Only support 8 or 32 or 64 bits output"
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 10
    config_cycles = 516

    if i_bits == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    if i_bits == 8 and o_bits == 8:
        config_cycles = 516 + 10
    elif i_bits == 8 and o_bits == 32:
        config_cycles = 550 - 20
    elif i_bits == 8 and o_bits == 64:
        config_cycles = 550 - 20
    elif i_bits == 32 and o_bits == 8:
        config_cycles = 509 + 35
    elif i_bits == 32 and o_bits == 32:
        config_cycles = 516 + 30
    elif i_bits == 32 and o_bits == 64:
        config_cycles = 516

    return gap_cycles + config_cycles + max(load_cycles, save_cycles)


# estimate misc op cycles
@record_est_params()
def est_dma_cycles1(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate dma op cycles"""
    assert i_device in (0, 1, 2), "Only support share memory, psram and flash now"
    assert o_device in (0, 1), "Only support share memory, psram now"
    assert i_bits in (8,), "Only support 8 or 32 bits input"
    assert o_bits in (8,), "Only support 8 or 32 bits output"

    # cycles = k* size +  b
    cycles = float(size) * 0.9534328833447452 + 15499.450654119093

    return int(cycles)


# estimate misc op cycles
@record_est_params()
def est_dma_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate dma op cycles"""
    assert i_device in (0, 1, 2), "Only support share memory, psram and flash now"
    assert o_device in (0, 1), "Only support share memory, psram now"
    assert i_bits in (8,), "Only support 8 or 32 bits input"
    assert o_bits in (8,), "Only support 8 or 32 bits output"

    mem_memcpy_dma_flash_share_coeff = [
        -5.982892466636664e-29,
        5.677883140421153e-24,
        -1.6565540011855167e-19,
        3.5625000000000013,
        5.999999999981517,
    ]
    mem_memcpy_dma_flash_share_x = [
        256,
        512,
        784,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        49152,
        65536,
    ]
    mem_memcpy_dma_flash_share_y = [
        2278,
        3814,
        5446,
        6890,
        13034,
        25322,
        49898,
        99050,
        197350,
        295658,
        393962,
    ]

    mem_memcpy_dma_share_share_coeff = [
        -5.982892466636664e-29,
        5.677883140421153e-24,
        -1.6565540011855167e-19,
        3.5625000000000013,
        5.999999999981517,
    ]
    mem_memcpy_dma_share_share_x = [
        256,
        512,
        784,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        49152,
        65536,
    ]
    mem_memcpy_dma_share_share_y = [
        801,
        970,
        1152,
        1307,
        1979,
        3323,
        6010,
        11385,
        22137,
        32891,
        43643,
    ]

    mem_memcpy_dma_psram_share_coeff = [
        -5.982892466636664e-29,
        5.677883140421153e-24,
        -1.6565540011855167e-19,
        3.5625000000000013,
        5.999999999981517,
    ]
    mem_memcpy_dma_psram_share_x = [
        256,
        512,
        784,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        49152,
        65536,
    ]
    mem_memcpy_dma_psram_share_y = [
        821,
        1150,
        1451,
        1645,
        2633,
        4612,
        8566,
        16468,
        32284,
        48106,
        63922,
    ]

    mem_memcpy_dma_share_psram_coeff = [
        1.3266921958093294e-17,
        -2.1340217834375686e-12,
        1.143291587851051e-07,
        0.7697210459372988,
        625.6178834483525,
    ]
    mem_memcpy_dma_share_psram_x = [
        256,
        512,
        784,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        49152,
        65536,
    ]
    mem_memcpy_dma_share_psram_y = [
        829,
        1021,
        1235,
        1405,
        2197,
        3778,
        6940,
        13261,
        25909,
        38560,
        51205,
    ]

    total_cycles = 0
    if i_device == 2 and o_device == 0:
        polynomial = _l_polyfit(
            np.array(mem_memcpy_dma_flash_share_x),
            np.array(mem_memcpy_dma_flash_share_y),
            4,
        )
        total_cycles = polynomial(size)
    elif i_device == 0 and o_device == 0:
        polynomial = _l_polyfit(
            np.array(mem_memcpy_dma_share_share_x),
            np.array(mem_memcpy_dma_share_share_y),
            4,
        )
        total_cycles = polynomial(size)
    elif i_device == 1 and o_device == 0:
        polynomial = _l_polyfit(
            np.array(mem_memcpy_dma_psram_share_x),
            np.array(mem_memcpy_dma_psram_share_y),
            4,
        )
        total_cycles = polynomial(size)
    else:
        polynomial = _l_polyfit(
            np.array(mem_memcpy_dma_share_psram_x),
            np.array(mem_memcpy_dma_share_psram_y),
            4,
        )
        total_cycles = polynomial(size)

    return int(total_cycles)


@record_est_params()
def est_memcpy_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate memcpy op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8,), "Only support 8 bits input"
    assert o_bits in (8,), "Only support 8 bits output"
    # TODO: need more accurate model
    share_in_bw = int(SHARE_IN_BW * 0.9)  # why 0.9?
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits == 8:
        config_cycles = 330

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    # return int(gap_cycles + config_cycles + max(save_cycles, load_cycles))
    misc_luna_memcpy_i8o8_x = [256, 512, 784, 1024, 2048, 4096, 8192, 16384]
    misc_luna_memcpy_i8o8_y = [498, 548, 570, 616, 755, 1054, 1629, 2776]
    misc_luna_memcpy_i8o8_a = 7.230311576634952
    misc_luna_memcpy_i8o8_b = 467.45329860949965
    polynomial = _l_polyfit(
        np.array(misc_luna_memcpy_i8o8_x), np.array(misc_luna_memcpy_i8o8_y)
    )
    # print(polynomial)
    return int(polynomial(size))


@record_est_params()
def est_psrammemcpy_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate memcpy op cycles"""
    assert i_device in (1, 2) and o_device == 0, "Only support psram/flash memory now"
    assert i_bits in (8,), "Only support 8 bits input"
    assert o_bits in (8,), "Only support 8 bits output"
    logging.warning("LUNA psrammemcpy warning, polyfit size(784) error.")
    # TODO: need more accurate model
    share_in_bw = int(SHARE_IN_BW * 0.9)  # why 0.9?
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits == 8:
        config_cycles = 330

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    mem_memcpy_psram2share_psram_share_x = [
        256,
        512,
        784,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        49152,
        65536,
    ]
    mem_memcpy_psram2share_psram_share_y = [
        837,
        1113,
        1416,
        2289,
        4128,
        7692,
        14895,
        29337,
        58205,
        87048,
        115908,
    ]
    mem_memcpy_psram2share_psram_share_coeff = [
        -3.447781705078598e-16,
        5.0017650401933625e-11,
        -2.3902052444210008e-06,
        1.8029903603093662,
        291.3270163792185,
    ]
    mem_memcpy_psram2share_psram_share_a = 0.5546341356081432
    mem_memcpy_psram2share_psram_share_b = 291.3270163792185
    polynomial = _l_polyfit(
        np.array(mem_memcpy_psram2share_psram_share_x),
        np.array(mem_memcpy_psram2share_psram_share_y),
    )
    # print(polynomial)
    return int(polynomial(size))


@record_est_params()
def est_memset_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate memset op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 16, 32), "Only support 8 or 16 or 32 bits input"
    assert o_bits in (8, 16, 32), "Only support 8 or 16 or 32 bits output"
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits == 8:
        config_cycles = 350
    elif i_bits == 16:
        config_cycles = 350 - 15
    elif i_bits == 32:
        config_cycles = 350 - 15

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    return gap_cycles + config_cycles + max(save_cycles, load_cycles)


@record_est_params()
def est_relu_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate relu op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 16 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 16 or 32 bits output"
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW

    if i_bits == 8:
        config_cycles = 534 - 50
    elif i_bits == 32:
        config_cycles = 476

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    return gap_cycles + config_cycles + max(save_cycles, load_cycles)


@record_est_params()
def est_prelu_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate prelu op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW

    if i_bits == 8:
        config_cycles = 534 - 50
    elif i_bits == 32:
        config_cycles = 476

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    return gap_cycles + config_cycles + max(save_cycles, load_cycles)


@record_est_params()
def est_relux_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate prelu op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits == 8:
        config_cycles = 534 - 60
    elif i_bits == 32:
        config_cycles = 476

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    return gap_cycles + config_cycles + max(save_cycles, load_cycles)


@record_est_params()
def est_sigmoid_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate sigmoid op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"

    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW

    config_cycles = 406

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    return gap_cycles + config_cycles + max(save_cycles, load_cycles)


@record_est_params()
def est_tanh_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate tanh op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    share_in_bw = SHARE_IN_BW
    share_out_bw = SHARE_OUT_BW
    gap_cycles = 0
    config_cycles = 0

    if i_bits == 8 and o_bits == 8:  # wait for 4 cycles
        share_in_bw = SHARE_IN_BW / 2
        share_out_bw = SHARE_OUT_BW

    config_cycles = 406

    load_cycles = (size * i_bits) / (share_in_bw)
    save_cycles = (size * o_bits) / (share_out_bw)

    return gap_cycles + config_cycles + max(save_cycles, load_cycles)


@record_est_params()
def est_exp_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate exp op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    # logging.warning("LUNA softmax not supported yet, return 0")
    total_cycles = 0
    loop_num = (size + 2047) // 2048
    for i in range(loop_num):
        if i == loop_num - 1:
            use_size = size - i * 2048
        else:
            use_size = 2048
        total_cycles += 150 + 1007 + use_size * 3  # iter 6/(64/32) = 3
    return total_cycles + 50

    # polynomial = _l_polyfit(
    #     np.array([256,512,784, 1024, 2048, 4096, 8192, 16384]),
    #     np.array([1971, 2742, 3575, 4271, 7348, 14652, 29250, 58450]), 5)
    # #print(polynomial)

    # return polynomial(size)


@record_est_params()
def est_softmax_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate softmax op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (32,), "Only support 32 bits input"
    assert o_bits in (32,), "Only support 32 bits output"
    # logging.warning("LUNA softmax not supported yet, return 0")
    # TODO: need more accurate model
    total_cycles = 0
    total_cycles += est_vector_max_cycles(size, i_bits, o_bits, i_device, o_device)
    total_cycles += est_vector_sub_cycles(
        size, i_bits, i_bits, o_bits, i_device, i_device, o_device
    )
    total_cycles += est_exp_cycles(size, i_bits, o_bits, i_device, o_device)
    total_cycles += est_vector_sum_cycles(size, i_bits, o_bits, i_device, o_device)
    total_cycles += est_vector_scale_cycles(
        size, i_bits, i_bits, o_bits, i_device, i_device, o_device
    )
    total_cycles += -1300
    return total_cycles


@record_est_params()
def est_logsoftmax_cycles(
    size: int, i_bits: int = 8, o_bits: int = 8, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate logsoftmax op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (32,), "Only support 32 bits input"
    assert o_bits in (32,), "Only support 32 bits output"
    # logging.warning("LUNA logsoftmax not supported yet, return 0")
    # TODO: need more accurate model
    total_cycles = 0
    total_cycles += est_vector_max_cycles(size, i_bits, o_bits, i_device, o_device)
    total_cycles += est_exp_cycles(size, i_bits, o_bits, i_device, o_device)
    total_cycles += est_vector_sum_cycles(size, i_bits, o_bits, i_device, o_device)
    total_cycles += est_vector_offset_cycles(
        size, i_bits, i_bits, o_bits, i_device, i_device, o_device
    )
    total_cycles += est_vector_offset_cycles(
        size, i_bits, i_bits, o_bits, i_device, i_device, o_device
    )
    total_cycles += -800
    return total_cycles


# estimate matrix op cycles


@record_est_params()
def est_mat_mul_bias_cycles(
    row: int,
    col: int,
    col2: int,
    i_bits_0: int = 8,
    i_bits_1: int = 8,
    i_bits_bias=32,
    o_bits: int = 8,
    i_device_0: int = 0,
    i_device_1: int = 0,
    i_device_bias: int = 0,
    o_device: int = 0,
) -> int:
    """estimate mat mul bias op cycles"""
    assert i_device_0 in (0, 1, 2), "Only support share memory, psram and flash now"
    assert i_device_1 in (0, 1, 2), "Only support share memory, psram and flash now"
    assert o_device in (0,), "Only support share memory now"
    assert i_bits_0 in (4, 8, 32), "Only support 8 or 32 bits input"
    assert i_bits_1 in (4, 8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    # logging.warning("LUNA mat mul op not supported yet, return 0")
    total_cycles = _est_matrix_mul_cycles(
        row, col, col2, i_bits_0, i_bits_1, o_bits, i_device_0, i_device_1, o_device
    )
    return total_cycles


@record_est_params()
def est_mat_trans_cycles(
    row: int,
    col: int,
    i_bits: int = 8,
    o_bits: int = 8,
    i_device: int = 0,
    o_device: int = 0,
) -> int:
    """estimate mat trans op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    return _est_matrix_transponse_cycles(row, col, i_bits, o_bits, i_device, o_device)


@record_est_params()
def est_mat_trans3d_cycles(
    d1: int,
    d2: int,
    d3: int,
    permte1: int,
    permute2: int,
    permute3: int,
    i_bits: int = 8,
    o_bits: int = 8,
    i_device: int = 0,
    o_device: int = 0,
) -> int:
    """estimate mat trans3d op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    logging.warning("LUNA mat trans3d op not supported yet, return 0")
    return 0


@record_est_params()
def est_mat_copy_cycles(
    channel: int,
    row: int,
    col: int,
    i_bits: int = 8,
    o_bits: int = 8,
    i_device: int = 0,
    o_device: int = 0,
) -> int:
    """estimate mat copy op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (8, 32), "Only support 8 or 32 bits input"
    assert o_bits in (8, 32), "Only support 8 or 32 bits output"
    logging.warning("LUNA mat copy op not supported yet, return 0")
    return 0


# estimate nn op cycles
@record_est_params()
def est_nn_conv2d_cycles(
    input_c: int,
    input_h: int,
    input_w: int,
    output_c: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
    padding_h_0: int = 0,
    padding_h_1: int = 0,
    padding_w_0: int = 0,
    padding_w_1: int = 0,
    groups: int = 1,
    i_bits_i: int = 8,
    i_bits_w: int = 8,
    i_bits_b: int = 8,
    o_bits: int = 8,
    i_device_i: int = 0,
    i_device_w: int = 0,
    i_device_b: int = 0,
    o_device: int = 0,
):
    """Calculate the total time consumption of conv2d convolution, unit is cycle"""
    # total_cycles = _calc_conv2d_cycles(input_c, input_h, input_w, output_c,
    #     kernel_h, kernel_w,
    #     stride_h, stride_w,
    #     dilation_h, dilation_w,
    #     padding_h_0, padding_h_1, padding_w_0, padding_w_1,
    #     groups)
    total_cycles = _calc_conv2d_cycles_for_split_weight(
        input_c,
        input_h,
        input_w,
        output_c,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        padding_h_0,
        padding_h_1,
        padding_w_0,
        padding_w_1,
        groups,
    )
    return total_cycles


@record_est_params()
def est_nn_deconv2d_cycles(
    input_c: int,
    input_h: int,
    input_w: int,
    output_c: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
    padding_h_0: int = 0,
    padding_h_1: int = 0,
    padding_w_0: int = 0,
    padding_w_1: int = 0,
    groups: int = 1,
    i_bits_i: int = 8,
    i_bits_w: int = 8,
    i_bits_b: int = 8,
    o_bits: int = 8,
    i_device_i: int = 0,
    i_device_w: int = 0,
    i_device_b: int = 0,
    o_device: int = 0,
    convert_torch2luna=False,
):
    """Calculate the total time consumption of deconv2d convolution, unit is cycle"""
    # logging.warning("LUNA deconv2d op not supported yet, return 0")
    total_cycles = _calc_deconv2d_cycles(
        input_c,
        input_h,
        input_w,
        output_c,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        padding_h_0,
        padding_h_1,
        padding_w_0,
        padding_w_1,
        groups,
        convert_torch2luna,
    )
    return total_cycles


@record_est_params()
def est_nn_depthwise2d_cycles(
    input_c: int,
    input_h: int,
    input_w: int,
    output_c: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
    padding_h_0: int = 0,
    padding_h_1: int = 0,
    padding_w_0: int = 0,
    padding_w_1: int = 0,
    groups: int = 1,
    i_bits_i: int = 8,
    i_bits_w: int = 8,
    i_bits_b: int = 8,
    o_bits: int = 8,
    i_device_i: int = 0,
    i_device_w: int = 0,
    i_device_b: int = 0,
    o_device: int = 0,
):
    """Calculate the total time consumption of depthwise2d convolution, unit is cycle"""
    total_cycles = _calc_dwconv2d_cycles(
        input_c,
        input_h,
        input_w,
        output_c,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        padding_h_0,
        padding_h_1,
        padding_w_0,
        padding_w_1,
        groups,
    )
    return total_cycles


@record_est_params()
def est_nn_maxpool2d_cycles(
    input_c: int,
    input_h: int,
    input_w: int,
    output_c: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
    padding_h_0: int = 0,
    padding_h_1: int = 0,
    padding_w_0: int = 0,
    padding_w_1: int = 0,
    groups: int = 1,
    i_bits_i: int = 8,
    i_bits_w: int = 8,
    i_bits_b: int = 8,
    o_bits: int = 8,
    i_device_i: int = 0,
    i_device_w: int = 0,
    i_device_b: int = 0,
    o_device: int = 0,
):
    """Calculate the total time consumption of maxpooling2d, unit is cycle"""
    # logging.warning("LUNA maxpool op not supported yet, return 0")
    total_cycles = _calc_dwconv2d_cycles(
        input_c,
        input_h,
        input_w,
        output_c,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        padding_h_0,
        padding_h_1,
        padding_w_0,
        padding_w_1,
        groups,
        est_load_weight=False,
    )
    return total_cycles


@record_est_params()
def est_nn_avgpool2d_cycles(
    input_c: int,
    input_h: int,
    input_w: int,
    output_c: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int = 0,
    dilation_w: int = 0,
    padding_h_0: int = 0,
    padding_h_1: int = 0,
    padding_w_0: int = 0,
    padding_w_1: int = 0,
    groups: int = 1,
    i_bits_i: int = 8,
    i_bits_w: int = 8,
    i_bits_b: int = 8,
    o_bits: int = 8,
    i_device_i: int = 0,
    i_device_w: int = 0,
    i_device_b: int = 0,
    o_device: int = 0,
):
    """Calculate the total time consumption of avgpooling2d, unit is cycle"""
    # logging.warning("LUNA avgpooling2d op not supported yet, return 0")
    total_cycles = _calc_dwconv2d_cycles(
        input_c,
        input_h,
        input_w,
        output_c,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        padding_h_0,
        padding_h_1,
        padding_w_0,
        padding_w_1,
        groups,
        est_load_weight=False,
    )
    return total_cycles


# estimate fft op cycles
@record_est_params()
def est_cfft_cycles(
    size: int, i_bits: int = 32, o_bits: int = 32, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate max op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (32,), "Only 32 bits input"
    assert o_bits in (32,), "Only 32 bits output"
    assert size in (64, 128, 256, 512), "Only support 64,128,256,512 size"
    fft_cycles = {64: 2320, 128: 2734, 256: 3680, 512: 5542}
    return fft_cycles[size]


@record_est_params()
def est_cifft_cycles(
    size: int, i_bits: int = 32, o_bits: int = 32, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate max op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (32,), "Only 32 bits input"
    assert o_bits in (32,), "Only 32 bits output"
    assert size in (64, 128, 256, 512), "Only support 64,128,256,512 size"
    return est_cfft_cycles(size, i_bits, o_bits, i_device, o_device)


@record_est_params()
def est_rfft_cycles(
    size: int, i_bits: int = 32, o_bits: int = 32, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate max op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (32,), "Only 32 bits input"
    assert o_bits in (32,), "Only 32 bits output"
    assert size in (64, 128, 256, 512), "Only support 64,128,256,512 size"
    return est_cfft_cycles(size, i_bits, o_bits, i_device, o_device)


@record_est_params()
def est_rifft_cycles(
    size: int, i_bits: int = 32, o_bits: int = 32, i_device: int = 0, o_device: int = 0
) -> int:
    """estimate max op cycles"""
    assert i_device == 0 and o_device == 0, "Only support share memory now"
    assert i_bits in (32,), "Only 32 bits input"
    assert o_bits in (32,), "Only 32 bits output"
    assert size in (64, 128, 256, 512), "Only support 64,128,256,512 size"
    return est_cfft_cycles(size, i_bits, o_bits, i_device, o_device)

__all__ = ['']