#ifndef _CONV2DINT_VENUS_H_
#define _CONV2DINT_VENUS_H_

#include <math.h>
#include <stdint.h>
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_cnn_tools.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"

/**
 * @brief Quantize and round up a 32-bit integer
 * @param x Input integer
 * @param shift Number of bits to shift right
 * @return int32_t Quantized result
 */
static int32_t luna_quant_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Initialize parameters for 2D convolution
 * @param attrs Convolution attributes
 * @param conv_attrs Convolution structure for 2D convolution
 * @param X Input tensor
 * @param W Weight tensor
 * @param Bias Bias tensor (optional)
 * @param Y Output tensor
 */
static void conv2dint_luna_para_init(Conv2dIntAttrs *attrs, conv_struct_t *conv_attrs, tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y) {
    memset(conv_attrs, 0, sizeof(conv_struct_t));
    conv_attrs->is_bias = (Bias != NULL) ? 1 : 0;

    conv_attrs->input_c = X->shape_.dims_[1];
    conv_attrs->input_h = X->shape_.dims_[2];
    conv_attrs->input_w = X->shape_.dims_[3];
    conv_attrs->output_c = Y->shape_.dims_[1];
    conv_attrs->output_h = Y->shape_.dims_[2];
    conv_attrs->output_w = Y->shape_.dims_[3];
    conv_attrs->weight_h = attrs->kernel[0];
    conv_attrs->weight_w = attrs->kernel[1];
    conv_attrs->stride_h = attrs->stride[0];
    conv_attrs->stride_w = attrs->stride[1];
    conv_attrs->padding_h_up = attrs->pad[0];
    conv_attrs->padding_h_down = attrs->pad[2];
    conv_attrs->padding_w_left = attrs->pad[1];
    conv_attrs->padding_w_right = attrs->pad[3];
    conv_attrs->dilation_h = attrs->dilation[0];
    conv_attrs->dilation_w = attrs->dilation[1];

    switch (attrs->act_type) {
        case 1:
            conv_attrs->activation_type = RELU;
            break;
        case 2:
            conv_attrs->activation_type = PRELU;
            break;
        default:
            conv_attrs->activation_type = NO_ACTIVE;
            break;
    }

    int32_t q_x = (int32_t)X->scale_;
    int32_t q_w = (int32_t)W->scale_;
    int32_t q_y = (int32_t)Y->scale_;
    conv_attrs->positive_shift_type = ShiftType_FloorX05;
    conv_attrs->positive_shift_value = q_x + q_w - q_y;
    conv_attrs->negative_shift_type = ShiftType_FloorX05;
    conv_attrs->negative_shift_value = conv_attrs->positive_shift_value;

    uint8_t data_mem_type = (X->mem_.type_ & 0x0F) + 1;
    data_mem_type = (data_mem_type == 3) ? 0 : data_mem_type;
    uint8_t weight_mem_type = (W->mem_.type_ & 0x0F) + 1;
    weight_mem_type = (weight_mem_type == 3) ? 0 : weight_mem_type;
    conv_attrs->data_mem_type = (data_mem_type << 4) | weight_mem_type;
    conv_attrs->ou_bits = Y->byte_ * 8;
    conv_attrs->weight_bits = (W->dtype_ == Int4) ? 4 : 8;
    conv_attrs->out_padding_h = 0;
    conv_attrs->out_padding_w = 0;
    conv_attrs->group = attrs->group;
}

static int32_t conv2dint_luna_calc_conv(tTensor *W, tTensor *Y, int8_t *src,
                                        int8_t *weight, int32_t *bias,
                                        void *dst,
                                        luna_cnn_static_para_t *conv_static_para) {
    if (W->dtype_ == Int4) {
        if (Y->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(conv2d_i8i4o8)(src, weight, bias, (int8_t *)dst, conv_static_para), "luna_conv2d_i8i4o8");
        } else if (Y->dtype_ == Int32) {
            THINKER_RET_CHECK(API_LIB(conv2d_i8i4o32)(src, weight, bias, (int32_t *)dst, conv_static_para), "luna_conv2d_i8i4o32");
        } else {
            return T_ERR_INVALID_DATATYPE;
        }
    } else if (W->dtype_ == Int8) {
        if (Y->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(conv2d_i8i8o8)(src, weight, bias, (int8_t *)dst, conv_static_para), "luna_conv2d_i8i8o8");
        } else if (Y->dtype_ == Int32) {
            THINKER_RET_CHECK(API_LIB(conv2d_i8i8o32)(src, weight, bias, (int32_t *)dst, conv_static_para), "luna_conv2d_i8i8o32");
        } else {
            return T_ERR_INVALID_DATATYPE;
        }
    } else {
        return T_ERR_INVALID_DATATYPE;
    }
    return T_SUCCESS;
}

static int32_t conv2dint_luna_calc_depthwise(tTensor *W, tTensor *Y, int8_t *src,
                                             int8_t *weight, int32_t *bias,
                                             void *dst,
                                             luna_cnn_static_para_t *conv_static_para) {
    if (W->dtype_ == Int4) {
        if (Y->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(depthwise2d_i8i4o8)(src, weight, bias, (int8_t *)dst, conv_static_para), "luna_depthwise2d_i8i4o8");
        } else if (Y->dtype_ == Int32) {
            THINKER_RET_CHECK(API_LIB(depthwise2d_i8i4o32)(src, weight, bias, (int32_t *)dst, conv_static_para), "luna_depthwise2d_i8i4o32");
        } else {
            return T_ERR_INVALID_DATATYPE;
        }
    } else if (W->dtype_ == Int8) {
        if (Y->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(depthwise2d_i8i8o8)(src, weight, bias, (int8_t *)dst, conv_static_para), "luna_depthwise2d_i8i8o8");
        } else if (Y->dtype_ == Int32) {
            THINKER_RET_CHECK(API_LIB(depthwise2d_i8i8o32)(src, weight, bias, (int32_t *)dst, conv_static_para), "luna_depthwise2d_i8i8o32");
        } else {
            return T_ERR_INVALID_DATATYPE;
        }
    } else {
        return T_ERR_INVALID_DATATYPE;
    }
    return T_SUCCESS;
}

static void conv2dint_luna_split_h_info(conv_struct_t *conv_attrs,
                                          int32_t ou_h_start, int32_t cur_ou_h,
                                          int32_t *in_h_start, int32_t *cur_in_h,
                                          uint8_t *pad_h_up, uint8_t *pad_h_down) {
    int32_t kernel_extent_h = (conv_attrs->weight_h - 1) * conv_attrs->dilation_h + 1;
    int32_t raw_in_start = ou_h_start * conv_attrs->stride_h - conv_attrs->padding_h_up;
    int32_t raw_in_end = raw_in_start + (cur_ou_h - 1) * conv_attrs->stride_h + kernel_extent_h - 1;
    int32_t split_in_start = (raw_in_start > 0) ? raw_in_start : 0;
    int32_t split_in_end = (raw_in_end < (int32_t)conv_attrs->input_h) ?
        raw_in_end : (int32_t)conv_attrs->input_h - 1;

    *in_h_start = split_in_start;
    *cur_in_h = split_in_end - split_in_start + 1;
    *pad_h_up = (raw_in_start < 0) ? (uint8_t)(-raw_in_start) : 0;
    *pad_h_down = (raw_in_end >= (int32_t)conv_attrs->input_h) ?
        (uint8_t)(raw_in_end - (int32_t)conv_attrs->input_h + 1) : 0;
}

static int32_t conv2dint_luna_split_workspace(conv_struct_t *conv_attrs,
                                              tTensor *Y, int32_t split_ou_h,
                                              int32_t *input_size,
                                              int32_t *output_size) {
    int32_t kernel_extent_h = (conv_attrs->weight_h - 1) * conv_attrs->dilation_h + 1;
    int32_t split_in_h = (split_ou_h - 1) * conv_attrs->stride_h + kernel_extent_h;

    *input_size = conv_attrs->input_c * split_in_h * conv_attrs->input_w;
    *output_size = conv_attrs->output_c * split_ou_h * conv_attrs->output_w * Y->byte_;
    return ALIGN4(*input_size) + *output_size;
}

static int32_t conv2dint_luna_split_ou_h(conv_struct_t *conv_attrs,
                                         tTensor *Y, int32_t workspace_size) {
    int32_t log2n_stride_w = (conv_attrs->stride_w >> 1);
    int32_t align_c = luna_quant_ceil(conv_attrs->input_c, 3) << 3;
    int32_t input_size_align_withouth = align_c *
        (luna_quant_ceil(conv_attrs->input_w, (2 + log2n_stride_w)) << (2 + log2n_stride_w));

    if (input_size_align_withouth <= 0 || workspace_size <= 0) {
        return 0;
    }

    for (int32_t split_ou_h = conv_attrs->output_h; split_ou_h > 0; split_ou_h--) {
        int32_t input_size = 0;
        int32_t output_size = 0;
        int32_t kernel_extent_h = (conv_attrs->weight_h - 1) * conv_attrs->dilation_h + 1;
        int32_t split_in_h = (split_ou_h - 1) * conv_attrs->stride_h + kernel_extent_h;

        if (split_in_h * input_size_align_withouth > CONV_IN_CONDITION) {
            continue;
        }
        if (conv2dint_luna_split_workspace(conv_attrs, Y, split_ou_h,
                                           &input_size, &output_size) <= workspace_size) {
            return split_ou_h;
        }
    }
    return 0;
}

static tStatus conv2dint_luna_copy_input_h(tTensor *X, int8_t *dst,
                                        int32_t in_h_start, int32_t cur_in_h,
                                        conv_struct_t *conv_attrs) {
    int8_t *src = (int8_t *)(X->dptr_);
    int32_t src_channel_size = conv_attrs->input_h * conv_attrs->input_w;
    int32_t dst_channel_size = cur_in_h * conv_attrs->input_w;
    int32_t copy_size = dst_channel_size;

    for (int32_t c = 0; c < conv_attrs->input_c; c++) {
        int8_t *src_ptr = src + c * src_channel_size + in_h_start * conv_attrs->input_w;
        int8_t *dst_ptr = dst + c * dst_channel_size;
        THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst_ptr, src_ptr, copy_size), "luna_memcpy_i8o8");
    }
    return T_SUCCESS;
}

static int32_t conv2dint_luna_split_output_h(
    tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y, int8_t *temp,
    int32_t workspace_size, conv_struct_t *conv_attrs, int32_t depthwise) {
    int32_t split_ou_h = conv2dint_luna_split_ou_h(conv_attrs, Y, workspace_size);
    int32_t output_channel_size = conv_attrs->output_h * conv_attrs->output_w * Y->byte_;
    int8_t *weight = (int8_t *)(W->dptr_);
    int32_t *bias = (Bias != NULL) ? (int32_t *)(Bias->dptr_) : NULL;

    if (temp == NULL || split_ou_h <= 0) {
        return T_ERR_NO_WORKSPACE;
    }

    for (int32_t ou_h_start = 0; ou_h_start < (int32_t)conv_attrs->output_h; ou_h_start += split_ou_h) {
        int32_t cur_ou_h = conv_attrs->output_h - ou_h_start;
        int32_t in_h_start = 0;
        int32_t cur_in_h = 0;
        int32_t input_size = 0;
        int32_t output_size = 0;
        uint8_t pad_h_up = 0;
        uint8_t pad_h_down = 0;
        conv_struct_t split_attrs = *conv_attrs;
        luna_cnn_static_para_t split_static_para;

        if (cur_ou_h > split_ou_h) {
            cur_ou_h = split_ou_h;
        }
        conv2dint_luna_split_h_info(conv_attrs, ou_h_start, cur_ou_h,
                                     &in_h_start, &cur_in_h,
                                     &pad_h_up, &pad_h_down);
        if (cur_in_h <= 0) {
            return T_ERR_INVALID_PARA;
        }

        input_size = conv_attrs->input_c * cur_in_h * conv_attrs->input_w;
        output_size = conv_attrs->output_c * cur_ou_h * conv_attrs->output_w * Y->byte_;
        if (ALIGN4(input_size) + output_size > workspace_size) {
            return T_ERR_NO_WORKSPACE;
        }

        int8_t *input_tmp = temp;
        int8_t *output_tmp = temp + ALIGN4(input_size);
        THINKER_RET_CHECK(conv2dint_luna_copy_input_h(X, input_tmp, in_h_start, cur_in_h, conv_attrs), "conv2dint_luna_copy_input_h");

        split_attrs.input_h = cur_in_h;
        split_attrs.output_h = cur_ou_h;
        split_attrs.padding_h_up = pad_h_up;
        split_attrs.padding_h_down = pad_h_down;
        split_attrs.data_mem_type = split_attrs.data_mem_type & 0x0F;
        split_attrs.reserved = 0;
        THINKER_RET_CHECK(luna_split_conv_para_pack(&split_attrs, &split_static_para,
                          depthwise ? LUNA_DEPTHWISE : LUNA_CONV), "luna_split_conv_para_pack");
        if (depthwise) {
            THINKER_RET_CHECK(conv2dint_luna_calc_depthwise(W, Y, input_tmp, weight, bias,
                                                            output_tmp, &split_static_para),
                              "conv2dint_luna_calc_depthwise");
        } else {
            THINKER_RET_CHECK(conv2dint_luna_calc_conv(W, Y, input_tmp, weight, bias,
                                                       output_tmp, &split_static_para),
                              "conv2dint_luna_calc_conv");
        }

        int32_t temp_channel_size = conv_attrs->output_w * cur_ou_h * Y->byte_;
        int32_t dst_h_offset = ou_h_start * conv_attrs->output_w * Y->byte_;
        for (int32_t c = 0; c < conv_attrs->output_c; c++) {
            int8_t *dst = (int8_t *)Y->dptr_ + c * output_channel_size + dst_h_offset;
            int8_t *src = output_tmp + c * temp_channel_size;
            if (Y->mem_.type_ != 2) {
                opi_psram_cpy_out(dst, src, temp_channel_size);
            }
            else {
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst, src, temp_channel_size), "luna_memcpy_i8o8");
            }
        }
    }
    return T_SUCCESS;
}

/**
 * @brief Execute 2D convolution with integer precision
 * @param X Input tensor
 * @param W Weight tensor
 * @param Bias Bias tensor (optional)
 * @param Y Output tensor
 * @param Temp Workspace tensor
 * @param attrs Convolution attributes
 * @return int32_t Execution status
 */
int32_t conv2dint_luna(tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y, tTensor *Temp, Conv2dIntAttrs *attrs) {
    int8_t *src = (int8_t *)(X->dptr_);
    int8_t *weight = (int8_t *)(W->dptr_);
    int32_t *bias = (Bias != NULL) ? (int32_t *)(Bias->dptr_) : NULL;
    int8_t *dst = (int8_t *)(Y->dptr_);
    int8_t *temp = (Temp != NULL) ? (int8_t *)(Temp->dptr_) : NULL;
    int32_t workspace_size = (Temp != NULL) ? Temp->shape_.dims_[0] : 0;

    conv_struct_t conv_attrs;
    luna_cnn_static_para_t conv_static_para;
    conv2dint_luna_para_init(attrs, &conv_attrs, X, W, Bias, Y);

    uint32_t input_c = conv_attrs.input_c;
    uint32_t output_c = conv_attrs.output_c;
    int32_t k_h = conv_attrs.weight_h;
    int32_t k_w = conv_attrs.weight_w;
    int32_t ou_h = conv_attrs.output_h;
    int32_t ou_w = conv_attrs.output_w;
    int32_t ou_is_psram = (Y->mem_.type_ != 2) ? 1 : 0;
    int32_t output_size = output_c * ou_h * ou_w * Y->byte_;
    int32_t log2n_stride_w = (conv_attrs.stride_w >> 1);
    int32_t input_size_align_withouth = (luna_quant_ceil(input_c, 3) << 3) *
        (luna_quant_ceil(conv_attrs.input_w, (2 + log2n_stride_w)) << (2 + log2n_stride_w));
    int32_t input_condition = (input_size_align_withouth * conv_attrs.input_h <= CONV_IN_CONDITION) ? 1 : 0;

    if (X->dtype_ != Int8 || (W->dtype_ != Int4 && W->dtype_ != Int8)) {
        return T_ERR_INVALID_DATATYPE;
    }
    if (Y->dtype_ != Int8 && Y->dtype_ != Int32) {
        return T_ERR_INVALID_DATATYPE;
    }
    if (Bias != NULL && Bias->dtype_ != Int32) {
        return T_ERR_INVALID_DATATYPE;
    }
    if (ou_is_psram && (temp == NULL || workspace_size <= 0)) {
        return T_ERR_NO_WORKSPACE;
    }

    if ((k_h <= 12) && (k_w <= 12)) { // Kernel size in [1, 12]
        if (attrs->group == 1) { // Common convolution
            if (ou_is_psram && output_size > workspace_size) {
                THINKER_RET_CHECK(conv2dint_luna_split_output_h(X, W, Bias, Y, temp,
                                                     workspace_size, &conv_attrs, 0), "conv2dint_luna_split_output_h");
            }
            conv_attrs.reserved = 0;
            THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_CONV), "luna_split_conv_para_pack");
            if (ou_is_psram) {
                dst = temp;
            }
            THINKER_RET_CHECK(conv2dint_luna_calc_conv(W, Y, src, weight, bias, dst,
                                                       &conv_static_para), "conv2dint_luna_calc_conv");
            if (ou_is_psram) {
                opi_psram_cpy_out((int8_t *)Y->dptr_, dst, output_size);
            }
        } else if (attrs->group == input_c && attrs->group == output_c) { // Depthwise convolution
            if (ou_is_psram && output_size > workspace_size) {
                THINKER_RET_CHECK(conv2dint_luna_split_output_h(X, W, Bias, Y, temp,
                                                     workspace_size, &conv_attrs, 1), "conv2dint_luna_split_output_h");
            }
            conv_attrs.reserved = 0;
            THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_DEPTHWISE), "luna_split_conv_para_pack");
            if (ou_is_psram) {
                dst = temp;
            }
            THINKER_RET_CHECK(conv2dint_luna_calc_depthwise(W, Y, src, weight, bias, dst,
                                                            &conv_static_para), "conv2dint_luna_calc_depthwise");
            if (ou_is_psram) {
                opi_psram_cpy_out((int8_t *)Y->dptr_, dst, output_size);
            }
        } else { // Group convolution, should be split in tpacker
            return T_ERR_INVALID_PARA;
        }
    } else {
        printf("conv2d do not support: kernel > 12\n");
        return T_ERR_INVALID_PARA;
    }
#if !(defined(WIN32) || defined(linux))
    if (ou_is_psram)
        HAL_FlushInvalidateDCache_by_Addr((uint32_t *)(Y->dptr_), output_size);
#endif
    return T_SUCCESS;
}

#endif  // _CONV2DINT_VENUS_H_
