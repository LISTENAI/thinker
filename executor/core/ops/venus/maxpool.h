#ifndef _MAXPOOL_LUNA_H_
#define _MAXPOOL_LUNA_H_

#include <stdio.h>
#include <string.h>
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "core/operator_attrs.h"
#include "c_api/thinker_define.h"

/**
 * @brief Quantized ceiling function for integer division
 * @param x Input integer
 * @param shift Number of bits to shift
 * @return int32_t Result after quantized ceiling operation
 */
static int32_t luna_quant_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Initialize parameters for max pooling operation
 * @param attrs Pooling attributes
 * @param conv_attrs Pointer to convolution structure for storing parameters
 * @param X Input tensor
 * @param Y Output tensor
 */
static void luna_maxpool_para_init(PoolAttrs* attrs, s_conv_struct *conv_attrs, tTensor *X, tTensor *Y) {
    memset(conv_attrs, 0, sizeof(s_conv_struct));

    // Set input dimensions
    conv_attrs->input_c = X->shape_.dims_[1];
    conv_attrs->input_h = X->shape_.dims_[2];
    conv_attrs->input_w = X->shape_.dims_[3];

    // Set output dimensions
    conv_attrs->output_c = Y->shape_.dims_[1];
    conv_attrs->output_h = Y->shape_.dims_[2];
    conv_attrs->output_w = Y->shape_.dims_[3];

    // Set kernel and stride dimensions
    conv_attrs->weight_h = attrs->kernel[0];
    conv_attrs->weight_w = attrs->kernel[1];
    conv_attrs->stride_h = attrs->stride[0];
    conv_attrs->stride_w = attrs->stride[1];

    // Set padding dimensions
    conv_attrs->padding_h_up = attrs->pad[0];
    conv_attrs->padding_h_down = attrs->pad[2];
    conv_attrs->padding_w_left = attrs->pad[1];
    conv_attrs->padding_w_right = attrs->pad[3];

    // Calculate input dimensions after padding
    conv_attrs->input_h_after_padding = conv_attrs->input_h + conv_attrs->padding_h_up + conv_attrs->padding_h_down;
    conv_attrs->input_w_after_padding = conv_attrs->input_w + conv_attrs->padding_w_left + conv_attrs->padding_w_right;

    // Set other parameters
    conv_attrs->is_bias = 0;
    conv_attrs->activation_type = NO_ACTIVE;
    conv_attrs->pooling_type = PoolMethod_MAX;
}

/**
 * @brief Perform max pooling operation on quantized integer tensors
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @param attrs Pooling attributes
 * @return int32_t Operation status
 */
int32_t maxpool_luna(const tTensor* X, tTensor* Y, tTensor* Temp, PoolAttrs *attrs) {
    int32_t ret = -1;

    s_conv_struct pool_struct_;
    luna_maxpool_para_init(attrs, &pool_struct_, (tTensor *)X, Y);

    int32_t batch = X->shape_.dims_[0];
    int32_t in_c = pool_struct_.input_c;
    int32_t in_h = pool_struct_.input_h;
    int32_t in_w = pool_struct_.input_w;
    int32_t ou_c = pool_struct_.output_c;
    int32_t ou_h = pool_struct_.output_h;
    int32_t ou_w = pool_struct_.output_w;
    int32_t k_h = pool_struct_.weight_h;
    int32_t k_w = pool_struct_.weight_w;
    int32_t s_h = pool_struct_.stride_h;
    int32_t padding_hd = pool_struct_.padding_h_down;
    int32_t padding_hu = pool_struct_.padding_h_up;
    int32_t log2n_stride_w = (pool_struct_.stride_w >> 1);

    // Calculate input condition for splitting
    int32_t input_condition = (luna_quant_ceil(in_c, 3) << 3) * in_h * (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
    input_condition = (input_condition <= 64 * 1024) ? 1 : 0;

    if (Int8 == X->dtype_) {
        if (input_condition) {  // No need to split input
            int32_t in_batch_size = in_c * in_h * in_w;
            int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
            for (int32_t n = 0; n < batch; n++) {
                int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
                int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
                ret = API_LIB(max_pooling)(p_in, p_out, &pool_struct_);
            }
        } else {  // Split input along height dimension
            int32_t input_limit_without_h = (luna_quant_ceil(in_c, 3) << 3) * (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
            int32_t split_num = 1;
            int32_t tmp_in_h = in_h;

            // Calculate optimal split number
            while ((tmp_in_h * input_limit_without_h > 65536) || ((ou_h % split_num) != 0)) {
                split_num += 1;
                tmp_in_h = (ou_h * s_h) / split_num + k_h - s_h;
                if ((split_num > in_h) || (split_num > ou_h)) {
                    break;
                }
            }

            int32_t cal_var0 = 0;
            while (padding_hd) {
                cal_var0 = in_h + padding_hu + padding_hd - k_h + s_h;
                if (cal_var0 % s_h) {
                    padding_hd = padding_hd - 1;
                } else {
                    break;
                }
            }

            int32_t tmp_ou_h = ou_h / split_num;
            int32_t in_h_1st = tmp_in_h - padding_hu;
            int32_t in_h_last = tmp_in_h - padding_hd;
            int32_t pad_h_down_1st = 0;
            int32_t pad_h_down_last = 0;
            int32_t pad_h_down_mid = 0;
            int32_t in_addr_offset_1st = in_w * (tmp_in_h - k_h + s_h - padding_hu);
            int32_t in_addr_offset = in_w * (tmp_in_h - k_h + s_h);
            int32_t ou_addr_offset = ou_c * ou_w * tmp_ou_h * (0xF & Y->dtype_);

            // Calculate padding for different splits
            if ((in_h_1st + padding_hu) < ((tmp_ou_h - 1) * s_h + k_h)) {
                pad_h_down_1st = (tmp_ou_h - 1) * s_h + k_h - in_h_1st - padding_hu;
            }
            if (in_h_last < ((tmp_ou_h - 1) * s_h + k_h)) {
                pad_h_down_last = (tmp_ou_h - 1) * s_h + k_h - in_h_last;
            }
            if (tmp_in_h < ((tmp_ou_h - 1) * s_h + k_h)) {
                pad_h_down_mid = (tmp_ou_h - 1) * s_h + k_h - tmp_in_h;
            }

            int32_t in_batch_size = in_c * in_h * in_w;
            int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
            int32_t i, j, n;

            for (n = 0; n < batch; n++) {
                int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
                int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
                int8_t *p_in_tmp = p_in;
                int8_t *p_out_tmp = p_out;
                int8_t *p_tmp = (int8_t *)Temp->dptr_;

                for (i = 0; i < split_num; i++) {
                    if (i == 0) {
                        pool_struct_.input_h = in_h_1st;
                        pool_struct_.padding_h_up = padding_hu;
                        pool_struct_.padding_h_down = pad_h_down_1st;
                        p_in_tmp = p_in;
                    } else if (i == (split_num - 1)) {
                        pool_struct_.input_h = in_h_last;
                        pool_struct_.padding_h_up = 0;
                        pool_struct_.padding_h_down = pad_h_down_last;
                        p_in_tmp = p_in + in_addr_offset_1st + (i - 1) * in_addr_offset;
                    } else {
                        pool_struct_.input_h = tmp_in_h;
                        pool_struct_.padding_h_up = 0;
                        pool_struct_.padding_h_down = pad_h_down_mid;
                        p_in_tmp = p_in + in_addr_offset_1st + (i - 1) * in_addr_offset;
                    }

                    pool_struct_.input_h_after_padding = pool_struct_.input_h + pool_struct_.padding_h_up + pool_struct_.padding_h_down;
                    pool_struct_.output_h = tmp_ou_h;
                    p_out_tmp = p_out + i * ou_addr_offset;

                    int32_t c;
                    int32_t o_offset = in_w * pool_struct_.input_h;
                    int32_t i_offset = in_w * in_h;

                    // Copy input data to temporary buffer
                    for (c = 0; c < in_c; c++) {
                        memcpy(p_tmp + c * o_offset, p_in_tmp + c * i_offset, o_offset);
                    }

                    // Perform max pooling on temporary buffer
                    ret = API_LIB(max_pooling)(p_tmp, p_out_tmp, &pool_struct_);
                }

                int32_t one_channel_ou_offset = ou_w * tmp_ou_h * (0xF & Y->dtype_);
                for (j = 0; j < ou_c; j++) {
                    for (i = 0; i < split_num; i++) {
                        int32_t i_offset = i * ou_addr_offset + j * one_channel_ou_offset;
                        int32_t o_offset = i * one_channel_ou_offset + j * ou_w * ou_h;
                        memcpy(p_tmp + o_offset, p_out + i_offset, one_channel_ou_offset);
                    }
                }
                memcpy(p_out, p_tmp, ou_c * ou_h * ou_w);
            }
        }
    }

    return ret;
}

#endif