#ifndef _MAXPOOL_LUNA_H_
#define _MAXPOOL_LUNA_H_

#include <stdio.h>
#include <string.h>

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_cnn_tools.h"
#define API_LIB(api) luna_##api
#endif

#include "core/operator_attrs.h"
#include "c_api/thinker_define.h"

/**
 * @brief Ceiling division operation for quantized values
 * @param x Dividend
 * @param shift Right shift amount
 * @return Result of ceiling division
 */
static int32_t luna_quant_ceil(int32_t x, int32_t shift)
{
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    }
    else {
        return (x >> shift);
    }
}

/**
 * @brief Initialize pooling parameters for convolution structure
 * @param attrs Pooling attributes
 * @param conv_attrs Convolution structure to initialize
 * @param X Input tensor
 * @param Y Output tensor
 */
static void luna_maxpool_para_init(PoolAttrs* attrs, conv_struct_t *conv_attrs, tTensor *X, tTensor *Y)
{
    memset(conv_attrs, 0, sizeof(conv_struct_t));

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
    conv_attrs->dilation_h = 1;
    conv_attrs->dilation_w = 1;
    conv_attrs->data_mem_type = X->mem_.type_;
    conv_attrs->ou_bits = Y->byte_ * 8;
    conv_attrs->weight_bits = 8;
    conv_attrs->out_padding_h = 0;
    conv_attrs->out_padding_w = 0;
    conv_attrs->group = 1;
    conv_attrs->positive_shift_type = ShiftType_FloorX05;
    conv_attrs->positive_shift_value = 0;
    conv_attrs->negative_shift_type = ShiftType_FloorX05;
    conv_attrs->negative_shift_value = 0;
    conv_attrs->activation_type = NO_ACTIVE;
    conv_attrs->is_bias = 0;
}

/**
 * @brief Max pooling operation for integer tensors
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace buffer
 * @param attrs Pooling attributes
 * @return Operation result status
 */
int32_t maxpool_luna(const tTensor* X, tTensor* Y, tTensor* Temp, PoolAttrs *attrs)
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    conv_struct_t pool_struct_;
    luna_cnn_static_para_t conv_static_para;
    luna_maxpool_para_init(attrs, &pool_struct_, (tTensor *)X, Y);
    ret = luna_split_conv_para_pack(&pool_struct_, &conv_static_para, LUNA_MAX_POOLING);
    if (ret != T_SUCCESS)
        return ret;

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

    int32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;

    if (Int8 == X->dtype_)
    {
        // Case 1: Both input and output in fast memory
        if ((2 == X->mem_.type_) & (2 == Y->mem_.type_)) {
            int32_t in_batch_size = in_c * in_h * in_w;
            int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
            for (int32_t n = 0; n < batch; n++)
            {
                int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
                int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
                ret = API_LIB(max_pooling2d_i8o8)(p_in, p_out, &conv_static_para);
            }
        }
        // Case 2: Output in fast memory, workspace available for input copy
        else if ((2 == Y->mem_.type_) & (workspace_size >= in_c * in_h * in_w)) {
            int8_t *p_in = (int8_t *)Temp->dptr_;
            int8_t *p_out = (int8_t *)Y->dptr_;
            ret = API_LIB(memcpy_i8o8)(p_in, (int8_t *)X->dptr_, in_c * in_h * in_w);
            ret = API_LIB(max_pooling2d_i8o8)(p_in, p_out, &conv_static_para);
        }
        // Case 3: Split input height processing
        else if (2 == Y->mem_.type_) { //split input H
            /////only support H
            int32_t input_limit_without_h = (luna_quant_ceil(in_c, 3) << 3) * (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
            int32_t split_num = 1;
            int32_t tmp_in_h = in_h;
            while ((tmp_in_h * input_limit_without_h > workspace_size) || ((ou_h % split_num) != 0))
            {
                split_num += 1;
                tmp_in_h = (ou_h * s_h) / split_num + k_h - s_h;
                if ((split_num > in_h) || (split_num > ou_h))
                {
                    break;
                }
            }
            
            // Adjust padding values
            int32_t cal_var0 = 0;
            while (padding_hd)
            {
                cal_var0 = in_h + padding_hu + padding_hd - k_h + s_h;
                if (cal_var0 % s_h)
                {
                    padding_hd = padding_hd - 1;
                }
                else
                {
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
            
            // Calculate padding adjustments
            if ((in_h_1st + padding_hu) < ((tmp_ou_h - 1) * s_h + k_h)) {
                pad_h_down_1st = (tmp_ou_h - 1) * s_h + k_h - in_h_1st - padding_hu;
            }
            if (in_h_last < ((tmp_ou_h - 1) * s_h + k_h)) {
                pad_h_down_last = (tmp_ou_h - 1) * s_h + k_h - in_h_last;
            }
            if (tmp_in_h < ((tmp_ou_h - 1) * s_h + k_h)) {
                pad_h_down_mid = (tmp_ou_h - 1) * s_h + k_h - tmp_in_h;
            }
            
            // Process splits
            int32_t in_batch_size = in_c * in_h * in_w;
            int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
            int32_t i, j, n;
            for (n = 0; n < batch; n++)
            {
                int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
                int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
                int8_t *p_in_tmp = p_in;
                int8_t *p_out_tmp = p_out;
                int8_t *p_tmp = (int8_t *)Temp->dptr_;
                
                for (i = 0; i < split_num; i++) {
                    if (i == 0) {
                        pool_struct_.input_h = in_h_1st ;
                        pool_struct_.padding_h_up = padding_hu;
                        pool_struct_.padding_h_down = pad_h_down_1st;
                        p_in_tmp = p_in;
                    }
                    else if (i == (split_num - 1)) {
                        pool_struct_.input_h = in_h_last;
                        pool_struct_.padding_h_up = 0;
                        pool_struct_.padding_h_down = pad_h_down_last;
                        p_in_tmp = p_in + in_addr_offset_1st + (i - 1) * in_addr_offset;
                    }
                    else {
                        pool_struct_.input_h = tmp_in_h;
                        pool_struct_.padding_h_up = 0;
                        pool_struct_.padding_h_down = pad_h_down_mid;
                        p_in_tmp = p_in + in_addr_offset_1st + (i - 1) * in_addr_offset;
                    }
                    pool_struct_.output_h = tmp_ou_h;
                    p_out_tmp = p_out + i * ou_addr_offset;

                    int32_t c;
                    int32_t o_offset = in_w * pool_struct_.input_h;
                    int32_t i_offset = in_w * in_h;
                    for (c = 0; c < in_c; c++)
                    {
                        ret = API_LIB(memcpy_i8o8)(p_tmp + c * o_offset, p_in_tmp + c * i_offset, o_offset);
                    }
                    ret = luna_split_conv_para_pack(&pool_struct_, &conv_static_para, LUNA_MAX_POOLING);
                    ret = API_LIB(max_pooling2d_i8o8)(p_tmp, p_out_tmp, &conv_static_para);
                }

                // Finalize results
                int32_t one_channel_ou_offset = ou_w * tmp_ou_h * (0xF & Y->dtype_);
                for (j = 0; j < ou_c; j++)
                {
                    for (i = 0; i < split_num; i++)
                    {
                        int32_t i_offset = i * ou_addr_offset + j * one_channel_ou_offset;
                        int32_t o_offset = i * one_channel_ou_offset + j * ou_w * ou_h;
                        ret = API_LIB(memcpy_i8o8)(p_tmp + o_offset, p_out + i_offset, one_channel_ou_offset);
                    }
                }
                ret = API_LIB(memcpy_i8o8)(p_out, p_tmp, ou_c * ou_h * ou_w);
            }
        }
        else {
            return T_ERR_INVALID_DATATYPE;
        }    
    }

    return ret;
}

#endif