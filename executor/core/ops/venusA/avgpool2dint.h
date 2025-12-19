#ifndef _AVGPOOL2DINT_VENUS_H_
#define _AVGPOOL2DINT_VENUS_H_

#include <math.h>
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_cnn_tools.h"
#define API_LIB(api) luna_##api
#endif
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "c_api/thinker_define.h"

/**
 * @brief Quantized ceiling function
 * @param x Input value
 * @param shift Number of bits to shift
 * @return int32_t Quantized ceiling value
 */
static int32_t luna_quant_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Calculate base-2 logarithm of a float
 * @param x Input value
 * @return int32_t Base-2 logarithm
 */
static int32_t my_log2(float x) {
    char *in_addr = (char *)&x;
    uint32_t ix = (uint32_t)(*((uint32_t *)in_addr));
    uint32_t exp = (ix >> 23) & 0xFF;
    return (int32_t)(exp - 127);
}

/**
 * @brief Initialize parameters for mean pooling
 * @param attrs Pooling attributes
 * @param conv_attrs Convolution structure to be initialized
 * @param X Input tensor
 * @param Y Output tensor
 */
static void luna_meanpool_para_init(PoolAttrs* attrs, conv_struct_t *conv_attrs, tTensor *X, tTensor *Y) {
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
    int32_t q_x = (int32_t)X->scale_;
    int32_t q_y = (int32_t)Y->scale_;
    conv_attrs->activation_type = NO_ACTIVE;
    conv_attrs->positive_shift_type = ShiftType_FloorX05;
    conv_attrs->positive_shift_value = q_x - q_y;
    conv_attrs->negative_shift_type = ShiftType_FloorX05;
    conv_attrs->negative_shift_value = conv_attrs->positive_shift_value;

    uint8_t data_mem_type = (X->mem_.type_ & 0x0F) + 1;
    data_mem_type = (data_mem_type == 3) ? 0 : data_mem_type;
    conv_attrs->data_mem_type = (data_mem_type << 4) | 0;
    conv_attrs->ou_bits = Y->byte_ * 8;
    conv_attrs->out_padding_h = 0;
    conv_attrs->out_padding_w = 0;
    conv_attrs->group = 1;
    conv_attrs->is_bias = 0;
}

/**
 * @brief Perform 2D average pooling on integer data
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @param attrs Pooling attributes
 * @return int32_t Operation status
 */
int32_t avgpool2dint_luna(const tTensor* X, tTensor* Y, tTensor* Temp, PoolAttrs *attrs) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    if (Y->dtype_ != Int8 && Y->dtype_ != Int32) {
        return T_ERR_INVALID_DATATYPE;
    }

    conv_struct_t pool_struct_;
    luna_cnn_static_para_t pool_static_para;
    luna_meanpool_para_init(attrs, &pool_struct_, (tTensor *)X, Y);

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
    int32_t in_batch_size = in_c * in_h * in_w;
    int32_t in_channel_size = in_h * in_w;
    int32_t ou_channel_size = ou_h * ou_w;
    int32_t input_h_after_padding = in_h + pool_struct_.padding_h_down + pool_struct_.padding_h_up;
    int32_t input_w_after_padding = in_w + pool_struct_.padding_w_left + pool_struct_.padding_w_right;
    int32_t log2n_stride_w = (pool_struct_.stride_w >> 1);
    int32_t input_condition = (luna_quant_ceil(in_c, 3) << 3) * in_h * (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
    input_condition = (input_condition <= 64 * 1024) ? 1 : 0;

    int32_t shift = 0;
    int32_t one_kernel_size = k_h * k_w;

    {
        int32_t split_num = input_condition ? 1 : floor(in_c / 8);
        int32_t s_num = input_condition ? 0 : (in_c - split_num * 8);
        int32_t in_c_split = input_condition ? in_c : 8;

        if ((one_kernel_size & (one_kernel_size - 1)) == 0) { // kernel_size is power of 2
            int32_t *p_tmp = (int32_t *)Temp->dptr_;
            shift = my_log2((float)one_kernel_size);
            if ((input_h_after_padding == k_h) && (input_w_after_padding == k_w)) { // kernel_size == input_size
                int8_t *p0 = (int8_t *)Temp->dptr_;
                int32_t *p1 = (int32_t *)(p0 + in_h * in_w);
                int8_t *p_in = (int8_t *)X->dptr_;

                ret = API_LIB(memset_i8o8)(p0, 1, in_h * in_w);
                ret |= API_LIB(split_mat_mul_i8i8o32)(p_in, p0, p1, in_c, in_h * in_w, 1, 0);

                if (Y->dtype_ == Int8) {
                    int8_t *p_out = (int8_t *)Y->dptr_;
                    ret |= API_LIB(scale_i32i32o8)((int32_t *)p1, 1, p_out, in_c * ou_channel_size, shift);
                } else {
                    int32_t *p_out = (int32_t *)Y->dptr_;
                    ret |= API_LIB(scale_i32i32o32)((int32_t *)p1, 1, p_out, in_c * ou_channel_size, shift);
                }
            } else {
                if (in_c_split * ou_channel_size * Y->byte_ > Temp->shape_.dims_[0]) {
                    return T_ERR_NO_WORKSPACE;
                }

                for (int32_t n = 0; n < split_num; n++) {
                    int8_t *p_in = (int8_t *)X->dptr_ + n * in_channel_size * in_c_split;
                    int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_channel_size * in_c_split;
                    pool_struct_.input_c = in_c_split;
                    ret = luna_split_conv_para_pack(&pool_struct_, &pool_static_para, LUNA_MEAN_POOLING);
                    ret |= API_LIB(mean_pooling2d_i8o32)(p_in, (int32_t *)p_tmp, &pool_static_para);
                    ret |= API_LIB(scale_i32i32o8)(p_tmp, 1, p_out, in_c_split * ou_channel_size, shift);
                }

                if (0 != s_num) {
                    int8_t *p_in = (int8_t *)X->dptr_ + in_channel_size * (split_num - 1) * in_c_split;
                    int8_t *p_out = (int8_t *)Y->dptr_ + ou_channel_size * (split_num - 1) * in_c_split;
                    pool_struct_.input_c = s_num;
                    ret = luna_split_conv_para_pack(&pool_struct_, &pool_static_para, LUNA_MEAN_POOLING);
                    ret |= API_LIB(mean_pooling2d_i8o32)(p_in, (int32_t *)p_tmp, &pool_static_para);
                    ret |= API_LIB(scale_i32i32o8)(p_tmp, 1, p_out, s_num * ou_channel_size, shift);
                }
            }
        } else {
            int32_t q_x = (int32_t)X->scale_;
            int32_t q_o = (int32_t)Y->scale_;

            if ((input_h_after_padding == k_h) && (input_w_after_padding == k_w)) {
                int32_t *p_tmp1 = (int32_t *)Temp->dptr_;
                int32_t *p_tmp2 = (int32_t *)(p_tmp1 + in_c * ou_channel_size);
                int8_t *p_in = (int8_t *)X->dptr_;

                ret = API_LIB(memset_i8o8)((int8_t *)p_tmp1, 1, in_h * in_w);
                ret |= API_LIB(split_mat_mul_i8i8o32)(p_in, (int8_t *)p_tmp1, (int32_t *)p_tmp2, in_c, in_h * in_w, 1, 0);
                ret |= API_LIB(memset_i32o32)(p_tmp1, one_kernel_size, in_c * ou_channel_size);
                ret |= API_LIB(div_i32i32o32)(p_tmp2, p_tmp1, p_tmp1, in_c * ou_channel_size, q_o - q_x);

                if (Y->dtype_ == Int8) {
                    int8_t *p_out = (int8_t *)Y->dptr_;
                    ret |= API_LIB(scale_i32i32o8)(p_tmp1, 1, p_out, in_c * ou_channel_size, 0);
                } else {
                    int32_t *p_out = (int32_t *)Y->dptr_;
                    ret |= API_LIB(scale_i32i32o32)(p_tmp1, 1, p_out, in_c * ou_channel_size, 0);
                }
            } else {
                if ((in_c_split * ou_channel_size + one_kernel_size) * Y->byte_ > Temp->shape_.dims_[0]) {
                    return T_ERR_NO_WORKSPACE;
                }

                int32_t *p_tmp1 = (int32_t *)Temp->dptr_;
                int32_t *p_tmp2 = (int32_t *)(p_tmp1 + in_c_split * ou_channel_size);

                for (int32_t n = 0; n < split_num; n++) {
                    int8_t *p_in = (int8_t *)X->dptr_ + n * in_channel_size * in_c_split;
                    int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_channel_size * in_c_split;
                    pool_struct_.input_c = in_c_split;
                    ret = luna_split_conv_para_pack(&pool_struct_, &pool_static_para, LUNA_MEAN_POOLING);
                    ret |= API_LIB(mean_pooling2d_i8o32)(p_in, (int32_t *)p_tmp2, &pool_static_para);
                    ret |= API_LIB(memset_i32o32)(p_tmp1, one_kernel_size, in_c_split * ou_channel_size);
                    ret |= API_LIB(div_i32i32o32)(p_tmp2, p_tmp1, p_tmp1, in_c_split * ou_channel_size, q_o - q_x);
                    ret |= API_LIB(scale_i32i32o8)(p_tmp1, 1, p_out, in_c_split * ou_channel_size, 0);
                }

                if (0 != s_num) {
                    int8_t *p_in = (int8_t *)X->dptr_ + in_c_split * in_channel_size * (split_num - 1);
                    int8_t *p_out = (int8_t *)Y->dptr_ + in_c_split * ou_channel_size * (split_num - 1);
                    pool_struct_.input_c = s_num;
                    ret = luna_split_conv_para_pack(&pool_struct_, &pool_static_para, LUNA_MEAN_POOLING);
                    ret |= API_LIB(mean_pooling2d_i8o32)(p_in, (int32_t *)p_tmp2, &pool_static_para);
                    ret |= API_LIB(memset_i32o32)(p_tmp1, one_kernel_size, s_num * ou_channel_size);
                    ret |= API_LIB(div_i32i32o32)(p_tmp2, p_tmp1, p_tmp1, s_num * ou_channel_size, q_o - q_x);
                    ret |= API_LIB(scale_i32i32o8)(p_tmp1, 1, p_out, s_num * ou_channel_size, 0);
                }
            }
        }
    }

    return ret;
}

#endif  //_AVGPOOL2DINT_VENUS_H_