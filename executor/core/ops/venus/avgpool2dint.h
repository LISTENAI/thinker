#ifndef _AVGPOOL2DINT_VENUS_H_
#define _AVGPOOL2DINT_VENUS_H_

#include <math.h>
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "c_api/thinker_define.h"

/**
 * @brief Calculate the ceiling of a division using bit shifting
 * @param x Numerator
 * @param shift Bit shift amount (equivalent to dividing by 2^shift)
 * @return Ceiling of x divided by 2^shift
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
 * @brief Calculate the base-2 logarithm of a float
 * @param x Input float value
 * @return Base-2 logarithm of x as an integer
 */
static int32_t my_log2(float x)
{
    char *in_addr = (char *)&x;
    uint32_t ix = (uint32_t)(*((uint32_t *)in_addr));
    uint32_t exp = (ix >> 23) & 0xFF;
    return (int32_t)(exp - 127);
}

/**
 * @brief Initialize parameters for average pooling operation
 * @param attrs Pooling attributes including kernel size, stride, padding, etc.
 * @param conv_attrs Pointer to the structure storing convolution/pooling parameters
 * @param X Input tensor
 * @param Y Output tensor
 */
static void luna_meanpool_para_init(PoolAttrs* attrs, s_conv_struct *conv_attrs, tTensor *X, tTensor *Y)
{
    memset(conv_attrs, 0, sizeof(s_conv_struct));

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
    conv_attrs->input_h_after_padding = conv_attrs->input_h + conv_attrs->padding_h_up + conv_attrs->padding_h_down;
    conv_attrs->input_w_after_padding = conv_attrs->input_w + conv_attrs->padding_w_left + conv_attrs->padding_w_right;
    conv_attrs->is_bias = 0;
    conv_attrs->activation_type = NO_ACTIVE;
    conv_attrs->pooling_type = PoolMethod_AVE;
}

/**
 * @brief Perform 2D average pooling on quantized integer tensors
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace buffer
 * @param attrs Pooling attributes including kernel size, stride, padding, etc.
 * @return Operation status
 */
int32_t avgpool2dint_luna(const tTensor* X, tTensor* Y, tTensor* Temp, PoolAttrs *attrs)
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    if (Int8 == X->dtype_) {
        s_conv_struct pool_struct_;
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
        int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
        int32_t in_channel_size = in_h * in_w;
        int32_t ou_channel_size = ou_h * ou_w * (Y->dtype_ & 0xF);
        int32_t log2n_stride_w = (pool_struct_.stride_w >> 1);
        int32_t input_condition = (luna_quant_ceil(in_c, 3) << 3) * in_h * (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
        input_condition = (input_condition <= 64 * 1024) ? 1 : 0;

        int32_t shift = 0;
        int32_t one_kernel_size = k_h * k_w;

        if (input_condition) { // No need to split
            if (0 == (one_kernel_size & (one_kernel_size - 1))) {
                int16_t *p_tmp = (int16_t *)Temp->dptr_;
                shift = my_log2((float)one_kernel_size);
                if ((pool_struct_.input_h_after_padding == k_h) && (pool_struct_.input_w_after_padding == k_w)) {
                    for (int32_t n = 0; n < batch; n++) {
                        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
                        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
                        ret = API_LIB(memset)(p_tmp, 1, in_h * in_w);
                        ret = API_LIB(mat_mul_q7_int16)(p_in, (int8_t *)p_tmp, (int16_t *)p_tmp, in_c, in_h * in_w, 1, 0);
                        ret |= API_LIB(scale_q15_int8)((int16_t *)p_tmp, 1, p_out, ou_batch_size, shift);
                    }
                } else {
                    for (int32_t n = 0; n < batch; n++) {
                        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
                        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
                        ret = API_LIB(mean_pooling_int16)(p_in, (int16_t *)p_tmp, &pool_struct_);
                        ret |= API_LIB(scale_q15_int8)(p_tmp, 1, p_out, ou_batch_size, shift);
                    }
                }
            } else {
                int32_t q_x = (int32_t)X->scale_;
                int32_t q_o = (int32_t)Y->scale_;
                if ((pool_struct_.input_h_after_padding == k_h) && (pool_struct_.input_w_after_padding == k_w)) {
                    int32_t *p_tmp1 = (int32_t *)Temp->dptr_;
                    int32_t *p_tmp2 = (int32_t *)((int8_t *)p_tmp1 + MAX(in_c * in_h * in_w, ou_batch_size * 4));
                    for (int32_t n = 0; n < batch; n++) {
                        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
                        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
                        API_LIB(memset)(p_tmp1, 1, in_h * in_w);
                        ret = API_LIB(mat_mul_q7_int32)(p_in, (int8_t *)p_tmp1, (int32_t *)p_tmp2, pool_struct_.input_c, in_h * in_w, 1, 0);
                        ret |= API_LIB(memset)(p_out, 1, ou_batch_size);
                        ret |= API_LIB(scale_q7_int32)(p_out, one_kernel_size, (int32_t *)p_tmp1, ou_batch_size, 0);
                        ret |= API_LIB(div_q31_int32)(p_tmp2, q_x, p_tmp1, 0, p_tmp1, q_o, ou_batch_size);
                        ret |= API_LIB(scale_q31_int8)(p_tmp1, 1, p_out, ou_batch_size, 0);
                    }
                } else {
                    int32_t *p_tmp1 = (int32_t *)Temp->dptr_;
                    int32_t *p_tmp2 = (int32_t *)(p_tmp1 + ou_batch_size);
                    for (int32_t n = 0; n < batch; n++) {
                        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
                        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
                        ret = API_LIB(mean_pooling_int16)(p_in, (int16_t *)p_tmp1, &pool_struct_);
                        ret |= API_LIB(scale_q15_int32)((int16_t *)p_tmp1, 1, (int32_t *)p_tmp2, ou_batch_size, 0);
                        ret |= API_LIB(memset)(p_out, 1, ou_batch_size);
                        ret |= API_LIB(scale_q7_int32)(p_out, one_kernel_size, (int32_t *)p_tmp1, ou_batch_size, 0);
                        ret |= API_LIB(div_q31_int32)(p_tmp2, q_x, p_tmp1, 0, p_tmp1, q_o, ou_batch_size);
                        ret |= API_LIB(scale_q31_int8)(p_tmp1, 1, p_out, ou_batch_size, 0);
                    }
                }
            }
        } else {
            int32_t split_num = floor(in_c / 8);
            int32_t s_num = in_c - split_num * 8;
            int32_t n = 0;
            if (0 == (one_kernel_size & (one_kernel_size - 1))) {
                int16_t *p_tmp = (int16_t *)Temp->dptr_;
                shift = my_log2((float)one_kernel_size);
                if ((pool_struct_.input_h_after_padding == k_h) && (pool_struct_.input_w_after_padding == k_w)) {
                    for (n = 0; n < split_num; n++) {
                        int8_t *p_in = (int8_t *)X->dptr_ + n * in_channel_size * 8;
                        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_channel_size * 8;
                        pool_struct_.input_c = 8;
                        API_LIB(memset)(p_tmp, 1, in_h * in_w);
                        ret = API_LIB(mat_mul_q7_int32)(p_in, (int8_t *)p_tmp, (int32_t *)p_tmp, 8, in_h * in_w, 1, 0);
                        ret |= API_LIB(scale_q31_int8)((int32_t *)p_tmp, 1, p_out, 8 * ou_channel_size, shift);
                    }
                    if (0 != s_num) {
                        int8_t *p_in = (int8_t *)X->dptr_ + in_channel_size * (split_num - 1) * 8;
                        int8_t *p_out = (int8_t *)Y->dptr_ + ou_channel_size * (split_num - 1) * 8;
                        pool_struct_.input_c = s_num;
                        API_LIB(memset)(p_tmp, 1, in_h * in_w);
                        ret = API_LIB(mat_mul_q7_int32)(p_in, (int8_t *)p_tmp, (int32_t *)p_tmp, s_num, in_h * in_w, 1, 0);
                        ret |= API_LIB(scale_q31_int8)((int32_t *)p_tmp, 1, p_out, s_num * ou_channel_size, shift);
                    }
                } else {
                    for (n = 0; n < split_num; n++) {
                        int8_t *p_in = (int8_t *)X->dptr_ + n * in_channel_size * 8;
                        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_channel_size * 8;
                        pool_struct_.input_c = 8;
                        ret = API_LIB(mean_pooling_int16)(p_in, (int16_t *)p_tmp, &pool_struct_);
                        ret |= API_LIB(scale_q15_int8)(p_tmp, 1, p_out, 8 * ou_channel_size, shift);
                    }
                    if (0 != s_num) {
                        int8_t *p_in = (int8_t *)X->dptr_ + in_channel_size * (split_num - 1) * 8;
                        int8_t *p_out = (int8_t *)Y->dptr_ + ou_channel_size * (split_num - 1) * 8;
                        pool_struct_.input_c = s_num;
                        ret = API_LIB(mean_pooling_int16)(p_in, (int16_t *)p_tmp, &pool_struct_);
                        ret |= API_LIB(scale_q15_int8)(p_tmp, 1, p_out, s_num * ou_channel_size, shift);
                    }
                }
            } else {
                int32_t q_x = (int32_t)X->scale_;
                int32_t q_o = (int32_t)Y->scale_;
                if ((pool_struct_.input_h_after_padding == k_h) && (pool_struct_.input_w_after_padding == k_w)) {
                    int32_t *p_tmp1 = (int32_t *)Temp->dptr_;
                    int32_t *p_tmp2 = (int32_t *)(p_tmp1 + 8 * ou_channel_size);
                    for (n = 0; n < split_num; n++) {
                        int8_t *p_in = (int8_t *)X->dptr_ + n * in_channel_size * 8;
                        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_channel_size * 8;
                        pool_struct_.input_c = 8;
                        API_LIB(memset)(p_tmp1, 1, in_h * in_w);
                        ret = API_LIB(mat_mul_q7_int32)(p_in, (int8_t *)p_tmp1, (int32_t *)p_tmp2, 8, in_h * in_w, 1, 0);
                        ret |= API_LIB(memset)(p_out, 1, 8 * ou_channel_size);
                        ret |= API_LIB(scale_q7_int32)(p_out, one_kernel_size, p_tmp1, 8 * ou_channel_size, 0);
                        ret |= API_LIB(div_q31_int32)(p_tmp2, q_x, p_tmp1, 0, p_tmp1, q_o, 8 * ou_channel_size);
                        ret |= API_LIB(scale_q31_int8)(p_tmp1, 1, p_out, 8 * ou_channel_size, 0);
                    }
                    if (0 != s_num) {
                        int8_t *p_in = (int8_t *)X->dptr_ + 8 * in_channel_size * (split_num - 1);
                        int8_t *p_out = (int8_t *)Y->dptr_ + 8 * ou_channel_size * (split_num - 1);
                        pool_struct_.input_c = s_num;
                        API_LIB(memset)(p_tmp1, 1, in_h * in_w);
                        ret = API_LIB(mat_mul_q7_int32)(p_in, (int8_t *)p_tmp1, (int32_t *)p_tmp2, s_num, in_h * in_w, 1, 0);
                        ret |= API_LIB(memset)(p_out, 1, s_num * ou_channel_size);
                        ret |= API_LIB(scale_q7_int32)(p_out, one_kernel_size, p_tmp1, s_num * ou_channel_size, 0);
                        ret |= API_LIB(div_q31_int32)(p_tmp2, q_x, p_tmp1, 0, p_tmp1, q_o, s_num * ou_channel_size);
                        ret |= API_LIB(scale_q31_int8)(p_tmp1, 1, p_out, s_num * ou_channel_size, 0);
                    }
                } else {
                    int32_t *p_tmp1 = (int32_t *)Temp->dptr_;
                    int32_t *p_tmp2 = (int32_t *)(p_tmp1 + 8 * ou_channel_size);
                    for (n = 0; n < split_num; n++) {
                        int8_t *p_in = (int8_t *)X->dptr_ + n * in_channel_size * 8;
                        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_channel_size * 8;
                        pool_struct_.input_c = 8;
                        ret = API_LIB(mean_pooling_int16)(p_in, (int16_t *)p_tmp1, &pool_struct_);
                        ret |= API_LIB(scale_q15_int32)((int16_t *)p_tmp1, 1, p_tmp2, 8 * ou_channel_size, 0);
                        ret |= API_LIB(memset)(p_out, 1, 8 * ou_channel_size);
                        ret |= API_LIB(scale_q7_int32)(p_out, one_kernel_size, p_tmp1, 8 * ou_channel_size, 0);
                        ret |= API_LIB(div_q31_int32)(p_tmp2, q_x, p_tmp1, 0, p_tmp1, q_o, 8 * ou_channel_size);
                        ret |= API_LIB(scale_q31_int8)(p_tmp1, 1, p_out, 8 * ou_channel_size, 0);
                    }
                    if (0 != s_num) {
                        int8_t *p_in = (int8_t *)X->dptr_ + 8 * in_channel_size * (split_num - 1);
                        int8_t *p_out = (int8_t *)Y->dptr_ + 8 * ou_channel_size * (split_num - 1);
                        pool_struct_.input_c = s_num;
                        ret = API_LIB(mean_pooling_int16)(p_in, (int16_t *)p_tmp1, &pool_struct_);
                        ret |= API_LIB(scale_q15_int32)((int16_t *)p_tmp1, 1, p_tmp2, s_num * ou_channel_size, 0);
                        ret |= API_LIB(memset)(p_out, 1, s_num * ou_channel_size);
                        ret |= API_LIB(scale_q7_int32)(p_out, one_kernel_size, p_tmp1, s_num * ou_channel_size, 0);
                        ret |= API_LIB(div_q31_int32)(p_tmp2, q_x, p_tmp1, 0, p_tmp1, q_o, s_num * ou_channel_size);
                        ret |= API_LIB(scale_q31_int8)(p_tmp1, 1, p_out, s_num * ou_channel_size, 0);
                    }
                }
            }
        }
    }
    return ret;
}

#endif  //_AVGPOOL2DINT_VENUS_H_