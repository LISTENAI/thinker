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
    conv_attrs->positive_shift_value = 0;
    conv_attrs->negative_shift_type = ShiftType_FloorX05;
    conv_attrs->negative_shift_value = 0;

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
#if THINKER_PARAM_CHECK
if (X->dtype_ != Int8 || Y->dtype_ != Int8) {
    return (T_ERR_INVALID_DATATYPE);
}

if (attrs->layout != 0) {
    return (T_ERR_INVALID_PARA);
}
#endif
    int32_t is_global_pool =
        attrs->kernel[0] == X->shape_.dims_[2] + attrs->pad[0] + attrs->pad[2] &&
        attrs->kernel[1] == X->shape_.dims_[3] + attrs->pad[1] + attrs->pad[3];
#if THINKER_PARAM_CHECK
if (attrs->kernel[0] < 1 || attrs->kernel[1] < 1 ||
                    (!is_global_pool && (attrs->kernel[0] > 7 || attrs->kernel[1] > 7))) {
    return (T_ERR_INVALID_PARA);
}

if (!is_global_pool &&
                    ((attrs->stride[0] != 1 && attrs->stride[0] != 2 && attrs->stride[0] != 4) ||
                     (attrs->stride[1] != 1 && attrs->stride[1] != 2 && attrs->stride[1] != 4) ||
                     attrs->kernel[0] < attrs->stride[0] || attrs->kernel[1] < attrs->stride[1] ||
                     attrs->pad[0] > 11 || attrs->pad[1] > 11 ||
                     attrs->pad[2] > 11 || attrs->pad[3] > 11 ||
                     attrs->pad[0] >= attrs->kernel[0] || attrs->pad[2] >= attrs->kernel[0] ||
                     attrs->pad[1] >= attrs->kernel[1] || attrs->pad[3] >= attrs->kernel[1])) {
    return (T_ERR_INVALID_PARA);
}
#endif

    conv_struct_t pool_struct_;
    luna_cnn_static_para_t pool_static_para;
    luna_meanpool_para_init(attrs, &pool_struct_, (tTensor *)X, Y);

    int32_t batch = X->shape_.dims_[0];
#if THINKER_PARAM_CHECK
if (batch != 1 || Y->shape_.dims_[0] != 1) {
    return (T_ERR_INVALID_PARA);
}
#endif
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
    int32_t h_eff = in_h + (pool_struct_.padding_h_up != 0 ? 1 : 0);
    int32_t input_condition = (luna_quant_ceil(in_c, 2) << 2) * h_eff * (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
    input_condition = (input_condition <= DW_IN_CONDITION) ? 1 : 0;
    #if THINKER_RUNTIME_CHECK
    if (!input_condition &&
        (luna_quant_ceil(MIN(8, in_c), 2) << 2) * h_eff *
            (luna_quant_ceil(in_w, 3 + log2n_stride_w) <<
             (3 + log2n_stride_w)) > DW_IN_CONDITION) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    #if THINKER_RUNTIME_CHECK
    if (X->dptr_ == 0 || Y->dptr_ == 0 || X->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (Temp == NULL || Temp->dptr_ == 0 ||
                          Temp->shape_.ndim_ != 1) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    int32_t workspace_size = Temp->shape_.dims_[0];
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t output_total_bytes = in_c * ou_channel_size * Y->byte_;
    int32_t work_limit = workspace_size;
    int8_t *out_base = (int8_t *)Y->dptr_;
    if (y_in_psram) {
        #if THINKER_RUNTIME_CHECK
        if (workspace_size < output_total_bytes) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        work_limit = (workspace_size - output_total_bytes) & ~3;
        out_base = (int8_t *)Temp->dptr_ + work_limit;
    }

    int32_t shift = 0;
    int32_t one_kernel_size = k_h * k_w;

    {
        int32_t in_c_split = input_condition ? in_c : 8;
        int32_t split_num = input_condition ? 1 : (in_c / in_c_split);
        int32_t s_num = input_condition ? 0 : (in_c - split_num * in_c_split);
        if (!input_condition && split_num == 0) {
            split_num = 1;
            in_c_split = in_c;
            s_num = 0;
        }

        if ((one_kernel_size & (one_kernel_size - 1)) == 0) { // kernel_size is power of 2
            int32_t *p_tmp = (int32_t *)Temp->dptr_;
            shift = my_log2((float)one_kernel_size) + X->scale_ - Y->scale_;
#if THINKER_PARAM_CHECK
if (shift > 63 || shift < -30) {
    return (T_ERR_INVALID_PARA);
}
#endif
            uint32_t shift1 = shift < 0 ? 1UL << -shift : 1;
            uint32_t shift2 = shift < 0 ? 0 : shift;
            if ((input_h_after_padding == k_h) && (input_w_after_padding == k_w)) { // kernel_size == input_size
                int8_t *p0 = (int8_t *)Temp->dptr_;
                int32_t *p1 = (int32_t *)(p0 + ALIGN4(in_h * in_w));
                int8_t *p_in = (int8_t *)X->dptr_;
                #if THINKER_RUNTIME_CHECK
                if (ALIGN4(in_h * in_w) +
                    in_c * ou_channel_size * (int32_t)sizeof(int32_t) > work_limit) {
                    return (T_ERR_NO_WORKSPACE);
                }
#endif

                THINKER_RET_CHECK(API_LIB(memset_i8o8)(p0, 1, in_h * in_w), "luna_memset_i8o8");
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i8i8o32)(p_in, p0, p1, in_c, in_h * in_w, 1, 0), "luna_split_mat_mul_i8i8o32");

                if (Y->dtype_ == Int8) {
                    int8_t *p_out = out_base;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o8)((int32_t *)p1, shift1, p_out, in_c * ou_channel_size, shift2), "luna_scale_i32i32o8");
                } 
                else if (Y->dtype_ == Int16) {
                    int16_t *p_out = (int16_t *)out_base;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o16)((int32_t *)p1, shift1, p_out, in_c * ou_channel_size, shift2), "luna_scale_i32i32o16");
                } 
                else {
                    int32_t *p_out = (int32_t *)out_base;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)p1, shift1, p_out, in_c * ou_channel_size, shift2), "luna_scale_i32i32o32");
                }
            } 
            else {
                #if THINKER_RUNTIME_CHECK
                if (in_c_split * ou_channel_size * (int32_t)sizeof(int32_t) > work_limit) {
                    return (T_ERR_NO_WORKSPACE);
                }
#endif

                for (int32_t n = 0; n < split_num; n++) {
                    int8_t *p_in = (int8_t *)X->dptr_ + n * in_channel_size * in_c_split;
                    int8_t *p_out = out_base + n * ou_channel_size * in_c_split * Y->byte_;
                    pool_struct_.input_c = in_c_split;
                    THINKER_RET_CHECK(luna_split_conv_para_pack(&pool_struct_, &pool_static_para, LUNA_MEAN_POOLING), "luna_split_conv_para_pack");
                    THINKER_RET_CHECK(API_LIB(mean_pooling2d_i8o32)(p_in, (int32_t *)p_tmp, &pool_static_para), "luna_mean_pooling2d_i8o32");
                    if (Y->dtype_ == Int8) {
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(p_tmp, shift1, p_out, in_c_split * ou_channel_size, shift2), "luna_scale_i32i32o8");
                    } else {
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_tmp, shift1, (int32_t *)p_out, in_c_split * ou_channel_size, shift2), "luna_scale_i32i32o32");
                    }
                }

                if (0 != s_num) {
                    int8_t *p_in = (int8_t *)X->dptr_ + in_channel_size * split_num * in_c_split;
                    int8_t *p_out = out_base + ou_channel_size * split_num * in_c_split * Y->byte_;
                    pool_struct_.input_c = s_num;
                    THINKER_RET_CHECK(luna_split_conv_para_pack(&pool_struct_, &pool_static_para, LUNA_MEAN_POOLING), "luna_split_conv_para_pack");
                    THINKER_RET_CHECK(API_LIB(mean_pooling2d_i8o32)(p_in, (int32_t *)p_tmp, &pool_static_para), "luna_mean_pooling2d_i8o32");
                    if (Y->dtype_ == Int8) {
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(p_tmp, shift1, p_out, s_num * ou_channel_size, shift2), "luna_scale_i32i32o8");
                    } else {
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_tmp, shift1, (int32_t *)p_out, s_num * ou_channel_size, shift2), "luna_scale_i32i32o32");
                    }
                }
            }
        } 
        else {
            int32_t q_x = (int32_t)X->scale_;
            int32_t q_o = (int32_t)Y->scale_;
            int32_t shift = q_x - q_o;
            #if THINKER_PARAM_CHECK
            if (shift > 63 || shift < -30) {
                return (T_ERR_INVALID_PARA);
            }
#endif
            uint32_t shift1 = shift < 0 ? 1UL << -shift : 1;
            uint32_t shift2 = shift < 0 ? 0 : shift;
            if ((input_h_after_padding == k_h) && (input_w_after_padding == k_w)) {
                int8_t *p_tmp_bytes = (int8_t *)Temp->dptr_;
                int32_t *p_tmp1 = (int32_t *)p_tmp_bytes;
                int32_t *p_tmp2 = (int32_t *)(p_tmp_bytes + MAX(ALIGN4(in_h * in_w), in_c * ou_channel_size * (int32_t)sizeof(int32_t)));
                int8_t *p_in = (int8_t *)X->dptr_;
                #if THINKER_RUNTIME_CHECK
                if (MAX(ALIGN4(in_h * in_w),
                        in_c * ou_channel_size * (int32_t)sizeof(int32_t)) +
                    in_c * ou_channel_size * (int32_t)sizeof(int32_t) > work_limit) {
                    return (T_ERR_NO_WORKSPACE);
                }
#endif

                THINKER_RET_CHECK(API_LIB(memset_i8o8)((int8_t *)p_tmp1, 1, in_h * in_w), "luna_memset_i8o8");
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i8i8o32)(p_in, (int8_t *)p_tmp1, (int32_t *)p_tmp2, in_c, in_h * in_w, 1, 0), "luna_split_mat_mul_i8i8o32");
                THINKER_RET_CHECK(API_LIB(memset_i32o32)(p_tmp1, one_kernel_size, in_c * ou_channel_size), "luna_memset_i32o32");
                THINKER_RET_CHECK(API_LIB(div_i32i32o32)(p_tmp2, p_tmp1, p_tmp1, in_c * ou_channel_size, 0), "luna_div_i32i32o32");

                if (Y->dtype_ == Int8) {
                    int8_t *p_out = out_base;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(p_tmp1, shift1, p_out, in_c * ou_channel_size, shift2), "luna_scale_i32i32o8");
                }
                else if (Y->dtype_ == Int16) {
                    int16_t *p_out = (int16_t *)out_base;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o16)(p_tmp1, shift1, p_out, in_c * ou_channel_size, shift2), "luna_scale_i32i32o16");
                }
                else {
                    int32_t *p_out = (int32_t *)out_base;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_tmp1, shift1, p_out, in_c * ou_channel_size, shift2), "luna_scale_i32i32o32");
                }
            } 
            else {
                #if THINKER_RUNTIME_CHECK
                if (2 * in_c_split * ou_channel_size * (int32_t)sizeof(int32_t) > work_limit) {
                    return (T_ERR_NO_WORKSPACE);
                }
#endif

                int32_t *p_tmp1 = (int32_t *)Temp->dptr_;
                int32_t *p_tmp2 = (int32_t *)(p_tmp1 + in_c_split * ou_channel_size);

                for (int32_t n = 0; n < split_num; n++) {
                    int8_t *p_in = (int8_t *)X->dptr_ + n * in_channel_size * in_c_split;
                    int8_t *p_out = out_base + n * ou_channel_size * in_c_split * Y->byte_;
                    pool_struct_.input_c = in_c_split;
                    THINKER_RET_CHECK(luna_split_conv_para_pack(&pool_struct_, &pool_static_para, LUNA_MEAN_POOLING), "luna_split_conv_para_pack");
                    THINKER_RET_CHECK(API_LIB(mean_pooling2d_i8o32)(p_in, (int32_t *)p_tmp2, &pool_static_para), "luna_mean_pooling2d_i8o32");
                    THINKER_RET_CHECK(API_LIB(memset_i32o32)(p_tmp1, one_kernel_size, in_c_split * ou_channel_size), "luna_memset_i32o32");
                    THINKER_RET_CHECK(API_LIB(div_i32i32o32)(p_tmp2, p_tmp1, p_tmp1, in_c_split * ou_channel_size, 0), "luna_div_i32i32o32");
                    if (Y->dtype_ == Int8) {
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(p_tmp1, shift1, p_out, in_c_split * ou_channel_size, shift2), "luna_scale_i32i32o8");
                    } else {
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_tmp1, shift1, (int32_t *)p_out, in_c_split * ou_channel_size, shift2), "luna_scale_i32i32o32");
                    }
                }

                if (0 != s_num) {
                    int8_t *p_in = (int8_t *)X->dptr_ + in_c_split * in_channel_size * split_num;
                    int8_t *p_out = out_base + in_c_split * ou_channel_size * split_num * Y->byte_;
                    pool_struct_.input_c = s_num;
                    THINKER_RET_CHECK(luna_split_conv_para_pack(&pool_struct_, &pool_static_para, LUNA_MEAN_POOLING), "luna_split_conv_para_pack");
                    THINKER_RET_CHECK(API_LIB(mean_pooling2d_i8o32)(p_in, (int32_t *)p_tmp2, &pool_static_para), "luna_mean_pooling2d_i8o32");
                    THINKER_RET_CHECK(API_LIB(memset_i32o32)(p_tmp1, one_kernel_size, s_num * ou_channel_size), "luna_memset_i32o32");
                    THINKER_RET_CHECK(API_LIB(div_i32i32o32)(p_tmp2, p_tmp1, p_tmp1, s_num * ou_channel_size, 0), "luna_div_i32i32o32");
                    if (Y->dtype_ == Int8) {
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(p_tmp1, shift1, p_out, s_num * ou_channel_size, shift2), "luna_scale_i32i32o8");
                    } else {
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_tmp1, shift1, (int32_t *)p_out, s_num * ou_channel_size, shift2), "luna_scale_i32i32o32");
                    }
                }
            }
        }
    }

    if (y_in_psram) {
        opi_psram_cpy_out((void *)Y->dptr_, out_base, output_total_bytes);
    }
    return T_SUCCESS;
}

#endif  //_AVGPOOL2DINT_VENUS_H_
