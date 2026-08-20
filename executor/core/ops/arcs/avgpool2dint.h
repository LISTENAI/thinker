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

#define ARCS_AVGPOOL2DINT_INPUT_THRESHOLD (16 * 1024)

/**
 * @brief Quantized ceiling function
 * @param x Input value
 * @param shift Shift value for quantization
 * @return int32_t Quantized ceiling result
 */
static int32_t luna_quant_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Log base 2 function for floating-point numbers
 * @param x Input value
 * @return int32_t Log base 2 result
 */
static int32_t my_log2(float x) {
    char *in_addr = (char *)&x;
    uint32_t ix = (uint32_t)(*((uint32_t *)in_addr));
    uint32_t exp = (ix >> 23) & 0xFF;
    return (int32_t)(exp - 127);
}


static int32_t avgpool2dint_calc_channel_split(int32_t channel, int32_t in_h,
                                                int32_t kernel_h,
                                                int32_t input_w_align,
                                                int32_t *split_num,
                                                int32_t *channel_split) {
    int32_t input_condition = (luna_quant_ceil(channel, 3) << 3) * in_h * input_w_align;
    int32_t split_condition = input_condition;

    if (input_condition <= ARCS_AVGPOOL2DINT_INPUT_THRESHOLD) {
        *split_num = 1;
        *channel_split = channel;
        return T_SUCCESS;
    }

    *split_num = (input_condition + ARCS_AVGPOOL2DINT_INPUT_THRESHOLD - 1) /
                 ARCS_AVGPOOL2DINT_INPUT_THRESHOLD;
    do {
        int32_t raw_in_c_split = (channel + *split_num - 1) / *split_num;
        if (raw_in_c_split < 8) {
            *channel_split = 8;
            split_condition = *channel_split * kernel_h * input_w_align;
        } else {
            *channel_split = luna_quant_ceil(raw_in_c_split, 3) << 3;
            split_condition = (luna_quant_ceil(*channel_split, 3) << 3) * in_h * input_w_align;
        }
        if (split_condition <= ARCS_AVGPOOL2DINT_INPUT_THRESHOLD) {
            break;
        }
        (*split_num)++;
    } while (*split_num <= channel);

    #if THINKER_RUNTIME_CHECK
    if (split_condition > ARCS_AVGPOOL2DINT_INPUT_THRESHOLD) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    *split_num = (channel + *channel_split - 1) / *channel_split;
    return T_SUCCESS;
}

static uint8_t avgpool2dint_luna_data_mem_type(const tTensor *tensor) {
    uint8_t data_mem_type = (tensor->mem_.type_ & 0x0F) + 1;
    return (data_mem_type == 3) ? 0 : data_mem_type;
}

static int32_t avgpool2dint_check_workspace(const tTensor *Temp, uint32_t bytes) {
    #if THINKER_RUNTIME_CHECK
    if (Temp == NULL || Temp->dptr_ == 0 ||
                          Temp->shape_.ndim_ != 1 ||
                          bytes > Temp->shape_.dims_[0]) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    return T_SUCCESS;
}

/**
 * @brief Initialize parameters for mean pooling
 * @param attrs Pooling attributes
 * @param conv_attrs Convolution structure for pooling
 * @param X Input tensor
 * @param Y Output tensor
 */
static void luna_meanpool_para_init(PoolAttrs *attrs, conv_struct_t *conv_attrs, tTensor *X, tTensor *Y) {
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
 * @brief Execute 2D average pooling with integer precision
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Workspace tensor
 * @param attrs Pooling attributes
 * @return int32_t Execution status
 */
int32_t avgpool2dint_luna(const tTensor *X, tTensor *Y, tTensor *Temp, PoolAttrs *attrs) {
    #if THINKER_PARAM_CHECK
    if (X->dtype_ != Int8 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (attrs->layout != 0) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->shape_.dims_[0] != 1 || Y->shape_.dims_[0] != 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    conv_struct_t pool_struct_;
    luna_cnn_static_para_t pool_static_para;
    luna_meanpool_para_init(attrs, &pool_struct_, (tTensor *)X, Y);

    int32_t channel = pool_struct_.input_c;
    int32_t in_h = pool_struct_.input_h;
    int32_t in_w = pool_struct_.input_w;
    int32_t ou_h = pool_struct_.output_h;
    int32_t ou_w = pool_struct_.output_w;
    int32_t k_h = pool_struct_.weight_h;
    int32_t k_w = pool_struct_.weight_w;
    int32_t hw_in = in_h * in_w;
    int32_t hw_out = ou_h * ou_w;
    int32_t hw_kernel = k_h * k_w;
    int32_t input_h_after_padding = in_h + pool_struct_.padding_h_down + pool_struct_.padding_h_up;
    int32_t input_w_after_padding = in_w + pool_struct_.padding_w_left + pool_struct_.padding_w_right;
    int32_t log2n_stride_w = (pool_struct_.stride_w >> 1);
    int32_t input_w_align = luna_quant_ceil(in_w, (2 + log2n_stride_w)) << (2 + log2n_stride_w);
    int32_t split_num = 1;
    int32_t channel_split = channel;
    int32_t x_in_psram = (X->mem_.type_ != 2) ? 1 : 0;
    int32_t y_in_psram = (Y->mem_.type_ != 2) ? 1 : 0;
    int32_t global_pool = input_h_after_padding == k_h && input_w_after_padding == k_w;

    #if THINKER_PARAM_CHECK
    if (k_h < 1 || k_w < 1 ||
                        (!global_pool && (k_h > 7 || k_w > 7 ||
                         (pool_struct_.stride_h != 1 && pool_struct_.stride_h != 2 && pool_struct_.stride_h != 4) ||
                         (pool_struct_.stride_w != 1 && pool_struct_.stride_w != 2 && pool_struct_.stride_w != 4))) ||
                        pool_struct_.padding_h_up < 0 || pool_struct_.padding_h_up > 11 ||
                        pool_struct_.padding_h_down < 0 || pool_struct_.padding_h_down > 11 ||
                        pool_struct_.padding_w_left < 0 || pool_struct_.padding_w_left > 11 ||
                        pool_struct_.padding_w_right < 0 || pool_struct_.padding_w_right > 11 ||
                        pool_struct_.padding_h_up >= k_h || pool_struct_.padding_h_down >= k_h ||
                        pool_struct_.padding_w_left >= k_w || pool_struct_.padding_w_right >= k_w ||
                        k_h < pool_struct_.stride_h || k_w < pool_struct_.stride_w ||
                         input_h_after_padding < k_h || input_w_after_padding < k_w) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (X->dptr_ == 0 || Y->dptr_ == 0 || X->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int32_t shift = 0;
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    shift = my_log2((float)hw_kernel);
    #if THINKER_PARAM_CHECK
    if (((hw_kernel & (hw_kernel - 1)) == 0 &&
                         (shift + Y->scale_ - X->scale_ < 0 || shift + Y->scale_ - X->scale_ > 63)) ||
                        ((hw_kernel & (hw_kernel - 1)) != 0 &&
                         (Y->scale_ - X->scale_ < 0 || Y->scale_ - X->scale_ > 63))) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    if (global_pool) { // global averagepooling
        int8_t *p_in = (int8_t *)X->dptr_;
        int32_t *p1  = (int32_t *)workspace;
        int32_t sum_bytes = channel * (int32_t)sizeof(int32_t);
        int8_t *p0 = workspace + sum_bytes;
        int32_t offset = sum_bytes + ALIGN4(in_h * in_w);
        int32_t *p_divisor = NULL;
        int32_t *p_divided = NULL;
        if ((hw_kernel & (hw_kernel - 1)) != 0) {
            p_divisor = (int32_t *)(workspace + offset);
            offset += sum_bytes;
            p_divided = (int32_t *)(workspace + offset);
            offset += sum_bytes;
        }
        int8_t *p_out = y_in_psram ? workspace + offset : (int8_t *)Y->dptr_;
        int32_t required = offset + (y_in_psram ? channel * hw_out * Y->byte_ : 0);
        THINKER_RET_CHECK(avgpool2dint_check_workspace(Temp, required), "avgpool2dint_check_workspace");

        THINKER_RET_CHECK(API_LIB(memset_i8o8)(p0, 1, in_h * in_w), "luna_memset_i8o8");
        THINKER_RET_CHECK(API_LIB(split_mat_mul_i8i8o32)(p_in, p0, p1, channel, in_h * in_w, 1, 0), "luna_split_mat_mul_i8i8o32");
        if ((hw_kernel & (hw_kernel - 1)) != 0) {
            shift = Y->scale_ - X->scale_;
            THINKER_RET_CHECK(API_LIB(memset_i32o32)(p_divisor, hw_kernel, channel), "luna_memset_i32o32");
            THINKER_RET_CHECK(API_LIB(div_i32i32o32)(p1, p_divisor, p_divided, channel, shift), "luna_div_i32i32o32");
            p1 = p_divided;
            shift = 0;
        }
        else
            shift += Y->scale_ - X->scale_;

        if (Y->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(scale_i32i32o8)((int32_t *)p1, 1, p_out, channel * hw_out, shift), "luna_scale_i32i32o8");
        } 
        else {
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)p1, 1, (int32_t *)p_out, channel * hw_out, shift), "luna_scale_i32i32o32");
        }

        if (y_in_psram) {
            opi_psram_cpy_out((int8_t *)Y->dptr_, p_out, channel * hw_out * Y->byte_);
        }
    } 
    else {

        THINKER_RET_CHECK(avgpool2dint_calc_channel_split(channel, in_h, k_h, input_w_align,
                                                        &split_num, &channel_split),
                        "avgpool2dint_calc_channel_split");
        shift = Y->scale_ - X->scale_;
        for (int32_t n = 0; n < split_num; n++) {
            int32_t ch_start = n * channel_split;
            int32_t cur_in_c = MIN(channel_split, channel - ch_start);
            int8_t *p_in = (int8_t *)X->dptr_ + ch_start * hw_in;
            int32_t *p1  = (int32_t *)workspace;
            int32_t sum_bytes = cur_in_c * hw_out * (int32_t)sizeof(int32_t);
            int32_t offset = sum_bytes;
            int8_t *p0 = workspace + offset;
            if (x_in_psram) offset += ALIGN4(cur_in_c * hw_in);
            int32_t *p_divisor = NULL;
            int32_t *p_divided = NULL;
            if ((hw_kernel & (hw_kernel - 1)) != 0) {
                p_divisor = (int32_t *)(workspace + offset);
                offset += sum_bytes;
                p_divided = (int32_t *)(workspace + offset);
                offset += sum_bytes;
            }
            int8_t *p_out = y_in_psram ? workspace + offset :
                (int8_t *)Y->dptr_ + ch_start * hw_out * Y->byte_;
            int32_t required = offset + (y_in_psram ? cur_in_c * hw_out * Y->byte_ : 0);
            THINKER_RET_CHECK(avgpool2dint_check_workspace(Temp, required), "avgpool2dint_check_workspace");

            pool_struct_.input_c = cur_in_c;
            pool_struct_.output_c = cur_in_c;
            pool_struct_.data_mem_type = 0;
            pool_struct_.ou_bits = 32;
            THINKER_RET_CHECK(API_LIB(split_conv_para_pack)(&pool_struct_, &pool_static_para, LUNA_MEAN_POOLING), "luna_split_conv_para_pack");
            if (x_in_psram) {
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(p0, p_in, cur_in_c * hw_in), "luna_memcpy_i8o8");
                p_in = p0;
            }

            THINKER_RET_CHECK(API_LIB(mean_pooling2d_i8o32)(p_in, p1, &pool_static_para), "luna_mean_pooling2d_i8o32");

            int32_t output_shift = shift;
            if ((hw_kernel & (hw_kernel - 1)) != 0) {
                THINKER_RET_CHECK(API_LIB(memset_i32o32)(p_divisor, hw_kernel, cur_in_c * hw_out), "luna_memset_i32o32");
                THINKER_RET_CHECK(API_LIB(div_i32i32o32)(p1, p_divisor, p_divided, cur_in_c * hw_out, shift), "luna_div_i32i32o32");
                p1 = p_divided;
                output_shift = 0;
            }

            if (Y->dtype_ == Int8) {
                THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(p1, 1, p_out, cur_in_c * hw_out, output_shift), "luna_scale_i32i32o8");
            } 
            else {
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p1, 1, (int32_t *)p_out, cur_in_c * hw_out, output_shift), "luna_scale_i32i32o32");
            }
            if (y_in_psram) {
                opi_psram_cpy_out((int8_t *)Y->dptr_ + ch_start * hw_out * Y->byte_,
                                    p_out, cur_in_c * hw_out * Y->byte_);
            }
        }
    }

    return T_SUCCESS;
}

#endif  // _AVGPOOL2DINT_VENUS_H_
