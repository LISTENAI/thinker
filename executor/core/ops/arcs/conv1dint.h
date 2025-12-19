#ifndef _CONV1DINT_ARCS_H_
#define _CONV1DINT_ARCS_H_

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
 * @brief Initialize parameters for 1D convolution
 * @param attrs Convolution attributes
 * @param conv_attrs Convolution structure for 1D convolution
 * @param X Input tensor
 * @param W Weight tensor
 * @param Bias Bias tensor (optional)
 * @param Y Output tensor
 */
static void conv1dint_para_init(Conv1dIntAttrs *attrs, conv_struct_t *conv_attrs, tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y) {
    memset(conv_attrs, 0, sizeof(conv_struct_t));
    conv_attrs->is_bias = (Bias != NULL) ? 1 : 0;

    conv_attrs->input_c = X->shape_.dims_[1];
    conv_attrs->input_h = 1;
    conv_attrs->input_w = X->shape_.dims_[2];
    conv_attrs->output_c = Y->shape_.dims_[1];
    conv_attrs->output_h = 1;
    conv_attrs->output_w = Y->shape_.dims_[2];
    conv_attrs->weight_h = 1;
    conv_attrs->weight_w = attrs->kernel;
    conv_attrs->stride_h = 1;
    conv_attrs->stride_w = attrs->stride;
    conv_attrs->padding_h_up = 0;
    conv_attrs->padding_h_down = 0;
    conv_attrs->padding_w_left = attrs->pad[0];
    conv_attrs->padding_w_right = attrs->pad[1];

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
    conv_attrs->positive_shift_value = ((q_x + q_w) > q_y) ? (q_x + q_w - q_y) : 0;
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

/**
 * @brief Execute 1D convolution with integer precision
 * @param X Input tensor
 * @param W Weight tensor
 * @param Bias Bias tensor (optional)
 * @param Y Output tensor
 * @param Temp Workspace tensor
 * @param attrs Convolution attributes
 * @return int32_t Execution status
 */
int32_t conv1dint_luna(tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y, tTensor *Temp, Conv1dIntAttrs *attrs) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    int8_t *src = (int8_t *)(X->dptr_);
    int8_t *weight = (int8_t *)(W->dptr_);
    int32_t *bias = (Bias != NULL) ? (int32_t *)(Bias->dptr_) : NULL;

    conv_struct_t conv_attrs;
    luna_cnn_static_para_t conv_static_para;
    conv1dint_para_init(attrs, &conv_attrs, X, W, Bias, Y);

    int32_t shift = 0;
    int32_t q_x = (int32_t)X->scale_;
    int32_t q_w = (int32_t)W->scale_;
    int32_t q_y = (int32_t)Y->scale_;
    if (q_x + q_w - q_y < 0) {
        shift = q_y - q_x - q_w;
    }

    uint32_t input_c = conv_attrs.input_c;
    uint32_t output_c = conv_attrs.output_c;
    int32_t group = conv_attrs.group;
    int32_t kernel = attrs->kernel;

    if (kernel <= 12) {
        if (group == 1) {
            ret = luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_CONV1D);
            if (Y->dtype_ == Int8) {
                int8_t *dst = (int8_t *)(Y->dptr_);
                if (W->dtype_ == Int4) {
                    ret |= API_LIB(conv1d_i8i4o8)(src, weight, bias, dst, &conv_static_para);
                } else if (W->dtype_ == Int8) {
                    ret |= API_LIB(conv1d_i8i8o8)(src, weight, bias, dst, &conv_static_para);
                }
            } else if (Y->dtype_ == Int32) {
                int32_t *dst = (int32_t *)(Y->dptr_);
                int32_t size = getShapeSize(&(Y->shape_));
                if (W->dtype_ == Int4) {
                    ret |= API_LIB(conv1d_i8i4o32)(src, weight, bias, dst, &conv_static_para);
                } else if (W->dtype_ == Int8) {
                    ret |= API_LIB(conv1d_i8i8o32)(src, weight, bias, dst, &conv_static_para);
                }
                if (shift != 0) {
                    ret |= API_LIB(scale_i32i32o32)(dst, 1UL << shift, dst, size, 0);
                }
            }
        } else if ((group == input_c) && (group == output_c)) { // Depthwise convolution
            ret = luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_DEPTHWISE1D);
            if (Y->dtype_ == Int8) {
                int8_t *dst = (int8_t *)(Y->dptr_);
                if (W->dtype_ == Int4) {
                    ret |= API_LIB(depthwise1d_i8i4o8)(src, weight, bias, dst, &conv_static_para);
                } else if (W->dtype_ == Int8) {
                    ret |= API_LIB(depthwise1d_i8i8o8)(src, weight, bias, dst, &conv_static_para);
                }
            } else if (Y->dtype_ == Int32) {
                int32_t *dst = (int32_t *)(Y->dptr_);
                int32_t size = getShapeSize(&(Y->shape_));
                if (W->dtype_ == Int4) {
                    ret |= API_LIB(depthwise1d_i8i4o32)(src, weight, bias, dst, &conv_static_para);
                } else if (W->dtype_ == Int8) {
                    ret |= API_LIB(depthwise1d_i8i8o32)(src, weight, bias, dst, &conv_static_para);
                }
                if (shift != 0) {
                    ret |= API_LIB(scale_i32i32o32)(dst, 1UL << shift, dst, size, 0);
                }
            }
        } else { // Group convolution, should be split in tpacker
            return T_ERR_INVALID_PARA;
        }
    } else {
        printf("conv1d do not support: kernel > 12\n");
        return T_ERR_INVALID_PARA;
    }

    return ret;
}

#endif  // _CONV1DINT_ARCS_H_