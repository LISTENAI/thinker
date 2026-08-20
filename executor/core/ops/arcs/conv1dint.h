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
    conv_attrs->dilation_h = 1;
    conv_attrs->dilation_w = 1;
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
    #if THINKER_RUNTIME_CHECK
    if (X->shape_.dims_[0] != 1 || Y->shape_.dims_[0] != 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
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
    int32_t depthwise = (group == input_c) && (group == output_c);
    int32_t ou_is_psram = (Y->mem_.type_ != 2);
    int32_t output_size = getShapeSize(&(Y->shape_)) * Y->byte_;
    int32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;

    #if THINKER_PARAM_CHECK
    if (X->dtype_ != Int8 || (W->dtype_ != Int4 && W->dtype_ != Int8)) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (Y->dtype_ != Int8 && Y->dtype_ != Int32) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (Bias != NULL && Bias->dtype_ != Int32) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (q_x + q_w - q_y > 63 ||
                        (shift != 0 && (Y->dtype_ != Int32 || shift > 30))) {
        return (T_ERR_INVALID_PARA);
    }

    if (conv_attrs.input_h != 1 || conv_attrs.weight_h != 1 || conv_attrs.output_h != 1) {
        return (T_ERR_INVALID_PARA);
    }

    if (conv_attrs.dilation_h != 1 || conv_attrs.dilation_w != 1) {
        return (T_ERR_INVALID_PARA);
    }

    if (kernel < 1 || kernel > 12) {
        return (T_ERR_INVALID_PARA);
    }

    if (conv_attrs.stride_w != 1 && conv_attrs.stride_w != 2 && conv_attrs.stride_w != 4) {
        return (T_ERR_INVALID_PARA);
    }

    if (conv_attrs.padding_w_left < 0 || conv_attrs.padding_w_left > 11 ||
                        conv_attrs.padding_w_right < 0 || conv_attrs.padding_w_right > 11) {
        return (T_ERR_INVALID_PARA);
    }

    if (conv_attrs.padding_w_left >= kernel || conv_attrs.padding_w_right >= kernel) {
        return (T_ERR_INVALID_PARA);
    }

    if (group != 1 && !depthwise) {
        return (T_ERR_INVALID_PARA);
    }

    if (kernel < conv_attrs.stride_w ||
                        conv_attrs.input_w + conv_attrs.padding_w_left + conv_attrs.padding_w_right < kernel) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (X->dptr_ == 0 || W->dptr_ == 0 || Y->dptr_ == 0 ||
                          X->dptr_ == Y->dptr_ ||
                          (Bias != NULL &&
                           (Bias->dptr_ == 0 ||
                            getTensorSize(Bias) !=
                                (size_t)Y->shape_.dims_[1]))) {
        return (T_ERR_INVALID_PARA);
    }

    if (q_x + q_w - q_y > 63 || shift > 30) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (ou_is_psram &&
                          (Temp == NULL || Temp->dptr_ == 0 ||
                           Temp->mem_.type_ != 2 || Temp->shape_.ndim_ != 1 ||
                           workspace_size < output_size)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para,
                                                depthwise ? LUNA_DEPTHWISE1D : LUNA_CONV1D),
                      "luna_split_conv_para_pack");
    if (Y->dtype_ == Int8) {
        int8_t *dst = ou_is_psram ? (int8_t *)Temp->dptr_ : (int8_t *)(Y->dptr_);
        if (W->dtype_ == Int4) {
            THINKER_RET_CHECK(depthwise ? API_LIB(depthwise1d_i8i4o8)(src, weight, bias, dst, &conv_static_para)
                                        : API_LIB(conv1d_i8i4o8)(src, weight, bias, dst, &conv_static_para),
                              depthwise ? "luna_depthwise1d_i8i4o8" : "luna_conv1d_i8i4o8");
        } else {
            THINKER_RET_CHECK(depthwise ? API_LIB(depthwise1d_i8i8o8)(src, weight, bias, dst, &conv_static_para)
                                        : API_LIB(conv1d_i8i8o8)(src, weight, bias, dst, &conv_static_para),
                              depthwise ? "luna_depthwise1d_i8i8o8" : "luna_conv1d_i8i8o8");
        }
    } else {
        int32_t *dst = ou_is_psram ? (int32_t *)Temp->dptr_ : (int32_t *)(Y->dptr_);
        int32_t size = getShapeSize(&(Y->shape_));
        if (W->dtype_ == Int4) {
            THINKER_RET_CHECK(depthwise ? API_LIB(depthwise1d_i8i4o32)(src, weight, bias, dst, &conv_static_para)
                                        : API_LIB(conv1d_i8i4o32)(src, weight, bias, dst, &conv_static_para),
                              depthwise ? "luna_depthwise1d_i8i4o32" : "luna_conv1d_i8i4o32");
        } else {
            THINKER_RET_CHECK(depthwise ? API_LIB(depthwise1d_i8i8o32)(src, weight, bias, dst, &conv_static_para)
                                        : API_LIB(conv1d_i8i8o32)(src, weight, bias, dst, &conv_static_para),
                              depthwise ? "luna_depthwise1d_i8i8o32" : "luna_conv1d_i8i8o32");
        }
        if (shift != 0) {
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(dst, 1UL << shift, dst, size, 0),
                              "luna_scale_i32i32o32");
        }
    }

    if (ou_is_psram) {
        opi_psram_cpy_out((int8_t *)Y->dptr_, (int8_t *)Temp->dptr_, output_size);
    }

    return T_SUCCESS;
}

#endif  // _CONV1DINT_ARCS_H_
