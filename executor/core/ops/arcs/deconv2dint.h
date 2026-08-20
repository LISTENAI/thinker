#ifndef _DECONV2DINT_ARCS_H_
#define _DECONV2DINT_ARCS_H_

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

static int32_t luna_quant_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    }
    return x >> shift;
}

/**
 * @brief Initialize parameters for 2D transposed convolution (deconvolution)
 * @param attrs Transposed convolution attributes
 * @param conv_attrs Convolution structure for 2D transposed convolution
 * @param X Input tensor
 * @param W Weight tensor
 * @param Bias Bias tensor (optional)
 * @param Y Output tensor
 */
static void deconv2dint_luna_para_init(ConvTranspose2dIntAttrs *attrs, conv_struct_t *conv_attrs, tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y) {
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
    conv_attrs->padding_h_up = attrs->kernel[0] - attrs->pad[0] - 1;
    conv_attrs->padding_h_down = attrs->kernel[0] - attrs->pad[2] - 1 + attrs->output_padding[0];
    conv_attrs->padding_w_left = attrs->kernel[1] - attrs->pad[1] - 1;
    conv_attrs->padding_w_right = attrs->kernel[1] - attrs->pad[3] - 1 + attrs->output_padding[1];
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
    conv_attrs->positive_shift_type = attrs->quant_type == 0 ? ShiftType_Floor : ShiftType_FloorX05;
    conv_attrs->positive_shift_value = q_x + q_w - q_y;
    conv_attrs->negative_shift_type = attrs->quant_type == 0 ? ShiftType_Floor : ShiftType_FloorX05;
    conv_attrs->negative_shift_value = conv_attrs->positive_shift_value;

    uint8_t data_mem_type = (X->mem_.type_ & 0x0F) << 4;
    if (attrs->group == conv_attrs->input_c && attrs->group == conv_attrs->output_c) {
        conv_attrs->data_mem_type = data_mem_type | (W->mem_.type_ & 0x0F);
    } else {
        conv_attrs->data_mem_type = W->mem_.type_;
    }
    conv_attrs->ou_bits = Y->byte_ * 8;
    conv_attrs->weight_bits = (W->dtype_ == Int4) ? 4 : 8;
    conv_attrs->out_padding_h = 0;
    conv_attrs->out_padding_w = 0;
    conv_attrs->group = attrs->group;
}

/**
 * @brief Execute 2D transposed convolution (deconvolution) with integer precision
 * @param X Input tensor
 * @param W Weight tensor
 * @param Bias Bias tensor (optional)
 * @param Y Output tensor
 * @param Temp Workspace tensor
 * @param attrs Transposed convolution attributes
 * @return int32_t Execution status
 */
int32_t deconv2dint_luna(tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y, tTensor *Temp, ConvTranspose2dIntAttrs *attrs) {
    int32_t quant_shift = (int32_t)X->scale_ + (int32_t)W->scale_ - (int32_t)Y->scale_;
    if (quant_shift < 0 || quant_shift > 63 || attrs->pad[0] >= attrs->kernel[0] ||
        attrs->pad[2] >= attrs->kernel[0] || attrs->pad[1] >= attrs->kernel[1] ||
        attrs->pad[3] >= attrs->kernel[1]) return T_ERR_INVALID_PARA;
    #if THINKER_PARAM_CHECK
    if (X->shape_.dims_[0] != 1 || Y->shape_.dims_[0] != 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int8_t *src = (int8_t *)(X->dptr_);
    int8_t *weight = (int8_t *)(W->dptr_);
    int32_t *bias = (Bias != NULL) ? (int32_t *)(Bias->dptr_) : NULL;
    int8_t *dst = (int8_t *)(Y->dptr_);

    conv_struct_t conv_attrs;
    luna_cnn_static_para_t conv_static_para;
    deconv2dint_luna_para_init(attrs, &conv_attrs, X, W, Bias, Y);

    uint32_t input_c = conv_attrs.input_c;
    uint32_t output_c = conv_attrs.output_c;
    int32_t group = conv_attrs.group;
    int32_t k_h = conv_attrs.weight_h;
    int32_t k_w = conv_attrs.weight_w;
    int32_t log2n_stride_w = (conv_attrs.stride_w >> 1);
    int32_t align_w =  luna_quant_ceil(conv_attrs.input_w, (2 + log2n_stride_w)) << (2 + log2n_stride_w);
    int32_t data_size = ALIGN8(conv_attrs.input_c) * align_w * conv_attrs.input_h;
    int32_t weight_size = ALIGN2(conv_attrs.output_c) * ALIGN8(conv_attrs.input_c) * k_h * k_w;

    #if THINKER_PARAM_CHECK
    if (X->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (X->shape_.dims_[0] != 1 || Y->shape_.dims_[0] != 1) {
        return (T_ERR_INVALID_PARA);
    }

    if ((W->dtype_ != Int4) && (W->dtype_ != Int8)) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (Bias != NULL && Bias->dtype_ != Int32) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (group != 1) {
        return (T_ERR_INVALID_PARA);
    }

    if (k_h < 1 || k_h > 12 || k_w < 1 || k_w > 12) {
        return (T_ERR_INVALID_PARA);
    }

    if (conv_attrs.stride_h != 1 && conv_attrs.stride_h != 2 && conv_attrs.stride_h != 4) {
        return (T_ERR_INVALID_PARA);
    }

    if (conv_attrs.stride_w != 1 && conv_attrs.stride_w != 2 && conv_attrs.stride_w != 4) {
        return (T_ERR_INVALID_PARA);
    }

    if (conv_attrs.dilation_h != 1 || conv_attrs.dilation_w != 1) {
        return (T_ERR_INVALID_PARA);
    }

    if (data_size > 16384 || weight_size > 8192) {
        return (T_ERR_INVALID_PARA);
    }

    if (attrs->pad[0] < 0 || attrs->pad[1] < 0 || attrs->pad[2] < 0 || attrs->pad[3] < 0 ||
                        attrs->pad[0] > 11 || attrs->pad[1] > 11 || attrs->pad[2] > 11 || attrs->pad[3] > 11 ||
                        k_h < conv_attrs.stride_h || k_w < conv_attrs.stride_w || X->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (Y->mem_.type_ != 2) {
        return (T_ERR_INVALID_PLATFROM);
    }
#endif

    THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_DECONV),
                      "luna_split_conv_para_pack");
    if (W->dtype_ == Int4) {
        THINKER_RET_CHECK(API_LIB(deconv2d_i8i4o8)(src, weight, bias, dst, &conv_static_para),
                          "luna_deconv2d_i8i4o8");
    } else {
        THINKER_RET_CHECK(API_LIB(deconv2d_i8i8o8)(src, weight, bias, dst, &conv_static_para),
                          "luna_deconv2d_i8i8o8");
    }

    return T_SUCCESS;
}

#endif  // _DECONV2DINT_ARCS_H_
