#ifndef _FFNINT_LUNA_H_
#define _FFNINT_LUNA_H_

#include <math.h>
#include "core/operator_attrs.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Execute integer-aware feed-forward network (FFN) transformation
 * @param p_input Input tensor (T, D)
 * @param p_weight_m0 First layer weight tensor (Dh, D)
 * @param p_bias_m0 First layer bias tensor
 * @param p_weight_m1 Second layer weight tensor (Do, Dh)
 * @param p_bias_m1 Second layer bias tensor
 * @param p_output Output tensor (T, Do)
 * @param p_temp Temporary workspace
 * @param dim_in Input dimension (D)
 * @param dim_hidden Hidden layer dimension (Dh)
 * @param dim_out Output dimension (Do)
 * @param seq_len Sequence length (T)
 * @param q_input_m0 Input quantization scale for first layer
 * @param q_weight_m0 Weight quantization scale for first layer
 * @param q_output_m0 Output quantization scale for first layer
 * @param q_input_m1 Input quantization scale for second layer
 * @param q_weight_m1 Weight quantization scale for second layer
 * @param q_output_m1 Output quantization scale for second layer
 * @return int32_t Execution status
 */
int32_t luna_ffn_int_trans(int8_t *p_input, 
                          int8_t *p_weight_m0, int32_t *p_bias_m0, 
                          int8_t *p_weight_m1, int32_t *p_bias_m1, 
                          int8_t *p_output, int8_t *p_temp,
                          uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len,
                          int32_t q_input_m0, int32_t q_weight_m0, int32_t q_output_m0, 
                          int32_t q_input_m1, int32_t q_weight_m1, int32_t q_output_m1) {
    int8_t *p_output1 = p_temp;
    p_temp += seq_len * dim_hidden;

    // First layer: (T, D) * (Dh, D) => (T, Dh)
    for (int32_t i = 0; i < seq_len; i++) {
        THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight_m0, p_input + i * dim_in, p_bias_m0, 
                                                      p_output1 + i * dim_hidden, dim_hidden, dim_in, 1, 
                                                      q_input_m0 + q_weight_m0 - q_output_m0), "luna_split_mat_mul_bias_i8i8i32o8");
    }

    // Apply ReLU activation
    THINKER_RET_CHECK(API_LIB(relu_i8o8)(p_output1, p_output1, seq_len * dim_hidden, 0), "luna_relu_i8o8");

    // Second layer: (T, Dh) * (Do, Dh) => (T, Do)
    for (int32_t i = 0; i < seq_len; i++) {
        THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight_m1, p_output1 + i * dim_hidden, p_bias_m1, 
                                                      p_output + i * dim_out, dim_out, dim_hidden, 1, 
                                                      q_input_m1 + q_weight_m1 - q_output_m1), "luna_split_mat_mul_bias_i8i8i32o8");
    }

    return T_SUCCESS;
}

/**
 * @brief Execute integer-aware feed-forward network (FFN)
 * @param X Input tensor
 * @param weight1 First layer weight tensor
 * @param bias1 First layer bias tensor
 * @param weight2 Second layer weight tensor
 * @param bias2 Second layer bias tensor
 * @param workspace Workspace tensor for intermediate results
 * @param Y Output tensor
 * @param attrs FFN attributes
 * @return int32_t Execution status
 */
int32_t ffnint_luna(tTensor *X, tTensor *weight1, tTensor *bias1, tTensor *weight2, tTensor *bias2, 
                    tTensor *workspace, tTensor *Y, FFNIntAttrs *attrs) {
    int64_t shift0;
    int64_t shift1;
    int64_t required_workspace;
    int32_t seq_len;
    int32_t dim_in;
    int32_t dim_hidden;
    int32_t dim_out;

    #if THINKER_PARAM_CHECK
    if (X == NULL || weight1 == NULL || bias1 == NULL || weight2 == NULL || bias2 == NULL ||
                        workspace == NULL || Y == NULL || attrs == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->dtype_ != Int8 || weight1->dtype_ != Int8 || weight2->dtype_ != Int8 ||
                        bias1->dtype_ != Int32 || bias2->dtype_ != Int32 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (X->shape_.ndim_ != 3 || Y->shape_.ndim_ != 3 || weight1->shape_.ndim_ != 2 ||
                        weight2->shape_.ndim_ != 2 || bias1->shape_.ndim_ != 1 || bias2->shape_.ndim_ != 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (X->mem_.type_ != 2 || Y->mem_.type_ != 2 || workspace->mem_.type_ != 2 ||
                          workspace->dptr_ == 0) {
        return (T_ERR_INVALID_PLATFROM);
    }

    if (X->dptr_ == Y->dptr_ || Y->dptr_ == workspace->dptr_) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif

    seq_len = X->shape_.dims_[0] * X->shape_.dims_[1];
    dim_in = X->shape_.dims_[2];
    dim_hidden = weight1->shape_.dims_[0];
    dim_out = weight2->shape_.dims_[0];
    #if THINKER_PARAM_CHECK
    if (seq_len <= 0 || dim_in <= 0 || dim_hidden <= 0 || dim_out <= 0 ||
                        Y->shape_.dims_[0] != X->shape_.dims_[0] || Y->shape_.dims_[1] != X->shape_.dims_[1] ||
                        Y->shape_.dims_[2] != dim_out || weight1->shape_.dims_[1] != dim_in ||
                        weight2->shape_.dims_[1] != dim_hidden || bias1->shape_.dims_[0] != dim_hidden ||
                        bias2->shape_.dims_[0] != dim_out) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    shift0 = (int64_t)X->scale_ + weight1->scale_ - attrs->middle_scale;
    shift1 = (int64_t)attrs->middle_scale + weight2->scale_ - Y->scale_;
    #if THINKER_PARAM_CHECK
    if (shift0 < 0 || shift0 > 63 || shift1 < 0 || shift1 > 63) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    required_workspace = (int64_t)(dim_hidden + dim_out) * sizeof(int32_t) +
                         (int64_t)seq_len * dim_hidden;
    #if THINKER_RUNTIME_CHECK
    if (required_workspace > INT32_MAX ||
                          (int64_t)getTensorSize(workspace) * workspace->byte_ < required_workspace) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    int8_t *p_input = (int8_t *)X->dptr_;
    int8_t *p_weight_m0 = (int8_t *)weight1->dptr_;
    int8_t *p_weight_m1 = (int8_t *)weight2->dptr_;
    int8_t *p_output = (int8_t *)Y->dptr_;

    int32_t *p_bias_m0 = (int32_t *)workspace->dptr_;
    uint32_t size_bias = getShapeSize(&(bias1->shape_)) * sizeof(int32_t);
    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)p_bias_m0, (int8_t *)bias1->dptr_, size_bias), "luna_memcpy_i8o8");

    int32_t *p_bias_m1 = (int32_t *)workspace->dptr_ + getShapeSize(&(bias1->shape_));
    size_bias = getShapeSize(&(bias2->shape_)) * sizeof(int32_t);
    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)p_bias_m1, (int8_t *)bias2->dptr_, size_bias), "luna_memcpy_i8o8");

    int8_t *p_temp = (int8_t *)p_bias_m1 + getShapeSize(&(bias2->shape_)) * 4;

    int32_t q_input_m0 = X->scale_;
    int32_t q_weight_m0 = weight1->scale_;
    int32_t q_output_m0 = attrs->middle_scale;
    int32_t q_input_m1 = attrs->middle_scale;
    int32_t q_weight_m1 = weight2->scale_;
    int32_t q_output_m1 = Y->scale_;

#if (defined(WIN32) || defined(linux))
    THINKER_RET_CHECK(luna_ffn_int_trans(p_input, p_weight_m0, p_bias_m0, p_weight_m1, p_bias_m1, p_output, p_temp,
                            dim_in, dim_hidden, dim_out, seq_len, q_input_m0, q_weight_m0, q_output_m0,
                            q_input_m1, q_weight_m1, q_output_m1), "luna_ffn_int_trans");
#else
#include "lunaext_ffn.h"
    THINKER_RET_CHECK(nlang_ffn_int_trans(p_input, p_weight_m0, p_bias_m0, p_weight_m1, p_bias_m1, p_output, p_temp,
                            dim_in, dim_hidden, dim_out, seq_len, q_input_m0, q_weight_m0, q_output_m0,
                            q_input_m1, q_weight_m1, q_output_m1), "nlang_ffn_int_trans");
#endif

    return T_SUCCESS;
}

#endif  // _FFNINT_LUNA_H_
