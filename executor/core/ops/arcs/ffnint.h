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
    int32_t ret = T_SUCCESS;
    int8_t *p_output1 = p_temp;
    p_temp += seq_len * dim_hidden;

    // First layer: (T, D) * (Dh, D) => (T, Dh)
    for (int32_t i = 0; i < seq_len; i++) {
        ret |= API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight_m0, p_input + i * dim_in, p_bias_m0, 
                                                      p_output1 + i * dim_hidden, dim_hidden, dim_in, 1, 
                                                      q_input_m0 + q_weight_m0 - q_output_m0);
    }

    // Apply ReLU activation
    ret |= API_LIB(relu_i8o8)(p_output1, p_output1, seq_len * dim_hidden, 0);

    // Second layer: (T, Dh) * (Do, Dh) => (T, Do)
    for (int32_t i = 0; i < seq_len; i++) {
        ret |= API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight_m1, p_output1 + i * dim_hidden, p_bias_m1, 
                                                      p_output + i * dim_out, dim_out, dim_hidden, 1, 
                                                      q_input_m1 + q_weight_m1 - q_output_m1);
    }

    return ret;
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
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    int8_t *p_input = (int8_t *)X->dptr_;
    int8_t *p_weight_m0 = (int8_t *)weight1->dptr_;
    int8_t *p_weight_m1 = (int8_t *)weight2->dptr_;
    int8_t *p_output = (int8_t *)Y->dptr_;

    int32_t *p_bias_m0 = (int32_t *)workspace->dptr_;
    uint32_t size_bias = getShapeSize(&(bias1->shape_)) * sizeof(int32_t);
    API_LIB(memcpy_i8o8)((int8_t *)p_bias_m0, (int8_t *)bias1->dptr_, size_bias);

    int32_t *p_bias_m1 = (int32_t *)workspace->dptr_ + getShapeSize(&(bias1->shape_));
    size_bias = getShapeSize(&(bias2->shape_)) * sizeof(int32_t);
    API_LIB(memcpy_i8o8)((int8_t *)p_bias_m1, (int8_t *)bias2->dptr_, size_bias);

    int8_t *p_temp = (int8_t *)p_bias_m1 + getShapeSize(&(bias2->shape_)) * 4;

    int32_t seq_len = X->shape_.dims_[0] * X->shape_.dims_[1];
    int32_t dim_in = X->shape_.dims_[2];
    int32_t dim_hidden = weight1->shape_.dims_[0];
    int32_t dim_out = weight2->shape_.dims_[0];

    int32_t q_input_m0 = X->scale_;
    int32_t q_weight_m0 = weight1->scale_;
    int32_t q_output_m0 = attrs->middle_scale;
    int32_t q_input_m1 = attrs->middle_scale;
    int32_t q_weight_m1 = weight2->scale_;
    int32_t q_output_m1 = Y->scale_;

#if (defined(WIN32) || defined(linux))
    ret = luna_ffn_int_trans(p_input, p_weight_m0, p_bias_m0, p_weight_m1, p_bias_m1, p_output, p_temp,
                            dim_in, dim_hidden, dim_out, seq_len, q_input_m0, q_weight_m0, q_output_m0,
                            q_input_m1, q_weight_m1, q_output_m1);
#else
#include "lunaext_ffn.h"
    ret = nlang_ffn_int_trans(p_input, p_weight_m0, p_bias_m0, p_weight_m1, p_bias_m1, p_output, p_temp,
                            dim_in, dim_hidden, dim_out, seq_len, q_input_m0, q_weight_m0, q_output_m0,
                            q_input_m1, q_weight_m1, q_output_m1);
#endif

    return ret;
}

#endif  // _FFNINT_LUNA_H_