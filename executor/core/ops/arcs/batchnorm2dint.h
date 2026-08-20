#ifndef _BATCHNORMINT_ARCS_H_
#define _BATCHNORMINT_ARCS_H_

#include <math.h>
#include "c_api/thinker_define.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"

/**
 * @brief Execute batch normalization with integer precision
 * @param X Input tensor
 * @param W Weight tensor (scale factors)
 * @param Bias Bias tensor (offset values)
 * @param Y Output tensor
 * @param workspace Workspace tensor for intermediate results
 * @return int32_t Execution status
 */
int32_t batchnormint_luna(const tTensor *X, const tTensor *W, const tTensor *Bias, tTensor *Y, tTensor *workspace) {
    #if THINKER_PARAM_CHECK
    if (X == NULL || W == NULL || Bias == NULL || Y == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->shape_.ndim_ != 4 || Y->shape_.ndim_ != 4 ||
                        X->dtype_ != Int8 || Y->dtype_ != Int8 || W->dtype_ != Int8 ||
                        Bias->dtype_ != Int32) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t N = X->shape_.dims_[0];                         // Batch size
    int32_t C = X->shape_.dims_[1];                         // Number of channels
    int32_t F = X->shape_.dims_[2] * X->shape_.dims_[3];    // Number of features per channel
    int32_t one_batch_size = F * C;                         // Size of one batch
    #if THINKER_RUNTIME_CHECK
    if (N <= 0 || C <= 0 || F <= 0 ||
                          !equalShape(&X->shape_, &Y->shape_) || X->mem_.type_ != 2 ||
                          Y->mem_.type_ != 2 ||
                          getShapeSize((tShape *)&W->shape_) != (size_t)C ||
                          getShapeSize((tShape *)&Bias->shape_) != (size_t)C ||
                          X->dptr_ == 0 || W->dptr_ == 0 || Bias->dptr_ == 0 ||
                          Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (workspace == NULL || workspace->dptr_ == 0 ||
                          workspace->mem_.type_ != 2 || workspace->shape_.ndim_ != 1 ||
                          workspace->shape_.dims_[0] < F * (int32_t)sizeof(int32_t)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    int8_t *p_src = (int8_t *)X->dptr_;    // Pointer to input data
    int8_t *p_dst = (int8_t *)Y->dptr_;    // Pointer to output data
    int8_t *p_weight = (int8_t *)W->dptr_; // Pointer to weight data (scale factors)
    int32_t *p_bias = (int32_t *)Bias->dptr_; // Pointer to bias data (offset values)
    int32_t *p_tmp = (int32_t *)workspace->dptr_;  // Pointer to temporary workspace

    int32_t q_x = (int32_t)X->scale_;     // Input quantization scale
    int32_t q_w = (int32_t)W->scale_;     // Weight quantization scale
    int32_t q_o = (int32_t)Y->scale_;     // Output quantization scale
    int32_t shift = q_x + q_w - q_o;      // Shift value for quantization
    #if THINKER_PARAM_CHECK
    if (shift < 0 || shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    for (int32_t i = 0; i < N; i++) { // Iterate over batches
        for (int32_t j = 0; j < C; j++) { // Iterate over channels
            int8_t w_val = *(p_weight + j); // Current channel's scale factor
            int32_t b_val = *(p_bias + j);   // Current channel's bias value

            int8_t *p_in = p_src + i * one_batch_size + j * F; // Input pointer for current channel
            int8_t *p_ou = p_dst + i * one_batch_size + j * F; // Output pointer for current channel

            THINKER_RET_CHECK(API_LIB(scale_i8i8o32)(p_in, w_val, p_tmp, F, 0), "luna_scale_i8i8o32"); // Apply scaling
            THINKER_RET_CHECK(API_LIB(offset_i32i32o8)(p_tmp, b_val, p_ou, F, shift), "luna_offset_i32i32o8"); // Apply offset and quantize
        }
    }

    return T_SUCCESS;
}

#endif  // _BATCHNORMINT_ARCS_H_
