#ifndef _BATCHNORM1DINT_VENUSA_H_
#define _BATCHNORM1DINT_VENUSA_H_

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
 * @brief Perform batch normalization on integer data
 * @param X Input tensor
 * @param W Weight tensor (gamma)
 * @param Bias Bias tensor (beta)
 * @param Y Output tensor
 * @param workspace Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t batchnorm1dint_luna(const tTensor *X, const tTensor *W, const tTensor *Bias, tTensor *Y, tTensor *workspace) {
    #if THINKER_PARAM_CHECK
    if (X == NULL || W == NULL || Bias == NULL || Y == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->shape_.ndim_ != 3 || Y->shape_.ndim_ != 3 ||
                        X->dtype_ != Int8 || W->dtype_ != Int8 ||
                        Bias->dtype_ != Int32 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int32_t N = X->shape_.dims_[0];    // Number of batches
    int32_t C = X->shape_.dims_[1];    // Number of channels
    int32_t F = X->shape_.dims_[2];    // Features per channel
    int32_t one_batch_size = F * C;    // Size of one batch

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
                          workspace->shape_.dims_[0] < ALIGN4(F * 6)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    int8_t *p_src = (int8_t *)X->dptr_;    // Input data pointer
    int8_t *p_dst = (int8_t *)Y->dptr_;    // Output data pointer
    int8_t *p_weight = (int8_t *)W->dptr_;    // Weight data pointer (gamma)
    int32_t *p_bias = (int32_t *)Bias->dptr_;    // Bias data pointer (beta)
    int16_t *p_tmp = (int16_t *)workspace->dptr_;  // Pointer to temporary workspace
    int32_t *p_tmp2 = (int32_t *)((int8_t *)workspace->dptr_ + ALIGN4(F * 2));

    int32_t q_x = (int32_t)X->scale_;    // Input scale factor
    int32_t q_w = (int32_t)W->scale_;    // Weight scale factor
    int32_t q_o = (int32_t)Y->scale_;    // Output scale factor
    int32_t shift = q_x + q_w - q_o;    // Scale shift for output

    #if THINKER_PARAM_CHECK
    if (shift < 0 || shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    for (int32_t i = 0; i < N; ++i) {    // Iterate over batches
        for (int32_t j = 0; j < C; ++j) {    // Iterate over channels
            int8_t w_val = *(p_weight + j);    // Current channel's gamma
            int32_t b_val = *(p_bias + j);    // Current channel's beta

            int8_t *p_in = p_src + i * one_batch_size + j * F;    // Input pointer for current channel
            int8_t *p_ou = p_dst + i * one_batch_size + j * F;    // Output pointer for current channel

            // Scale input by gamma and store intermediate results
            THINKER_RET_CHECK(API_LIB(scale_i8i8o16)(p_in, w_val, p_tmp, F, 0), "luna_scale_i8i8o16");
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(p_tmp, 1, p_tmp2, F, 0), "luna_scale_i16i16o32");
            // Apply bias and scale to get final output
            THINKER_RET_CHECK(API_LIB(offset_i32i32o8)(p_tmp2, b_val, p_ou, F, shift), "luna_offset_i32i32o8");
        }
    }

    return T_SUCCESS;
}

#endif  //_BATCHNORM1DINT_VENUSA_H_
