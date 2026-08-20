#ifndef _BATCHNORM1DINT_VENUS_H_
#define _BATCHNORM1DINT_VENUS_H_

#include "c_api/thinker_define.h"
#include "core/comm/utils.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"

int32_t batchnorm1dint_luna(const tTensor *X, const tTensor *W, const tTensor *Bias,
                            tTensor *Y, tTensor *workspace) {
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

    int32_t N = X->shape_.dims_[0];
    int32_t C = X->shape_.dims_[1];
    int32_t F = X->shape_.dims_[2];
    int32_t one_batch_size = C * F;
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
                          workspace->shape_.dims_[0] < F * 4) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif

    int8_t *p_src = (int8_t *)X->dptr_;
    int8_t *p_dst = (int8_t *)Y->dptr_;
    int8_t *p_weight = (int8_t *)W->dptr_;
    int32_t *p_bias = (int32_t *)Bias->dptr_;
    int32_t *p_tmp = (int32_t *)workspace->dptr_;
    int32_t shift = (int32_t)X->scale_ + (int32_t)W->scale_ - (int32_t)Y->scale_;
    #if THINKER_PARAM_CHECK
    if (shift < 0 || shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
    #endif

    for (int32_t i = 0; i < N; ++i) {
        for (int32_t j = 0; j < C; ++j) {
            int8_t *p_in = p_src + i * one_batch_size + j * F;
            int8_t *p_out = p_dst + i * one_batch_size + j * F;
            THINKER_RET_CHECK(API_LIB(scale_q7_int32)(p_in, p_weight[j], p_tmp, F, 0),
                              "luna_scale_q7_int32");
            THINKER_RET_CHECK(API_LIB(offset_q31_int8)(p_tmp, p_bias[j], p_out, F, shift),
                              "luna_offset_q31_int8");
        }
    }
    return T_SUCCESS;
}

#endif
