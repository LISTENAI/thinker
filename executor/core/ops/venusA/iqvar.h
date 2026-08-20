#ifndef _VAR_LUNA_H_
#define _VAR_LUNA_H_

#include <math.h>
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
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

static inline int8_t iqvar_sat_i8_from_i64(int64_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

/**
 * @brief Quantized variance calculation implementation
 * @param X Input tensor
 * @param Y Output tensor
 * @param temp Temporary workspace tensor
 * @param attrs Variance attributes
 * @return int32_t Operation status
 */
int32_t iqvar_luna(tTensor *X, tTensor *Y, tTensor *temp, iqvarAttrs *attrs) {
#if THINKER_PARAM_CHECK
if (X->dtype_ != Int8 || Y->dtype_ != Int8 ||
                    X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
    return (T_ERR_INVALID_DATATYPE);
}

if (X->shape_.ndim_ != 3 || temp == NULL || temp->dptr_ == 0 ||
                    temp->mem_.type_ != 2) {
    return (T_ERR_INVALID_PARA);
}
#endif

    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t shift = x_q * 2 - y_q;
#if THINKER_PARAM_CHECK
if (shift < -30 || shift > 30) {
    return (T_ERR_INVALID_PARA);
}
#endif

    int8_t *src = (int8_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    int32_t n_dim = X->shape_.ndim_;
    int32_t dims = attrs->dims;
    if (dims < 0) {
        dims += n_dim;
    }
#if THINKER_PARAM_CHECK
if (dims != n_dim - 1 && dims != n_dim - 2) {
    return (T_ERR_INVALID_PARA);
}
#endif

    int32_t d0 = X->shape_.dims_[n_dim - 3];
    int32_t d1 = X->shape_.dims_[n_dim - 2];
    int32_t d2 = X->shape_.dims_[n_dim - 1];
    int32_t leading = d0 * d1;
    int32_t F = d2;
    int32_t input_size = getTensorSize(X);
    int8_t *p_tmp = (int8_t *)temp->dptr_;
    size_t workspace_size = getTensorDataSize(temp);

    int8_t *work_base = p_tmp;
    if (dims == n_dim - 2) {
        uint32_t axis[3] = {0, 2, 1};
        uint32_t in_shape[3] = {d0, d1, d2};
        if (workspace_size < ALIGN4(input_size) + ALIGN4(d1 * (int32_t)sizeof(int16_t)) +
                             d1 * (int32_t)sizeof(int32_t) + 8) {
            return T_ERR_NO_WORKSPACE;
        }
        THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)(src, p_tmp, in_shape, axis, 3), "luna_trans_axis_i8o8");
        src = p_tmp;
        work_base = p_tmp + ALIGN4(input_size);
        leading = d0 * d2;
        F = d1;
    } else if (workspace_size < ALIGN4(F * (int32_t)sizeof(int16_t)) +
                                 F * (int32_t)sizeof(int32_t) + 8) {
        return T_ERR_NO_WORKSPACE;
    }
    #if THINKER_PARAM_CHECK
    if (F <= 0 || F > 131071) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int16_t *p_square_i16 = (int16_t *)work_base;
    int32_t *p_square_i32 = (int32_t *)(work_base + ALIGN4(F * (int32_t)sizeof(int16_t)));
    int32_t *sum_x = p_square_i32 + F;
    int32_t *sum_x2 = p_square_i32 + F + 1;

    for (int32_t i = 0; i < leading; i++) {
        int8_t *p_src_once = src + i * F;
        int8_t *p_dst_once = dst + i;

        THINKER_RET_CHECK(API_LIB(mul_i8i8o16)(p_src_once, p_src_once, p_square_i16, F, 0), "luna_mul_i8i8o16");
        THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(p_square_i16, 1, p_square_i32, F, 0), "luna_scale_i16i16o32");
        THINKER_RET_CHECK(API_LIB(vector_sum_i32o32)(p_square_i32, sum_x2, F, 0), "luna_vector_sum_i32o32");
        THINKER_RET_CHECK(API_LIB(vector_sum_i8o32)(p_src_once, sum_x, F, 0), "luna_vector_sum_i8o32");

        int64_t numerator = (int64_t)F * (int64_t)(*sum_x2) - (int64_t)(*sum_x) * (int64_t)(*sum_x);
        if (numerator < 0) {
            numerator = 0;
        }
        int64_t denom = (F > 1) ? (int64_t)F * (int64_t)(F - 1) : 1;
        double scaled = ldexp((double)numerator / (double)denom, -shift);
        int64_t rounded = (int64_t)(scaled + 0.5);
        *p_dst_once = iqvar_sat_i8_from_i64(rounded);
    }

    return T_SUCCESS;
}
#endif  // _VAR_LUNA_H_
