#ifndef _VAR_LUNA_H_
#define _VAR_LUNA_H_

#include <math.h>
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Integer Quantized Variance operation
 * @param X Input tensor
 * @param Y Output tensor
 * @param temp Temporary workspace tensor
 * @param attrs Variance attributes
 * @return int32_t Operation status
 */
int32_t iqvar_luna(tTensor *X, tTensor *Y, tTensor *temp, iqvarAttrs *attrs) {
    #if THINKER_PARAM_CHECK
    if (X->dtype_ != Int8 || Y->dtype_ != Int8 ||
                         X->shape_.ndim_ != 3 || X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (temp == NULL || temp->dptr_ == 0 || temp->mem_.type_ != 2) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int8_t *src = (int8_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;

    int32_t n_dim = X->shape_.ndim_;
    int32_t dims = attrs->dims;
    if (dims < 0) dims += n_dim;
    #if THINKER_PARAM_CHECK
    if (dims != n_dim - 1 && dims != n_dim - 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t leading = X->shape_.dims_[n_dim - 3] * X->shape_.dims_[n_dim - 2];
    int32_t F = X->shape_.dims_[n_dim - 1];
    size_t input_size = getTensorSize(X);

    if (X->dtype_ == Int8) {
        int32_t shift = x_q * 2 - y_q;
        #if THINKER_PARAM_CHECK
        if (shift < -30 || shift > 30) {
            return (T_ERR_INVALID_PARA);
        }
#endif
        int8_t *p_tmp = (int8_t *)temp->dptr_;
        int8_t *work_base = p_tmp;
        size_t workspace_size = getTensorDataSize(temp);

        if (-1 == dims || (n_dim - 1) == dims) {
            work_base = p_tmp;
        } else {
            leading = X->shape_.dims_[n_dim - 3] * X->shape_.dims_[n_dim - 1];
            F = X->shape_.dims_[n_dim - 2];
            uint32_t axis[3] = {0, 2, 1};
            uint32_t in_shape[3] = {
                X->shape_.dims_[n_dim - 3],
                X->shape_.dims_[n_dim - 2],
                X->shape_.dims_[n_dim - 1]
            };
            THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)(src, p_tmp, in_shape, axis, 3), "luna_trans_axis_i8o8");
            src = p_tmp;
            work_base = p_tmp + ALIGN4(input_size);
        }
        #if THINKER_PARAM_CHECK
        if (F <= 0 || F > 131071) {
            return (T_ERR_INVALID_PARA);
        }
#endif

        size_t required = (size_t)(work_base - p_tmp) + (size_t)F * sizeof(int32_t) + 8;
        #if THINKER_RUNTIME_CHECK
        if (workspace_size < required) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        int32_t *p_square = (int32_t *)work_base;
        int32_t *sum_x = p_square + F;
        int32_t *sum_x2 = p_square + F + 1;

        for (int32_t i = 0; i < leading; ++i) {
            int8_t *p_src_once = src + i * F;
            int8_t *p_dst_once = dst + i;

            THINKER_RET_CHECK(API_LIB(mul_i8i8o32)(p_src_once, p_src_once, p_square, F, 0), "luna_mul_i8i8o32");
            THINKER_RET_CHECK(API_LIB(vector_sum_i32o32)(p_square, sum_x2, F, 0), "luna_vector_sum_i32o32");
            THINKER_RET_CHECK(API_LIB(vector_sum_i8o32)(p_src_once, sum_x, F, 0), "luna_vector_sum_i8o32");

            int32_t sum_x_val = *sum_x;
            int32_t sum_x2_val = *sum_x2;
            int64_t value = (int64_t)F * sum_x2_val -
                            (int64_t)sum_x_val * sum_x_val;
            if (value < 0) value = 0;
            int64_t denominator = F > 1 ? (int64_t)F * (F - 1) : 1;
            double scaled = ldexp((double)value / (double)denominator, -shift);
            int64_t rounded = (int64_t)(scaled + 0.5);
            *p_dst_once = (int8_t)(rounded > 127 ? 127 : rounded < 0 ? 0 : rounded);
        }
    }

    return T_SUCCESS;
}

#endif
