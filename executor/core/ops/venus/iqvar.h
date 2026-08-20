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

/**
 * @brief Integer quantized variance calculation
 * @param X Input tensor
 * @param Y Output tensor
 * @param temp Temporary tensor (if needed)
 * @param attrs Variance calculation attributes
 * @return int32_t Operation status
 */
int32_t iqvar_luna(tTensor *X, tTensor *Y, tTensor *temp, iqvarAttrs *attrs) {
    #if THINKER_PARAM_CHECK
    if (X->dtype_ != Int8 || Y->dtype_ != Int8 ||
                        X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
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
    int32_t leading = X->shape_.dims_[n_dim - 3] * X->shape_.dims_[n_dim - 2];
    int32_t F = X->shape_.dims_[n_dim - 1];
    size_t workspace_size = getTensorDataSize(temp);
    for (int32_t i = 0; i < n_dim - 3; ++i) {
        #if THINKER_PARAM_CHECK
        if (X->shape_.dims_[i] != 1) {
            return (T_ERR_INVALID_PARA);
        }
        #endif
    }

    if (X->dtype_ == Int8) {
        int32_t shift = x_q * 2 - y_q;
        int8_t *p_tmp = (int8_t *)temp->dptr_;

        if (dims != -1 && dims != (n_dim - 1)) {
            leading = X->shape_.dims_[n_dim - 3] * X->shape_.dims_[n_dim - 1];
            F = X->shape_.dims_[n_dim - 2];
            uint32_t axis[3] = {0, 2, 1};
            uint32_t in_shape[3] = {
                X->shape_.dims_[n_dim - 3],
                X->shape_.dims_[n_dim - 2],
                X->shape_.dims_[n_dim - 1]
            };
            #if THINKER_RUNTIME_CHECK
            if (workspace_size < ALIGN4(getTensorSize(X))) {
                return (T_ERR_NO_WORKSPACE);
            }
            #endif
            THINKER_RET_CHECK(API_LIB(trans_axis_q7)(src, p_tmp, in_shape, axis, 3), "luna_trans_axis_q7");
            src = p_tmp;
        }
        #if THINKER_PARAM_CHECK
        if (shift < 0 || shift > 30 || F <= 0 || F > 23726566) {
            return (T_ERR_INVALID_PARA);
        }
        #endif

        for (int32_t i = 0; i < leading; ++i) {
            int8_t *p_src_once = src + i * F;
            int8_t *p_dst_once = dst + i;
            int64_t sum_x = 0;
            int64_t sum_x2 = 0;

            for (int32_t j = 0; j < F; ++j) {
                int64_t value = p_src_once[j];
                sum_x += value;
                sum_x2 += value * value;
            }

            int64_t numerator = (int64_t)F * sum_x2 - sum_x * sum_x;
            double divisor = F > 1 ? (double)F * (F - 1) : 1.0;
            float tmp_out = (float)((double)numerator / ldexp(divisor, shift));

            quant(&tmp_out, p_dst_once, 1, 0);
        }
    }

    return T_SUCCESS;
}

#endif
