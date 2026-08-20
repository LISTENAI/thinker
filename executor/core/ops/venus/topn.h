#ifndef _TOPN_LUNA_H_
#define _TOPN_LUNA_H_

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
 * @brief Find top N elements and their indices in a tensor
 * @param X Input tensor
 * @param Index Output tensor for indices
 * @param Y Output tensor for values
 * @param work_space Temporary workspace tensor
 * @param attrs TopN attributes containing dimension and max number
 * @return int32_t Operation status
 */
int32_t topn_luna(tTensor *X, tTensor *Index, tTensor *Y, tTensor *work_space, topNAttrs *attrs) {
    #if THINKER_PARAM_CHECK
    if (X == NULL || Index == NULL || Y == NULL || work_space == NULL ||
                        attrs == NULL || X->dptr_ == 0 || Index->dptr_ == 0 ||
                        Y->dptr_ == 0 || work_space->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t axis = attrs->dim;
    int32_t n = attrs->max_num;
    int32_t n_dims = X->shape_.ndim_;
    #if THINKER_PARAM_CHECK
    if (n_dims <= 0 || (axis != -1 && axis != n_dims - 1) || n != 1 ||
                        X->shape_.dims_[n_dims - 1] <= 0) {
        return (T_ERR_INVALID_PARA);
    }
    if (X->dtype_ != Int8 || Index->dtype_ != Int64 || Y->dtype_ != Int16) {
        return (T_ERR_INVALID_DATATYPE);
    }
    if (getTensorSize(Index) < 1 || Y->shape_.ndim_ != n_dims ||
        Y->shape_.dims_[0] != 2 ||
        (n_dims > 1 && Y->shape_.dims_[n_dims - 1] != 1)) {
        return (T_ERR_INVALID_DATA);
    }
    #endif
    for (int32_t i = 1; i < n_dims - 1; ++i) {
#if THINKER_PARAM_CHECK
        if (Y->shape_.dims_[i] != X->shape_.dims_[i]) {
            return (T_ERR_INVALID_DATA);
        }
#endif
    }
    #if THINKER_PARAM_CHECK
    if (X->shape_.dims_[0] != 1) {
        return (T_ERR_INVALID_DATA);
    }
    #endif
    #if THINKER_RUNTIME_CHECK
    if (work_space->shape_.ndim_ == 0 || work_space->shape_.dims_[0] < 8) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif
    int32_t once_size = axis == -1 ? X->shape_.dims_[n_dims - 1] : X->shape_.dims_[axis];
    int32_t leading = 1;
    int32_t *p_tmp = (int32_t *)work_space->dptr_;
    int64_t idx_offset = *(int64_t *)Index->dptr_;
    #if THINKER_PARAM_CHECK
    if (idx_offset < -32768 || idx_offset + once_size - 1 > 32767 || once_size <= 0) {
        return (T_ERR_INVALID_PARA);
    }
    #endif

    // Calculate leading dimensions
    if (axis == -1 || axis == n_dims - 1) {
        for (int i = 0; i < n_dims - 1; ++i) {
            leading *= X->shape_.dims_[i];
        }
    }

    // Handle different topN cases
    switch (n) {
        case 1: { // Top 1 case
            int16_t *p_dst_val = (int16_t *)Y->dptr_;
            int16_t *p_dst_idx = (int16_t *)Y->dptr_ + leading;

            for (int i = 0; i < leading; ++i) {
                int8_t *p_src = (int8_t *)X->dptr_ + i * once_size;
                THINKER_RET_CHECK(API_LIB(max_q7)(p_src, p_tmp, once_size), "luna_max_q7");
                p_dst_val[i] = (int16_t)p_tmp[0];
                p_dst_idx[i] = (int16_t)(p_tmp[1] + idx_offset);
            }
            break;
        }
        default:
            THINKER_LOG_FATAL("Only top 1 is supported currently.");
            #if THINKER_PARAM_CHECK
            if (1) {
                return (T_ERR_INVALID_PARA);
            }
            #endif
    }

    return T_SUCCESS;
}

#endif
