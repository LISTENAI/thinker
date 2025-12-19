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
    int32_t ret = -1;
    int32_t axis = attrs->dim;
    int32_t n = attrs->max_num;
    int32_t n_dims = X->shape_.ndim_;
    int32_t once_size = axis == -1 ? X->shape_.dims_[n_dims - 1] : X->shape_.dims_[axis];
    int32_t leading = 1;
    int32_t *p_tmp = (int32_t *)work_space->dptr_;
    int64_t idx_offset = *(int64_t *)Index->dptr_;

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
                ret = API_LIB(max_q7)(p_src, p_tmp, once_size);
                p_dst_val[i] = (int16_t)p_tmp[0];
                p_dst_idx[i] = (int16_t)(p_tmp[1] + idx_offset);
            }
            break;
        }
        default:
            THINKER_LOG_FATAL("Only top 1 is supported currently.");
            return -1;
    }

    return ret;
}

#endif