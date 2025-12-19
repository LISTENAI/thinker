#ifndef _TOPN2_LUNA_H_
#define _TOPN2_LUNA_H_

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
 * @param X Input tensor containing values and indices
 * @param Y Output tensor for values and indices
 * @param work_space Temporary workspace tensor
 * @param attrs TopN attributes containing dimension and max number
 * @return int32_t Operation status
 */
int32_t topn2_luna(tTensor *X, tTensor *Y, tTensor *work_space, topNAttrs *attrs) {
    int32_t ret = -1;
    int32_t axis = attrs->dim;
    int32_t n = attrs->max_num;
    int32_t n_dims = X->shape_.ndim_;
    int32_t once_size = (axis == -1) ? X->shape_.dims_[n_dims - 1] : X->shape_.dims_[axis];
    int32_t leading = 1;
    int32_t *p_tmp = (int32_t *)work_space->dptr_;

    // Calculate leading dimensions
    if (axis == -1 || axis == n_dims - 1) {
        leading = X->shape_.dims_[n_dims - 2];
    }

    // Handle different topN cases
    switch (n) {
        case 1: { // Top 1 case
            int16_t *p_src_val = (int16_t *)X->dptr_;
            int16_t *p_src_idx = (int16_t *)X->dptr_ + leading * once_size;
            int16_t *p_dst_val = (int16_t *)Y->dptr_;
            int16_t *p_dst_idx = (int16_t *)Y->dptr_ + leading;

            for (int i = 0; i < leading; ++i) {
                int16_t *p_src_val_tmp = p_src_val + i * once_size;
                int16_t *p_src_idx_tmp = p_src_idx + i * once_size;
                ret = API_LIB(max_q15)(p_src_val_tmp, p_tmp, once_size);
                p_dst_val[i] = (int16_t)p_tmp[0];
                p_dst_idx[i] = p_src_idx_tmp[p_tmp[1]];
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