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
 * @brief Find top-N values from pre-computed sorted data
 * @param X Input tensor with pre-sorted values and indices
 * @param Y Output tensor containing top-N results
 * @param work_space Workspace buffer
 * @param attrs TopN attributes including dimension and max number
 * @return Operation status
 */
int32_t topn2_luna(tTensor *X, tTensor *Y, tTensor *work_space, topNAttrs *attrs) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    int32_t axis = attrs->dim;
    int32_t n = attrs->max_num;
    int32_t n_dims = X->shape_.ndim_;
    int32_t once_size = axis == -1 ? X->shape_.dims_[n_dims - 1] : X->shape_.dims_[axis];
    
    int32_t leading = 1;
    int32_t *p_tmp = (int32_t *)work_space->dptr_;

    // Calculate leading dimension for batch processing
    if (-1 == axis || (n_dims - 1) == axis) {
        leading = X->shape_.dims_[n_dims - 2];
    }  

    // Handle top-1 case only (as per switch statement)
    switch (n) {
        case 1: // top 1
        {
            int32_t *p_src_val = (int32_t *)X->dptr_;
            int32_t *p_src_idx = (int32_t *)X->dptr_ + leading * once_size;
            int32_t *p_dst_val = (int32_t *)Y->dptr_;
            int32_t *p_dst_idx = (int32_t *)Y->dptr_ + leading;
            
            for (int i = 0; i < leading; i++) {
                int32_t *p_src_val_tmp = (int32_t *)p_src_val + i * once_size;
                int32_t *p_src_idx_tmp = (int32_t *)p_src_idx + i * once_size;     
                ret = API_LIB(max_i32o32)(p_src_val_tmp, p_tmp, once_size);
                p_dst_val[i] = (int32_t)p_tmp[0];
                p_dst_idx[i] = p_src_idx_tmp[p_tmp[1]];
            }
        }
        break;
        default:
            break;
    }

    return ret;
}

#endif