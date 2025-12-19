#ifndef _ARGMAX_LUNA_H_
#define _ARGMAX_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_basic_math.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

#include "thinker_status.h"

/**
 * @brief Find maximum values and their indices along specified axis
 * @param X Input tensor
 * @param Y Output tensor (containing values and indices)
 * @param workspace Workspace buffer
 * @param attrs ArgMax attributes including axis
 * @return Operation status
 */
int32_t argmax_luna(tTensor *X, tTensor *Y, tTensor *workspace, ArgMaxAttrs *attrs) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    int32_t axis = attrs->axis;
    int32_t n_dims = X->shape_.ndim_;
    int32_t once_size = axis == -1 ? X->shape_.dims_[n_dims - 1] : X->shape_.dims_[axis];
    
    int32_t leading = 1;
    int32_t *p_tmp = (int32_t *)workspace->dptr_;

    // Calculate leading dimension for batch processing
    if (-1 == axis || (n_dims - 1) == axis) {
        for (int i = 0; i < (n_dims - 1); i++) {
            leading *= X->shape_.dims_[i];
        }
    }  

    // Process each batch element
    int32_t *p_dst_val = (int32_t *)Y->dptr_;
    int32_t *p_dst_idx = (int32_t *)Y->dptr_ + leading;
    
    for (int32_t i = 0; i < leading; i++) {
        int8_t *p_src = (int8_t *)X->dptr_ + i * once_size;
        ret = API_LIB(max_q7)(p_src, p_tmp, once_size);
        p_dst_val[i] = (int32_t)p_tmp[0];  // Store maximum value
        p_dst_idx[i] = (int32_t)p_tmp[1];  // Store index of maximum value
    }

    return ret;
}

#endif