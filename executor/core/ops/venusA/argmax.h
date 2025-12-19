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
 * @brief Compute the maximum value and its index along a specified axis
 * @param X Input tensor
 * @param Y Output tensor containing maximum values and their indices
 * @param work_space Temporary workspace tensor
 * @param attrs Attributes specifying the axis for computation
 * @return int32_t Operation status
 */
int32_t argmax_luna(tTensor *X, tTensor *Y, tTensor *work_space, ArgMaxAttrs *attrs) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    int32_t axis = attrs->axis;
    int32_t n_dims = X->shape_.ndim_;
    int32_t once_size = (axis == -1) ? X->shape_.dims_[n_dims - 1] : X->shape_.dims_[axis];
    int32_t leading = 1;
    int32_t *p_tmp = (int32_t *)work_space->dptr_;

    // Calculate leading dimensions
    if (axis == -1 || axis == n_dims - 1) {
        for (int i = 0; i < n_dims - 1; ++i) {
            leading *= X->shape_.dims_[i];
        }
    }

    int32_t *p_dst_val = (int32_t *)Y->dptr_;
    int32_t *p_dst_idx = (int32_t *)Y->dptr_ + leading;

    for (int i = 0; i < leading; ++i) {
        switch (X->dtype_) {
            case Int8: {
                int8_t *p_src = (int8_t *)X->dptr_ + i * once_size;
                ret = API_LIB(max_i8o32)(p_src, p_tmp, once_size);
                break;
            }
            case Int16: {
                int16_t *p_src = (int16_t *)X->dptr_ + i * once_size;
                ret = API_LIB(max_i16o32)(p_src, p_tmp, once_size);
                break;
            }
            case Int32: {
                int32_t *p_src = (int32_t *)X->dptr_ + i * once_size;
                ret = API_LIB(max_i32o32)(p_src, p_tmp, once_size);
                break;
            }
            default:
                printf("Unsupported data type.");
                return T_ERR_INVALID_DATATYPE;
        }

        p_dst_val[i] = (int32_t)p_tmp[0];
        p_dst_idx[i] = (int32_t)p_tmp[1];
    }

    return ret;
}

#endif