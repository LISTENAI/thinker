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
    int32_t axis = attrs->axis;
    int32_t n_dims = X->shape_.ndim_;
    int32_t once_size = (axis == -1) ? X->shape_.dims_[n_dims - 1] : X->shape_.dims_[axis];
    int32_t leading = 1;
    #if THINKER_PARAM_CHECK
    if ((X->dtype_ != Int8 && X->dtype_ != Int16 && X->dtype_ != Int32 &&
                         X->dtype_ != Float32) ||
                        Y->dtype_ != Int32) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
    int32_t *p_tmp = (int32_t *)work_space->dptr_;

    // Calculate leading dimensions
    if (axis == -1 || axis == n_dims - 1) {
        for (int i = 0; i < n_dims - 1; ++i) {
            leading *= X->shape_.dims_[i];
        }
    }

#if THINKER_PARAM_CHECK
if (axis != -1 && axis != n_dims - 1) {
    return (T_ERR_INVALID_PARA);
}
#endif

    int32_t *p_dst_val = (int32_t *)Y->dptr_;
    int32_t *p_dst_idx = (int32_t *)Y->dptr_ + leading;

    for (int i = 0; i < leading; ++i) {
        switch (X->dtype_) {
            case Int8: {
                int8_t *p_src = (int8_t *)X->dptr_ + i * once_size;
                THINKER_RET_CHECK(API_LIB(max_i8o32)(p_src, p_tmp, once_size), "luna_max_i8o32");
                break;
            }
            case Int16: {
                int16_t *p_src = (int16_t *)X->dptr_ + i * once_size;
                THINKER_RET_CHECK(API_LIB(max_i16o32)(p_src, p_tmp, once_size), "luna_max_i16o32");
                break;
            }
            case Int32: {
                int32_t *p_src = (int32_t *)X->dptr_ + i * once_size;
                THINKER_RET_CHECK(API_LIB(max_i32o32)(p_src, p_tmp, once_size), "luna_max_i32o32");
                break;
            }
            case Float32: {
                float *p_src = (float *)X->dptr_ + i * once_size;
                int32_t max_idx = 0;
                for (int32_t j = 1; j < once_size; ++j) {
                    if (p_src[j] > p_src[max_idx]) {
                        max_idx = j;
                    }
                }
                p_tmp[0] = (int32_t)p_src[max_idx];
                p_tmp[1] = max_idx;
                break;
            }
            default:
                #if THINKER_PARAM_CHECK
                if (1) {
                    return (T_ERR_INVALID_DATATYPE);
                }
#endif
                break;
        }

        p_dst_val[i] = (int32_t)p_tmp[0];
        p_dst_idx[i] = (int32_t)p_tmp[1];
    }

    if (Y->mem_.type_ == 1) {
        thinker_psram_write_complete((void *)Y->dptr_, getTensorDataSize(Y));
    }

    return T_SUCCESS;
}

#endif
