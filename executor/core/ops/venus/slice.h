#ifndef __SLICE_H__
#define __SLICE_H__

#include <string.h>
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

/**
 * @brief Perform tensor slicing operation
 * @param X Input tensor
 * @param begin Start index of the slice
 * @param end End index of the slice (exclusive)
 * @param axis Axis along which to slice
 * @param step Step size for slicing
 * @param Y Output tensor
 * @return tStatus Operation status
 */
tStatus slice_luna(tTensor* X, int32_t begin, int32_t end, int32_t axis,
                   int32_t step, tTensor* Y) {
    #if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    if (step != 1) {
        return (T_ERR_NO_IMPLEMENTED);
    }
    #endif
    int32_t n_dims = X->shape_.ndim_;
    #if THINKER_PARAM_CHECK
    if (n_dims <= 0 || axis < -n_dims || axis >= n_dims ||
                        Y->shape_.ndim_ != n_dims) {
        return (T_ERR_INVALID_PARA);
    }
    if (X->dtype_ != Y->dtype_ || X->byte_ != Y->byte_) {
        return (T_ERR_INVALID_DATATYPE);
    }
    #endif
    int32_t real_axis = axis < 0 ? axis + n_dims : axis;
    int32_t axis_size = X->shape_.dims_[real_axis];
    #if THINKER_PARAM_CHECK
    if (axis_size <= 0) {
        return (T_ERR_INVALID_DATA);
    }
    #endif
    int32_t real_begin = begin < 0 ? begin + axis_size : begin;
    int32_t real_end = end < 0 ? end + axis_size : end;
    if (real_begin < 0) real_begin = 0;
    if (real_begin > axis_size) real_begin = axis_size;
    if (real_end < 0) real_end = 0;
    if (real_end > axis_size) real_end = axis_size;
    int32_t slice_size = real_end > real_begin ? real_end - real_begin : 0;
    for (int32_t i = 0; i < n_dims; ++i) {
        int32_t expected = i == real_axis ? slice_size : X->shape_.dims_[i];
        #if THINKER_PARAM_CHECK
        if (Y->shape_.dims_[i] != expected) {
            return (T_ERR_INVALID_DATA);
        }
        #endif
    }

    // Calculate the number of elements to copy
    int32_t num_elements = 1;
    for (size_t i = 0; i < Y->shape_.ndim_; ++i) {
        num_elements *= Y->shape_.dims_[i];
    }

    // If slicing along the first axis, perform direct memcpy
    if (real_axis == 0) {
        int32_t start = real_begin;
        for (int32_t i = 1; i < X->shape_.ndim_; ++i) {
            start *= X->shape_.dims_[i];
        }
        memcpy((int8_t*)Y->dptr_, (int8_t*)X->dptr_ + start * X->byte_,
               num_elements * Y->byte_);
        return T_SUCCESS;
    }

    // Calculate leading and trailing dimensions
    int32_t leading = 1;
    for (int32_t i = 0; i < real_axis; ++i) {
        leading *= X->shape_.dims_[i];
    }

    int32_t trailing = 1;
    for (int32_t i = real_axis + 1; i < X->shape_.ndim_; ++i) {
        trailing *= X->shape_.dims_[i];
    }

    int32_t mid = X->shape_.dims_[real_axis];
    int32_t i_mt = mid * trailing;
    int32_t o_mt = Y->shape_.dims_[real_axis] * trailing;
    int32_t offset = real_begin * trailing;

    // Copy data according to the byte size
    if (X->byte_ == 1) {
        for (int32_t l = 0; l < leading; ++l) {
            int32_t i_lmt_this = l * i_mt + offset;
            int32_t o_lmt_this = l * o_mt;
            memcpy((int8_t*)Y->dptr_ + o_lmt_this,
                   (int8_t*)X->dptr_ + i_lmt_this,
                   o_mt);
        }
    } else {
        for (int32_t l = 0; l < leading; ++l) {
            int32_t i_lmt_this = l * i_mt + offset;
            int32_t o_lmt_this = l * o_mt;
            memcpy((int8_t*)Y->dptr_ + o_lmt_this * Y->byte_,
                   (int8_t*)X->dptr_ + i_lmt_this * X->byte_,
                   o_mt * Y->byte_);
        }
    }

    return T_SUCCESS;
}

#endif
