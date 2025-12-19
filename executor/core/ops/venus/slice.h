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
    int32_t real_axis = (axis + X->shape_.ndim_) % X->shape_.ndim_;
    int32_t real_begin = begin;
    if (begin < 0) {
        real_begin += X->shape_.dims_[real_axis];
    }
    real_begin = real_begin % X->shape_.dims_[real_axis];

    // Calculate the number of elements to copy
    int32_t num_elements = 0;
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