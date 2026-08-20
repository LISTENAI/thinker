#ifndef _SPLIT_LUNA_H_
#define _SPLIT_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/type_switch.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#if !(defined(WIN32) || defined(linux))
#pragma clang optimize off
#endif

/**
 * @brief Split tensor along specified axis into multiple tensors
 * @param X Input tensor
 * @param tensors Array of output tensors
 * @param attrs Slice attributes containing axis and split dimensions
 * @return int32_t Operation status
 */
int32_t split_venus(tTensor *X, tTensor **tensors, SplitAttrs *attrs) {
    #if THINKER_PARAM_CHECK
    if (X == NULL || tensors == NULL || attrs == NULL || X->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    if (X->shape_.ndim_ == 0 || attrs->dims <= 0 || attrs->dims > 8) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    // Adjust negative axis
    if (attrs->axis < 0) {
        attrs->axis += X->shape_.ndim_;
    }
    // Check if axis is valid
    #if THINKER_PARAM_CHECK
    if (attrs->axis < 0 || attrs->axis >= X->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }
    #endif

    int32_t split_total = 0;
    for (int32_t n = 0; n < attrs->dims; ++n) {
        tTensor *out = tensors[n + 1];
        #if THINKER_PARAM_CHECK
        if (out == NULL || out->dptr_ == 0 || attrs->split[n] <= 0) {
            return (T_ERR_INVALID_PARA);
        }
        if (out->dtype_ != X->dtype_ || out->byte_ != X->byte_) {
            return (T_ERR_INVALID_DATATYPE);
        }
        if (out->shape_.ndim_ != X->shape_.ndim_) {
            return (T_ERR_INVALID_DATA);
        }
        #endif
        for (int32_t i = 0; i < X->shape_.ndim_; ++i) {
            int32_t expected = i == attrs->axis ? attrs->split[n] : X->shape_.dims_[i];
            #if THINKER_PARAM_CHECK
            if (out->shape_.dims_[i] != expected) {
                return (T_ERR_INVALID_DATA);
            }
            #endif
        }
        split_total += attrs->split[n];
    }
    #if THINKER_PARAM_CHECK
    if (split_total != X->shape_.dims_[attrs->axis]) {
        return (T_ERR_INVALID_DATA);
    }
    #endif

    // Calculate leading, middle, and stride dimensions
    int32_t leading = 1, middle = 1, stride = 1;
    int32_t index = 0;
    for (; index < attrs->axis; ++index) {
        leading *= X->shape_.dims_[index];
    }
    middle = X->shape_.dims_[index++];
    for (; index < X->shape_.ndim_; ++index) {
        stride *= X->shape_.dims_[index];
    }

    int32_t offset = 0;
    for (int32_t n = 0; n < attrs->dims; ++n) {
        const tTensor *out = tensors[n + 1];
        for (int32_t i = 0; i < leading; ++i) {
            int8_t *idst = (int8_t *)out->dptr_ +
                            i * attrs->split[n] * stride * out->byte_;
            const int8_t *isrc = (const int8_t *)X->dptr_ +
                                  (i * middle + offset) * stride * X->byte_;
            memcpy(idst, isrc, attrs->split[n] * stride * out->byte_);
        }
        offset += attrs->split[n];
    }

    return T_SUCCESS;
}

#if !(defined(WIN32) || defined(linux))
#pragma clang optimize on
#endif
#endif
