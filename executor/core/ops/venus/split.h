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
int32_t split_venus(tTensor *X, tTensor **tensors, SliceAttrs *attrs) {
    // Adjust negative axis
    if (attrs->axis < 0) {
        attrs->axis += X->shape_.ndim_;
    }
    // Check if axis is valid
    if (attrs->axis >= X->shape_.ndim_) {
        return -1;
    }

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
            DATA_TYPE_SWITCH_ALL(X->dtype_, Type, {
                Type *idst = (Type *)(out->dptr_) + i * attrs->split[n] * stride;
                const Type *isrc = (const Type *)(X->dptr_) + i * middle * stride + offset * stride;
                memcpy(idst, isrc, sizeof(Type) * attrs->split[n] * stride);
            });
        }
        offset += attrs->split[n];
    }

    return 0;
}

#if !(defined(WIN32) || defined(linux))
#pragma clang optimize on
#endif
#endif