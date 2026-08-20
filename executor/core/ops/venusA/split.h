#ifndef _SPLIT_LUNA_H_
#define _SPLIT_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/type_switch.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_basic_math.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Split a tensor along a specified axis into multiple tensors
 * @param X Input tensor to be split
 * @param tensors Output tensor array
 * @param attrs Split attributes containing axis and split dimensions
 * @return Execution status
 */
static int32_t split_copy_bytes(void *dst, const void *src, uint32_t size,
                                bool dst_psram, bool src_psram) {
    if (size == 0 || dst == src) return T_SUCCESS;
    if (src_psram) {
        return API_LIB(psrammemcpy_i8o8)((int8_t *)dst, (int8_t *)src, size);
    }
    if (dst_psram) {
        opi_psram_cpy_out(dst, (void *)src, size);
        return T_SUCCESS;
    }
    return API_LIB(memcpy_i8o8)((int8_t *)dst, (int8_t *)src, size);
}

int32_t split_venus(tTensor *X, tTensor **tensors, SplitAttrs *attrs) {
    // Adjust axis if negative
    if (attrs->axis < 0) {
        attrs->axis += X->shape_.ndim_;
    }
#if THINKER_PARAM_CHECK
if (attrs->axis >= X->shape_.ndim_) {
    return (T_ERR_INVALID_PARA);
}
#endif

    // Calculate dimensions
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
#if THINKER_PARAM_CHECK
if (X->dtype_ != out->dtype_) {
    return (T_ERR_INVALID_DATATYPE);
}
#endif

        for (int32_t i = 0; i < leading; ++i) {
            int8_t *idst = (int8_t *)out->dptr_ +
                           i * attrs->split[n] * stride * out->byte_;
            int8_t *isrc = (int8_t *)X->dptr_ +
                           (i * middle + offset) * stride * X->byte_;
            THINKER_RET_CHECK(split_copy_bytes(
                idst, isrc, attrs->split[n] * stride * out->byte_,
                out->mem_.type_ == 1, X->mem_.type_ == 1),
                "split_copy_bytes");
        }
        offset += attrs->split[n];
    }

    return T_SUCCESS;
}

#endif
