#ifndef _GATHER_LUNA_H_
#define _GATHER_LUNA_H_

#include <math.h>
#include "core/operator_attrs.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

static int32_t gather_copy_bytes(void *dst, const void *src, uint32_t size,
                                 bool dst_psram, bool src_psram) {
    if (size == 0 || dst == src) return T_SUCCESS;
    if (dst_psram) {
        opi_psram_cpy_out(dst, (void *)src, size);
        return T_SUCCESS;
    }
    return API_LIB(memcpy_i8o8)((int8_t *)dst, (int8_t *)src, size);
}

/**
 * @brief Perform gather operation along specified axis
 * @param X Input tensor
 * @param indices Index tensor
 * @param Y Output tensor
 * @param attr Operation attributes
 * @return int32_t Operation status
 */
int32_t gather_luna(tTensor *X, tTensor *indices, tTensor *Y, GatherAttrs *attr) {
    int32_t axis = attr->axis < 0 ? X->shape_.ndim_ + attr->axis : attr->axis;
    #if THINKER_PARAM_CHECK
    if (axis < 0 || axis >= X->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }

    if (indices->dtype_ != Int32 && indices->dtype_ != Int64) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif

    // Calculate number of elements in indices
    size_t ndim = 1;
    for (int32_t i = 0; i < indices->shape_.ndim_; ++i)
        ndim *= indices->shape_.dims_[i];
    if (ndim == 0) return T_SUCCESS;
    #if THINKER_PARAM_CHECK
    if (ndim > INT32_MAX) {
        return (T_ERR_INVALID_DATA);
    }

    if (Y->shape_.ndim_ != X->shape_.ndim_ - 1 + indices->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }

    for (int32_t i = 0; i < axis; ++i) {
        if (Y->shape_.dims_[i] != X->shape_.dims_[i]) {
            return (T_ERR_INVALID_PARA);
        }
    }
    for (int32_t i = 0; i < indices->shape_.ndim_; ++i) {
        if (Y->shape_.dims_[axis + i] != indices->shape_.dims_[i]) {
            return (T_ERR_INVALID_PARA);
        }
    }
    for (int32_t i = axis + 1; i < X->shape_.ndim_; ++i) {
        if (Y->shape_.dims_[i - 1 + indices->shape_.ndim_] != X->shape_.dims_[i]) {
            return (T_ERR_INVALID_PARA);
        }
    }
#endif

    // Calculate tensor dimensions
    int32_t leading = 1, i = 0;
    for (; i < axis; ++i) leading *= X->shape_.dims_[i];
    int32_t middle = X->shape_.dims_[i++];
    int32_t tail = 1;
    for (; i < X->shape_.ndim_; ++i) tail *= X->shape_.dims_[i];

    int8_t *input = (int8_t *)X->dptr_;
    int8_t *output = (int8_t *)Y->dptr_;

    if (indices->dtype_ == Int64)
    {
        int64_t *index = (int64_t *)indices->dptr_;
#if THINKER_PARAM_CHECK
        for (size_t j = 0; j < ndim; ++j) {
            if (index[j] < -1 || index[j] >= middle) {
                return (T_ERR_INVALID_PARA);
            }
        }
#endif
        if (X->dtype_ == Int4)
        {
            #if THINKER_PARAM_CHECK
            if (Y->dtype_ != Int8 || (tail & 1)) {
                return (T_ERR_INVALID_DATATYPE);
            }
#endif
            for (int32_t l = 0; l < leading; ++l)
                for (size_t j = 0; j < ndim; ++j)
                {
                    int64_t idx = index[j] == -1 ? X->shape_.dims_[axis] - 1 : index[j];
                    convert_4bitto8bit(output + l * ndim * tail + j * tail,
                                      input + l * middle * (tail/2) + idx * (tail/2), tail);
                }
        }
        else
        {
        for (int32_t l = 0; l < leading; ++l)
            for (size_t j = 0; j < ndim; ++j)
            {
                int64_t idx = index[j] == -1 ? X->shape_.dims_[axis] - 1 : index[j];
                THINKER_RET_CHECK(gather_copy_bytes(
                    output + (l * ndim * tail + j * tail) * X->byte_,
                    input + (l * middle * tail + idx * tail) * X->byte_,
                    X->byte_ * tail, Y->mem_.type_ == 1, X->mem_.type_ == 1),
                    "gather_copy_bytes");
                }
            }
    }
    else if (indices->dtype_ == Int32)
    {
        int32_t *index = (int32_t *)indices->dptr_;
#if THINKER_PARAM_CHECK
        for (size_t j = 0; j < ndim; ++j) {
            if (index[j] < -1 || index[j] >= middle) {
                return (T_ERR_INVALID_PARA);
            }
        }
#endif
        if (X->dtype_ == Int4) {
            #if THINKER_PARAM_CHECK
            if (Y->dtype_ != Int8 || (tail & 1)) {
                return (T_ERR_INVALID_DATATYPE);
            }
#endif
            for (int32_t l = 0; l < leading; ++l)
                for (size_t j = 0; j < ndim; ++j) {
                    int32_t idx = index[j] == -1 ? middle - 1 : index[j];
                    convert_4bitto8bit(output + l * ndim * tail + j * tail,
                                      input + l * middle * (tail / 2) +
                                          idx * (tail / 2),
                                      tail);
                }
        } else for (int32_t l = 0; l < leading; ++l)
            for (size_t j = 0; j < ndim; ++j)
            {
                int32_t idx = index[j] == -1 ? X->shape_.dims_[axis] - 1 : index[j];
                THINKER_RET_CHECK(gather_copy_bytes(
                    output + (l * ndim * tail + j * tail) * X->byte_,
                    input + (l * middle * tail + idx * tail) * X->byte_,
                    X->byte_ * tail, Y->mem_.type_ == 1, X->mem_.type_ == 1),
                    "gather_copy_bytes");
            }
    }
    else
        return T_ERR_INVALID_DATATYPE;

    if (X->dtype_ == Int4 && Y->mem_.type_ == 1) {
        thinker_psram_write_complete((void *)Y->dptr_, getTensorDataSize(Y));
    }

    return T_SUCCESS;
}
#endif  //_GATHER_LUNA_H_
