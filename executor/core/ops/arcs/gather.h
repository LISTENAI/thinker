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

/**
 * @brief Perform gather operation along specified axis
 * @param X Input tensor
 * @param indices Index tensor
 * @param Y Output tensor
 * @param attr Operation attributes
 * @return int32_t Operation status
 */
int32_t gather_luna(tTensor *X, tTensor *indices, tTensor *Y, GatherAttrs *attr) 
{
    int32_t ret = T_SUCCESS;
    
    int32_t axis = attr->axis < 0 ? X->shape_.ndim_ + attr->axis : attr->axis;
    
    // Calculate number of elements in indices
    int32_t ndim = 1;
    for (int32_t i = 0; i < indices->shape_.ndim_; ++i)
        ndim *= indices->shape_.dims_[i];
    ndim = ndim == 0 ? 1 : ndim;

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
        if (X->dtype_ == Int4)
        {
            if (Y->dtype_ != Int8) return T_ERR_INVALID_DATATYPE;
            for (int32_t l = 0; l < leading; ++l)
                for (int32_t j = 0; j < ndim; ++j)
                {
                    int64_t idx = index[j] == -1 ? X->shape_.dims_[axis] - 1 : index[j];
                    convert_4bitto8bit(output + l * ndim * tail + j * tail,
                                      input + l * middle * (tail/2) + idx * (tail/2), tail);
                }
            ret = T_SUCCESS;
        }
        else
        {
            for (int32_t l = 0; l < leading; ++l)
                for (int32_t j = 0; j < ndim; ++j)
                {
                    int64_t idx = index[j] == -1 ? X->shape_.dims_[axis] - 1 : index[j];
                    if (Y->mem_.type_ == 2)
                        ret = API_LIB(memcpy_i8o8)(output + (l * ndim * tail + j * tail) * X->byte_,
                                                  input + (l * middle * tail + idx * tail) * X->byte_,
                                                  X->byte_ * tail);
                    else {
                        opi_psram_cpy_out(output + (l * ndim * tail + j * tail) * X->byte_,
                                         input + (l * middle * tail + idx * tail) * X->byte_,
                                         X->byte_ * tail);
                        ret = T_SUCCESS;
                    }
                }
        }
    }
    else if (indices->dtype_ == Int32)
    {
        int32_t *index = (int32_t *)indices->dptr_;
        for (int32_t l = 0; l < leading; ++l)
            for (int32_t j = 0; j < ndim; ++j)
            {
                int32_t idx = index[j] == -1 ? X->shape_.dims_[axis] - 1 : index[j];
                if (Y->mem_.type_ == 2)
                    ret = API_LIB(memcpy_i8o8)(output + (l * ndim * tail + j * tail) * X->byte_,
                                              input + (l * middle * tail + idx * tail) * X->byte_,
                                              X->byte_ * tail);
                else {
                    opi_psram_cpy_out(output + (l * ndim * tail + j * tail) * X->byte_,
                                     input + (l * middle * tail + idx * tail) * X->byte_,
                                     X->byte_ * tail);
                    ret = T_SUCCESS;
                }
            }
    }
    else
        ret = T_ERR_INVALID_DATATYPE;

    return ret;
}
#endif  //_GATHER_LUNA_H_