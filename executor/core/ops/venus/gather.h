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
 * @brief Process indices for gather operation
 * @param indices Pointer to indices data (int32_t type)
 * @param ndim Number of elements in indices
 * @param middle Middle dimension size
 * @param tail Tail dimension size
 * @param input Pointer to input data
 * @param output Pointer to output data
 * @param byte_size Byte size per element
 * @param leading Leading dimension size
 * @param mem_type_X Memory type of input tensor
 * @param mem_type_Y Memory type of output tensor
 */
static void process_indices(int32_t *indices, int32_t ndim, int32_t middle, int32_t tail,
                           int8_t *input, int8_t *output, int32_t byte_size, int32_t leading,
                           int32_t mem_type_X, int32_t mem_type_Y) {
    for (int32_t l = 0; l < leading; ++l) {
        for (int32_t j = 0; j < ndim; ++j) {
            int32_t idx = indices[j];
            if (idx == -1) {
                idx = middle - 1;
            }
            if (mem_type_X != 2 || mem_type_Y != 2) {
                memcpy(output + (l * ndim * tail + j * tail) * byte_size,
                       input + (l * middle * tail + idx * tail) * byte_size,
                       byte_size * tail);
            } else {
                API_LIB(memcpy)(output + (l * ndim * tail + j * tail) * byte_size,
                               input + (l * middle * tail + idx * tail) * byte_size,
                               byte_size * tail);
            }
        }
    }
}

/**
 * @brief Perform gather operation on input tensor based on indices
 * @param X Pointer to input tensor
 * @param indices Pointer to indices tensor
 * @param Y Pointer to output tensor
 * @param attr Pointer to GatherAttrs containing gather attributes
 * @return int32_t Return status (T_SUCCESS if successful)
 */
int32_t gather_luna(tTensor *X, tTensor *indices, tTensor *Y, GatherAttrs *attr) {
    int32_t axis = attr->axis;
    axis = (axis < 0) ? (X->shape_.ndim_ + axis) : axis;

    // Calculate the total number of elements in indices
    int32_t ndim = 1;
    for (int32_t i = 0; i < indices->shape_.ndim_; ++i) {
        ndim *= indices->shape_.dims_[i];
    }
    if (ndim == 0) {
        ndim = 1;
    }

    // Calculate tensor dimensions
    int32_t leading = 1;
    int32_t dim_index = 0;
    for (; dim_index < axis; ++dim_index) {
        leading *= X->shape_.dims_[dim_index];
    }
    int32_t middle = X->shape_.dims_[dim_index++];
    int32_t tail = 1;
    for (; dim_index < X->shape_.ndim_; ++dim_index) {
        tail *= X->shape_.dims_[dim_index];
    }

    int8_t *input = (int8_t *)X->dptr_;
    int8_t *output = (int8_t *)Y->dptr_;

    // Convert indices to int32_t for processing
    int32_t *indices_data = NULL;
    if (indices->dtype_ == Int64) {
        int64_t *indices64 = (int64_t *)indices->dptr_;
        indices_data = (int32_t *)malloc(ndim * sizeof(int32_t));
        for (int32_t i = 0; i < ndim; ++i) {
            indices_data[i] = (int32_t)indices64[i];
        }
    } else if (indices->dtype_ == Int32) {
        indices_data = (int32_t *)indices->dptr_;
    }

    if (indices_data != NULL) {
        process_indices(indices_data, ndim, middle, tail, input, output, X->byte_, leading,
                       X->mem_.type_, Y->mem_.type_);
        if (indices->dtype_ == Int64) {
            free(indices_data);
        }
    }

    return T_SUCCESS;
}

#endif  // _GATHER_LUNA_H_