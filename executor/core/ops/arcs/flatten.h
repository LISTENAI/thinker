#ifndef _FLATTEN_LUNA_H_
#define _FLATTEN_LUNA_H_

#include <math.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_cnn_tools.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Flatten a tensor by copying its data to a contiguous memory space
 * @param X Input tensor
 * @param Y Output tensor (flattened result)
 * @param attr Flatten attributes (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t flatten_luna(tTensor *X, tTensor *Y, FlattenAttrs *attr) {
    int8_t *input = (int8_t *)X->dptr_;
    int8_t *output = (int8_t *)Y->dptr_;

    if (input != output) {
        size_t size = getTensorSize(X) * X->byte_;
        if (Y->mem_.type_ == 2) {
            return API_LIB(memcpy_i8o8)(output, input, size);
        } else {
            opi_psram_cpy_out(output, input, size);
            return T_SUCCESS;
        }
    }
    return T_SUCCESS;
}

#endif  // _FLATTEN_LUNA_H_