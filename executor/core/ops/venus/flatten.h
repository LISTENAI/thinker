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
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Flatten a tensor by copying data from input to output
 * @param X Pointer to input tensor
 * @param Y Pointer to output tensor
 * @param attr Pointer to FlattenAttrs (unused in this implementation)
 * @return int32_t Return status (T_SUCCESS if successful)
 */
int32_t flatten_luna(tTensor *X, tTensor *Y, FlattenAttrs *attr) {
    int8_t *input = (int8_t *)X->dptr_;
    int8_t *output = (int8_t *)Y->dptr_;

    // Check if memory type is NNBLAS specific (type 2)
    if ((X->mem_.type_ == 2) && (Y->mem_.type_ == 2)) {
        if (input != output) {
            size_t size = getTensorSize(X);
            API_LIB(memcpy)(output, input, X->byte_ * size);  // Use NNBLAS memcpy
        }
    } else {
        if (input != output) {
            size_t size = getTensorSize(X);
            memcpy(output, input, X->byte_ * size);  // Use standard memcpy
        }
    }

    return T_SUCCESS;
}

#endif  // _FLATTEN_LUNA_H_