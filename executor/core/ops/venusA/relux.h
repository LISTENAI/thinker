#ifndef __RELUX_H__
#define __RELUX_H__

#include "c_api/thinker_define.h"
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

#include "thinker_status.h"

/**
 * @brief Perform ReLUX activation operation
 * @param X Input tensor
 * @param Y Output tensor
 * @param attrs ReLUX attributes containing threshold and shift
 * @return Execution status
 */
tStatus relux_luna(tTensor *X, tTensor *Y, ReluxAttrs *attrs) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    // Check if input and output are in PSRAM
    if (X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
        return T_ERR_NO_IMPLEMENTED;
    }

    // Ensure output data type is Int8
    if (Y->dtype_ != Int8) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Get ReLUX parameters
    int32_t threshold = attrs->threshold;
    int32_t shift = attrs->shift;
    uint32_t size = getTensorSize(X);

    // Execute ReLUX operation based on input data type
    switch (X->dtype_) {
        case Int8:
            ret = API_LIB(relux_i8o8)((int8_t *)X->dptr_, threshold, (int8_t *)Y->dptr_, size, shift);
            break;
        case Int16:
            ret = API_LIB(relux_i16o8)((int16_t *)X->dptr_, threshold, (int8_t *)Y->dptr_, size, shift);
            break;
        case Int32:
            ret = API_LIB(relux_i32o8)((int32_t *)X->dptr_, threshold, (int8_t *)Y->dptr_, size, shift);
            break;
        default:
            return T_ERR_INVALID_DATATYPE;
    }

    return ret ? T_SUCCESS : T_ERR_FAIL;
}

#endif