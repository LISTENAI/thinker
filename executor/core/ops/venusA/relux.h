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
#if THINKER_RUNTIME_CHECK
if (X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
    return (T_ERR_NO_IMPLEMENTED);
}

if (X->dtype_ != Int8 || Y->dtype_ != Int8) {
    return (T_ERR_INVALID_DATATYPE);
}
#endif

    // Get ReLUX parameters
    int8_t threshold = attrs->threshold;
    int32_t shift = attrs->shift;
    #if THINKER_PARAM_CHECK
    if (attrs->threshold < -128 || attrs->threshold > 127) {
        return (T_ERR_INVALID_PARA);
    }

if (shift < 0 || shift > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif
    uint32_t size = getTensorSize(X);

    // Execute ReLUX operation based on input data type
    if (X->dtype_ == Int8)
        THINKER_RET_CHECK(API_LIB(relux_i8o8)((int8_t *)X->dptr_, threshold, (int8_t *)Y->dptr_, size, shift), "luna_relux_i8o8");
    else if (X->dtype_ == Int16)
        THINKER_RET_CHECK(API_LIB(relux_i16o8)((int16_t *)X->dptr_, threshold, (int8_t *)Y->dptr_, size, shift), "luna_relux_i16o8");
    else if (X->dtype_ == Int32)
        THINKER_RET_CHECK(API_LIB(relux_i32o8)((int32_t *)X->dptr_, threshold, (int8_t *)Y->dptr_, size, shift), "luna_relux_i32o8");
    else
        return T_ERR_INVALID_DATATYPE;

    return T_SUCCESS;
}

#endif
