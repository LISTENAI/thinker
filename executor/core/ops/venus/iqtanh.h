#ifndef _TANH_LUNA_H_
#define _TANH_LUNA_H_

#include <math.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"

/**
 * @brief Integer quantized hyperbolic tangent activation function
 * @param X Input tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqtanh(tTensor *X, tTensor *Y) {
    // Quantization parameters
    const int32_t Q_INPUT = 11;
    const int32_t Q_OUTPUT = 7;

    // Get tensor pointers and size
    int16_t *src = (int16_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    uint32_t size = getTensorSize(X);

    #if THINKER_PARAM_CHECK
    if (X->dtype_ != Int16 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
    if (X->scale_ != Q_INPUT || Y->scale_ != Q_OUTPUT) {
        return (T_ERR_INVALID_PARA);
    }
    if (X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
        return (T_ERR_NO_SUPPORT_OP);
    }
    #endif

    // Compute tanh and store result
    THINKER_RET_CHECK(API_LIB(tanh_int8)(src, dst, size), "luna_tanh_int8");

    return T_SUCCESS;
}

#endif
