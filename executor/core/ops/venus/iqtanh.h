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
    const int32_t Q_OUTPUT = 15;

    // Get tensor pointers and size
    int16_t *src = (int16_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    uint32_t size = getTensorSize(X);

    // Check if input quantization matches expected
    if (X->scale_ != Q_INPUT) {
        // Scale input to match Q_INPUT quantization
        API_LIB(scale_q15_int16)(src, 1, src, size, X->scale_ - Q_INPUT);
    }

    // Compute tanh and store result
    int32_t ret = API_LIB(tanh_int8)(src, dst, size);

    return ret;
}

#endif