#ifndef _TANH_LUNA_H_
#define _TANH_LUNA_H_

#include <math.h>
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
 * @brief Integer Quantized Hyperbolic Tangent operation
 * @param X Input tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqtanh(tTensor *X, tTensor *Y) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    const int32_t Q_INPUT = 27;
    const int32_t Q_OUTPUT = 7;
    int32_t x_q = X->scale_;
    int32_t y_q = Y->scale_;

    int32_t *src = (int32_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    uint32_t size = getTensorSize(X);

    if (Q_INPUT != x_q) {
        ret = API_LIB(scale_i32i32o32)(src, 1, src, size, x_q - Q_INPUT);
    }
    ret = API_LIB(tanh_i32o8)(src, dst, size);

    return ret;
}

#endif