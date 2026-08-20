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
int32_t iqtanh(tTensor *X, tTensor *Y) {
    const int32_t Q_INPUT = 27;
    const int32_t Q_OUTPUT = 7;
    int32_t *src = (int32_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    uint32_t size = getTensorSize(X);

    #if THINKER_PARAM_CHECK
    if (X->dtype_ != Int32 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (X->scale_ != Q_INPUT || Y->scale_ != Q_OUTPUT) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif
    return API_LIB(tanh_i32o8)(src, dst, size);
}

#endif
