#ifndef _SIGMOID_LUNA_H_
#define _SIGMOID_LUNA_H_

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
 * @brief Integer Quantized Sigmoid operation
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t iqsigmoid(tTensor *X, tTensor *Y, tTensor *Temp) 
{
    int32_t ret = -1;
    uint32_t input_size = getTensorSize(X);
    uint32_t workspace_size = getTensorSize(Temp);

    if (workspace_size < input_size * 4) {
        return T_ERR_NO_WORKSPACE;
    }

    const int32_t Q_INPUT = 27;
    const int32_t Q_OUTPUT = 7;
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;

    int8_t *src = (int8_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    int32_t *tmp = (int32_t *)Temp->dptr_;

    uint32_t shift = Q_INPUT - x_q;

    ret = API_LIB(scale_i8i8o32)(src, 1, tmp, input_size, 0);
    ret = API_LIB(scale_i32i32o32)(tmp, 1UL << shift, tmp, input_size, 0);
    ret |= API_LIB(sigmoid_i32o8)(tmp, dst, input_size);

    return ret;
}

#endif