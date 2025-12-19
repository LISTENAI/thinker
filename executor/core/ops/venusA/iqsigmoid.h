#ifndef _SIGMOID_LUNA_H_
#define _SIGMOID_LUNA_H_

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
 * @brief Quantized sigmoid activation function implementation
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t iqsigmoid(tTensor *X, tTensor *Y, tTensor *Temp) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    uint32_t input_size = getTensorSize(X);
    uint32_t workspace_size = getTensorSize(Temp);

    // Check if temporary workspace is sufficient
    if (workspace_size < input_size * 4) {
        return T_ERR_NO_WORKSPACE;
    }

    // Quantization parameters
    const int32_t Q_INPUT = 27;
    const int32_t Q_OUTPUT = 7;
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;

    // Pointers to tensor data
    int8_t *src = (int8_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    int16_t *tmp = (int16_t *)Temp->dptr_;
    int32_t *tmp1 = (int32_t *)(tmp + input_size);

    // Quantization shift
    uint32_t shift = Q_INPUT - x_q;

    // Perform quantized sigmoid computation
    ret = API_LIB(scale_i8i8o16)(src, 1, tmp, input_size, 0);  // Convert Int8 to Int16
    ret |= API_LIB(scale_i16i16o32)(tmp, 1UL << shift, tmp1, input_size, 0);  // Scale to Int32
    ret |= API_LIB(sigmoid_i32o8)(tmp1, dst, input_size);  // Apply sigmoid activation and convert to Int8

    return ret;
}

#endif  // _SIGMOID_LUNA_H_