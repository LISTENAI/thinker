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
 * @brief Quantized hyperbolic tangent activation function implementation
 * @param X Input tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqtanh(tTensor *X, tTensor *Y) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Quantization parameters
    const int32_t Q_INPUT = 27;
    const int32_t Q_OUTPUT = 7;
    int32_t x_q = X->scale_;
    int32_t y_q = Y->scale_;
    
    // Pointers to tensor data
    int32_t *src = (int32_t *)X->dptr_;
    uint32_t size = getTensorSize(X);
    
    // Adjust input data to match quantization requirements
    if (x_q != Q_INPUT) {
        ret = API_LIB(scale_i32i32o32)(src, 1, src, size, x_q - Q_INPUT);
    }
    
    // Apply tanh activation based on output data type
    switch (Y->dtype_) {
        case Int8: {
            int8_t *dst = (int8_t *)Y->dptr_;
            ret = API_LIB(tanh_i32o8)(src, dst, size);
            break;
        }
        case Int16: {
            int16_t *dst = (int16_t *)Y->dptr_;
            ret = API_LIB(tanh_i32o16)(src, dst, size);
            break;
        }
        case Int32: {
            int32_t *dst = (int32_t *)Y->dptr_;
            ret = API_LIB(tanh_i32o32)(src, dst, size);
            break;
        }
        default:
            return T_ERR_INVALID_DATATYPE;
    }
    
    return ret;
}

#endif  // _TANH_LUNA_H_