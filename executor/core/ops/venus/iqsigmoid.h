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
 * @brief Integer quantized sigmoid activation function
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t iqsigmoid(tTensor *X, tTensor *Y, tTensor *Temp) {
    const int32_t Q_INPUT = 11;  // Input quantization bits
    const int32_t Q_OUTPUT = 7;  // Output quantization bits
    int32_t x_q = (int32_t)X->scale_;
    int16_t *src = (int16_t *)X->dptr_;
    int32_t shift = Q_INPUT - x_q;

#if THINKER_PARAM_CHECK
if (X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
    return (T_ERR_NO_SUPPORT_OP);
}
        if (X->dtype_ != Int16 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
    if (shift < -63 || shift > 14) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    
    uint32_t input_size = getTensorSize(X);
    size_t workspace_size = Temp ? getTensorDataSize(Temp) : 0;

    if (shift != 0) {
#if THINKER_RUNTIME_CHECK
if (Temp == NULL || workspace_size < input_size * sizeof(int16_t)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
        int16_t *dst_temp = (int16_t *)Temp->dptr_;
        uint32_t shift1 = shift > 0 ? shift : 0;
        uint32_t shift2 = shift > 0 ? 0 : -shift;
        THINKER_RET_CHECK(API_LIB(scale_q15_int16)(src, 1UL << shift1, dst_temp, input_size, shift2), "luna_scale_q15_int16");
        
        if (Y->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(sigmoid_int8)(dst_temp, (int8_t *)Y->dptr_, input_size), "luna_sigmoid_int8");
        } 
        else {
            THINKER_RET_CHECK(API_LIB(sigmoid)(dst_temp, (int16_t *)Y->dptr_, input_size), "luna_sigmoid");
        }
    } 
    else {
        if (Y->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(sigmoid_int8)(src, (int8_t *)Y->dptr_, input_size), "luna_sigmoid_int8");
        } 
        else {
            THINKER_RET_CHECK(API_LIB(sigmoid)(src, (int16_t *)Y->dptr_, input_size), "luna_sigmoid");
        }
    }
    
    return T_SUCCESS;
}

#endif
