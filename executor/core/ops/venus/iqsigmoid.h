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
    int32_t ret = T_ERR_FAIL;
    const int32_t Q_INPUT = 11;  // Input quantization bits
    const int32_t Q_OUTPUT = 7;  // Output quantization bits
    
#ifdef PARAM_CHECK
    /*Check the storage locations for input and output, 
    as it is unnecessary because they have already been limited in tpacker.*/
    if ((X->mem_.type_ != 2) && (Y->mem_.type_ != 2))
        return T_ERR_INVALID_DATATYPE;
    if ((X->dtype_ != Int16) || ((Y->dtype_ != Int8) && (Y->dtype_ != Int16))) {
        return T_ERR_INVALID_DATATYPE;
    }
#endif

    int32_t x_q = (int32_t)X->scale_;
    int16_t *src = (int16_t *)X->dptr_;
    int32_t shift = x_q - Q_INPUT;
    
    uint32_t input_size = getTensorSize(X);
    uint32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;

    if (shift > 0) {
#ifdef PARAM_CHECK        
        if (workspace_size < input_size * 2) {
            return T_ERR_NO_WORKSPACE;
        }
#endif
        int16_t *dst_temp = (int16_t *)Temp->dptr_;
        ret = API_LIB(scale_q15_int16)(src, 1, dst_temp, input_size, x_q - Q_INPUT);
        
        if (Y->dtype_ == Int8) {
            ret |= API_LIB(sigmoid_int8)(dst_temp, (int8_t *)Y->dptr_, input_size);
        } else {
            ret |= API_LIB(sigmoid)(dst_temp, (int16_t *)Y->dptr_, input_size);
        }
    } 
    else {
        if (Y->dtype_ == Int8) {
            ret = API_LIB(sigmoid_int8)(src, (int8_t *)Y->dptr_, input_size);
        } else {
            ret = API_LIB(sigmoid)(src, (int16_t *)Y->dptr_, input_size);
        }
    }
    
    return ret;
}

#endif