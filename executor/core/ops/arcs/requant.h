#ifndef __REQUANT_H__
#define __REQUANT_H__

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Requantization operation for integer tensors
 * @param X Input tensor (must be Int8)
 * @param Y Output tensor
 * @return Operation result status
 */
int32_t requant_luna(tTensor* X, tTensor* Y) {    
    // Validate input data type
    #if THINKER_PARAM_CHECK
    if (X->dtype_ != Int8 || (Y->dtype_ != Int8 && Y->dtype_ != Int32)) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif

    size_t size = getTensorSize(X);
    int8_t* input = (int8_t*)X->dptr_;

    int32_t src_bits = (X->byte_) * 8;
    int32_t dst_bits = (Y->byte_) * 8;
    int32_t q_x = (int32_t)X->scale_;
    int32_t q_y = (int32_t)Y->scale_;
    int32_t q_delta = q_y - q_x;

    #if THINKER_RUNTIME_CHECK
    if (src_bits == dst_bits &&
                          (X->mem_.type_ != 2 || Y->mem_.type_ != 2)) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif

    // Up-sampling: destination bits > source bits
    if (dst_bits > src_bits) 
    {
        #if THINKER_PARAM_CHECK
        if (q_delta < 0 || q_delta > 24) {
            return (T_ERR_INVALID_PARA);
        }
#endif
        if (32 == dst_bits) {
            int32_t* output = (int32_t*)Y->dptr_;
            for (int32_t i = 0; i < size; ++i) {
                int64_t value = (int64_t)input[i] * ((int64_t)1 << q_delta);
                output[i] = (int32_t)SATURATE_32BITS(value);
            }
        } 
        else if (16 == dst_bits) {
            int16_t* output = (int16_t*)Y->dptr_;
            for (int32_t i = 0; i < size; ++i) {
                output[i] = input[i] << (q_y - q_x);
            }
        }
    }
    // Same bit width: perform scaling operation
    else if(dst_bits == src_bits)
    {
        int8_t* output = (int8_t*)Y->dptr_;
        #if THINKER_PARAM_CHECK
        if (q_delta > 30 || q_delta < -63) {
            return (T_ERR_INVALID_PARA);
        }
#endif
        int32_t scale = q_delta > 0 ? (int32_t)(1U << q_delta) : 1;
        int32_t shift = q_delta < 0 ? -q_delta : 0;
        THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(input, scale, output, size, shift), "luna_scale_i8i8o8");
    }
    // Down-sampling: not supported
    else{
        return T_ERR_NO_SUPPORT_OP;
    }
    if (Y->mem_.type_ == 1) {
        thinker_psram_write_complete((void *)Y->dptr_, getTensorDataSize(Y));
    }
    return T_SUCCESS;
}

#endif
