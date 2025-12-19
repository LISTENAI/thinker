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
int32_t requant_luna(tTensor* X, tTensor* Y) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Validate input data type
    if (X->dtype_ != Int8) 
        return T_ERR_INVALID_DATATYPE;

    size_t size = getTensorSize(X);
    int8_t* input = (int8_t*)X->dptr_;

    int32_t src_bits = (X->byte_) * 8;
    int32_t dst_bits = (Y->byte_) * 8;
    int32_t q_x = (int32_t)X->scale_;
    int32_t q_y = (int32_t)Y->scale_;

    // Up-sampling: destination bits > source bits
    if (dst_bits > src_bits) 
    {
        if (32 == dst_bits) {
            int32_t* output = (int32_t*)Y->dptr_;
            for (int32_t i = 0; i < size; ++i) {
                output[i] = input[i] << (q_y - q_x);
            }
        } 
        else if (16 == dst_bits) {
            int16_t* output = (int16_t*)Y->dptr_;
            for (int32_t i = 0; i < size; ++i) {
                output[i] = input[i] << (q_y - q_x);
            }
        }
        ret = T_SUCCESS;
    }
    // Same bit width: perform scaling operation
    else if(dst_bits == src_bits)
    {
        int8_t* output = (int8_t*)Y->dptr_;
        int scale = q_y - q_x > 0 ? 1 << (q_y - q_x) : 1;
        int shift = q_x - q_y > 0 ? q_x - q_y : 0;
        ret = API_LIB(scale_i8i8o8)(input, scale, output, size, shift);
    }
    // Down-sampling: not supported
    else{
        return T_ERR_FAIL;
    }

    return ret;
}

#endif