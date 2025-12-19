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
 * @brief Requantize tensor data from one quantization format to another
 * @param X Input tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t requant_luna(tTensor* X, tTensor* Y) {
    if (X->dtype_ != Int8) {
        return -1;
    }

    size_t size = getTensorSize(X);
    int8_t* input = (int8_t*)X->dptr_;
    int32_t src_bits = X->byte_ * 8;
    int32_t dst_bits = Y->byte_ * 8;
    int32_t q_x = (int32_t)X->scale_;
    int32_t q_y = (int32_t)Y->scale_;

    if (dst_bits > src_bits) {
        if (dst_bits == 32) {
            int32_t* output = (int32_t*)Y->dptr_;
            for (int32_t i = 0; i < size; ++i) {
                output[i] = input[i] << (q_y - q_x);
            }
        } else if (dst_bits == 16) {
            int16_t* output = (int16_t*)Y->dptr_;
            for (int32_t i = 0; i < size; ++i) {
                output[i] = input[i] << (q_y - q_x);
            }
        }
    } else if (dst_bits == src_bits) {
        if (dst_bits == 32) {
            int32_t* output = (int32_t*)Y->dptr_;
            int32_t* input = (int32_t*)X->dptr_;
            int32_t scale = q_y > q_x ? q_y - q_x : 1 << (q_y - q_x);
            int32_t shift = q_x > q_y ? q_x - q_y : 0;
            API_LIB(scale_q31_int32)(input, scale, output, size, shift);
        } else if (dst_bits == 16) {
            int16_t* output = (int16_t*)Y->dptr_;
            int16_t* input = (int16_t*)X->dptr_;
            int32_t scale = q_y > q_x ? q_y - q_x : 1 << (q_y - q_x);
            int32_t shift = q_x > q_y ? q_x - q_y : 0;
            API_LIB(scale_q15_int16)(input, scale, output, size, shift);
        } else if (dst_bits == 8) {
            int8_t* output = (int8_t*)Y->dptr_;
            int8_t* input = (int8_t*)X->dptr_;
            int32_t scale = q_y > q_x ? 1 << (q_y - q_x) : 1;
            int32_t shift = q_x > q_y ? q_x - q_y : 0;
            API_LIB(scale_q7_int8)(input, scale, output, size, shift);
        }
    } else {
        return T_ERR_FAIL;
    }

    return 0;
}

#endif