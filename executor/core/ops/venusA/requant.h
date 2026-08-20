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
 * @brief Perform re-quantization operation
 * @param X Input tensor
 * @param Y Output tensor
 * @return Execution status
 */
int32_t requant_luna(tTensor* X, tTensor* Y) {
#if THINKER_PARAM_CHECK
if (X->dtype_ != Int8 ||
                    (Y->dtype_ != Int8 && Y->dtype_ != Int16 && Y->dtype_ != Int32)) {
    return (T_ERR_INVALID_DATATYPE);
}
#endif

    size_t size = getTensorSize(X);
    int8_t* input = (int8_t*)X->dptr_;

    int32_t src_bits = (X->byte_) * 8;
    int32_t dst_bits = (Y->byte_) * 8;
    int32_t q_x = (int32_t)X->scale_;
    int32_t q_y = (int32_t)Y->scale_;

#if THINKER_RUNTIME_CHECK
if (src_bits == dst_bits &&
                      (X->mem_.type_ != 2 || Y->mem_.type_ != 2)) {
    return (T_ERR_NO_SUPPORT_OP);
}
#endif

    // Handle different bit width scenarios
    if (dst_bits > src_bits) {
#if THINKER_PARAM_CHECK
if (q_y < q_x || q_y - q_x > 30) {
    return (T_ERR_INVALID_PARA);
}
#endif
        int32_t left_shift = q_y - q_x;
        if (dst_bits == 32) {
            int32_t* output = (int32_t*)Y->dptr_;
            for (size_t i = 0; i < size; ++i) {
                int64_t value = (int64_t)input[i] * ((int64_t)1 << left_shift);
                output[i] = (int32_t)SATURATE_32BITS(value);
            }
        } else if (dst_bits == 16) {
            int16_t* output = (int16_t*)Y->dptr_;
            for (size_t i = 0; i < size; ++i) {
                int64_t value = (int64_t)input[i] * ((int64_t)1 << left_shift);
                if (value > 32767) value = 32767;
                if (value < -32768) value = -32768;
                output[i] = (int16_t)value;
            }
        }
    } else if (dst_bits == src_bits) {
#if THINKER_PARAM_CHECK
if (q_y - q_x > 30 || q_x - q_y > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif
        int8_t* output = (int8_t*)Y->dptr_;
        int scale = (q_y - q_x) > 0 ? (int)(1u << (q_y - q_x)) : 1;
        int shift = (q_x - q_y) > 0 ? (q_x - q_y) : 0;
        THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(input, scale, output, size, shift), "luna_scale_i8i8o8");
    } else {
        return T_ERR_NO_SUPPORT_OP;
    }

    if (Y->mem_.type_ == 1) {
        thinker_psram_write_complete((void *)Y->dptr_, getTensorDataSize(Y));
    }

    return T_SUCCESS;
}

#endif
