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
    #if THINKER_PARAM_CHECK
    if (X->dtype_ != Int8 ||
                        (Y->dtype_ != Int8 && Y->dtype_ != Int16 && Y->dtype_ != Int32)) {
        return (T_ERR_INVALID_DATATYPE);
    }
    #endif

    size_t size = getTensorSize(X);
    int8_t* input = (int8_t*)X->dptr_;
    int32_t src_bits = X->byte_ * 8;
    int32_t dst_bits = Y->byte_ * 8;
    int32_t q_x = (int32_t)X->scale_;
    int32_t q_y = (int32_t)Y->scale_;
    int32_t q_delta = q_y - q_x;

    #if THINKER_PARAM_CHECK
    if (dst_bits == src_bits &&
                        (X->mem_.type_ != 2 || Y->mem_.type_ != 2)) {
        return (T_ERR_INVALID_PARA);
    }
    #endif

    if (dst_bits > src_bits) {
        #if THINKER_PARAM_CHECK
        if (q_delta < 0 || q_delta > (dst_bits == 16 ? 14 : 30)) {
            return (T_ERR_INVALID_PARA);
        }
        #endif
        if (dst_bits == 32) {
            int32_t* output = (int32_t*)Y->dptr_;
            for (int32_t i = 0; i < size; ++i) {
                int64_t value = (int64_t)input[i] * ((int64_t)1 << q_delta);
                output[i] = (int32_t)SATURATE_32BITS(value);
            }
        } else if (dst_bits == 16) {
            int16_t* output = (int16_t*)Y->dptr_;
            for (int32_t i = 0; i < size; ++i) {
                int64_t value = (int64_t)input[i] * ((int64_t)1 << q_delta);
                if (value > 32767) value = 32767;
                if (value < -32768) value = -32768;
                output[i] = (int16_t)value;
            }
        }
    } else if (dst_bits == src_bits) {
        #if THINKER_PARAM_CHECK
        if (q_delta < -63 || q_delta > (dst_bits == 8 ? 6 : (dst_bits == 16 ? 14 : 30))) {
            return (T_ERR_INVALID_PARA);
        }
        #endif
        if (dst_bits == 32) {
            int32_t* output = (int32_t*)Y->dptr_;
            int32_t* input = (int32_t*)X->dptr_;
            int32_t scale = q_y > q_x ? 1 << (q_y - q_x) : 1;
            int32_t shift = q_x > q_y ? q_x - q_y : 0;
            THINKER_RET_CHECK(API_LIB(scale_q31_int32)(input, scale, output, size, shift), "scale_q31_int32");
        } else if (dst_bits == 16) {
            int16_t* output = (int16_t*)Y->dptr_;
            int16_t* input = (int16_t*)X->dptr_;
            int32_t scale = q_y > q_x ? 1 << (q_y - q_x) : 1;
            int32_t shift = q_x > q_y ? q_x - q_y : 0;
            THINKER_RET_CHECK(API_LIB(scale_q15_int16)(input, scale, output, size, shift), "scale_q15_int16");
        } else if (dst_bits == 8) {
            int8_t* output = (int8_t*)Y->dptr_;
            int8_t* input = (int8_t*)X->dptr_;
            int32_t scale = q_y > q_x ? 1 << (q_y - q_x) : 1;
            int32_t shift = q_x > q_y ? q_x - q_y : 0;
            THINKER_RET_CHECK(API_LIB(scale_q7_int8)(input, scale, output, size, shift), "scale_q7_int8");
        }
    } else {
        return T_ERR_NO_SUPPORT_OP;
    }

    return T_SUCCESS;
}

#endif
