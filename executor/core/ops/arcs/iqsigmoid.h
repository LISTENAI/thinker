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

static uint32_t iqsigmoid_workspace_bytes(uint16_t dtype, uint32_t size,
                                          int32_t x_in_psram,
                                          int32_t y_in_psram,
                                          int32_t need_i32_buf) {
    uint32_t bytes = 0;

    if (dtype == Int8) {
        bytes += x_in_psram ? ALIGN4(size) : 0;
        bytes += size * sizeof(int32_t);
    } else if (dtype == Int16) {
        bytes += x_in_psram ? ALIGN4(size * sizeof(int16_t)) : 0;
        bytes += size * sizeof(int32_t);
    } else if (dtype == Int32) {
        bytes += need_i32_buf ? size * sizeof(int32_t) : 0;
    }

    bytes = ALIGN4(bytes);
    bytes += y_in_psram ? size * sizeof(int8_t) : 0;
    return bytes;
}

static uint32_t iqsigmoid_split_size(uint16_t dtype, uint32_t input_size,
                                     uint32_t workspace_size,
                                     int32_t x_in_psram,
                                     int32_t y_in_psram,
                                     int32_t need_i32_buf) {
    uint32_t per_size = 0;
    uint32_t split_size = 0;

    if (input_size == 0) {
        return 0;
    }

    if (dtype == Int8) {
        per_size = sizeof(int32_t) + (x_in_psram ? sizeof(int8_t) : 0) +
                   (y_in_psram ? sizeof(int8_t) : 0);
    } else if (dtype == Int16) {
        per_size = sizeof(int32_t) + (x_in_psram ? sizeof(int16_t) : 0) +
                   (y_in_psram ? sizeof(int8_t) : 0);
    } else if (dtype == Int32) {
        per_size = (need_i32_buf ? sizeof(int32_t) : 0) +
                   (y_in_psram ? sizeof(int8_t) : 0);
    } else {
        return 0;
    }

    if (per_size == 0) {
        return input_size;
    }

    split_size = ALIGN4(workspace_size / per_size);
    if (split_size > input_size) {
        split_size = input_size;
    }

    while (split_size > 0 &&
           iqsigmoid_workspace_bytes(dtype, split_size, x_in_psram,
                                     y_in_psram, need_i32_buf) > workspace_size) {
        split_size = split_size - 4;
    }

    return split_size;
}

/**
 * @brief Integer Quantized Sigmoid operation
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t iqsigmoid(tTensor *X, tTensor *Y, tTensor *Temp) {
    uint32_t input_size = getTensorSize(X);
    uint32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    int32_t x_in_psram = (X->mem_.type_ != 2);
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    const int32_t Q_INPUT = 27;
    int32_t shift = Q_INPUT - (int32_t)X->scale_;

#ifdef RUNTIME_PARAM_CHECK
    if (Y->dtype_ != Int8) {
        return T_ERR_INVALID_DATATYPE;
    }
#endif

    if (input_size == 0) {
        return T_SUCCESS;
    }

    if (X->dtype_ == Int8) {
        uint32_t split_size = iqsigmoid_split_size(X->dtype_, input_size,
                                                   workspace_size,
                                                   x_in_psram, y_in_psram, 1);
        if (workspace == NULL || split_size == 0) {
            return T_ERR_NO_WORKSPACE;
        }

        for (uint32_t past_size = 0; past_size < input_size; past_size += split_size) {
            uint32_t cur_size = MIN(split_size, input_size - past_size);
            uint32_t input_bytes = x_in_psram ? ALIGN4(cur_size) : 0;
            int8_t *src_i8 = (int8_t *)X->dptr_ + past_size;
            int32_t *tmp_i32 = (int32_t *)(workspace + input_bytes);
            int8_t *dst_i8 = y_in_psram ?
                (int8_t *)(workspace + ALIGN4(input_bytes + cur_size * sizeof(int32_t))) :
                (int8_t *)Y->dptr_ + past_size;

            if (x_in_psram) {
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(workspace, src_i8, cur_size * sizeof(int8_t)), "luna_memcpy_i8o8");
                src_i8 = workspace;
            }

            uint32_t shift_l = (shift > 0) ? (uint32_t)shift : 0U;
            uint32_t shift_r = (shift > 0) ? 0U : (uint32_t)(-shift);
            THINKER_RET_CHECK(API_LIB(scale_i8i8o32)(src_i8, 1, tmp_i32, cur_size, 0), "luna_scale_i8i8o32");
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(tmp_i32, 1UL << shift_l, tmp_i32, cur_size, shift_r), "luna_scale_i32i32o32");
            THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(tmp_i32, dst_i8, cur_size), "luna_sigmoid_i32o8");
            if (y_in_psram) {
                opi_psram_cpy_out((int8_t *)Y->dptr_ + past_size, dst_i8, cur_size * sizeof(int8_t));
            }
        }
    } else if (X->dtype_ == Int32) {
        int32_t need_i32_buf = x_in_psram || (shift != 0);
        uint32_t split_size = input_size;

        if (need_i32_buf || y_in_psram) {
            split_size = iqsigmoid_split_size(X->dtype_, input_size,
                                              workspace_size,
                                              x_in_psram, y_in_psram,
                                              need_i32_buf);
            if (workspace == NULL || split_size == 0) {
                return T_ERR_NO_WORKSPACE;
            }
        }

        for (uint32_t past_size = 0; past_size < input_size; past_size += split_size) {
            uint32_t cur_size = MIN(split_size, input_size - past_size);
            uint32_t tmp_bytes = need_i32_buf ? cur_size * sizeof(int32_t) : 0;
            int32_t *src_i32 = (int32_t *)X->dptr_ + past_size;
            int32_t *tmp_i32 = need_i32_buf ? (int32_t *)workspace : NULL;
            int8_t *dst_i8 = y_in_psram ?
                (int8_t *)(workspace + ALIGN4(tmp_bytes)) :
                (int8_t *)Y->dptr_ + past_size;

            if (x_in_psram) {
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)tmp_i32, (int8_t *)src_i32, cur_size * sizeof(int32_t)), "luna_memcpy_i8o8");
                src_i32 = tmp_i32;
            }
            if (shift != 0) {
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src_i32, 1, tmp_i32, cur_size, shift), "luna_scale_i32o32");
                src_i32 = tmp_i32;
            }

            THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(src_i32, dst_i8, cur_size), "luna_sigmoid_i32o8");
            if (y_in_psram) {
                opi_psram_cpy_out((int8_t *)Y->dptr_ + past_size, dst_i8, cur_size * sizeof(int8_t));
            }
        }
    } else {
        return T_ERR_INVALID_DATATYPE;
    }

    return T_SUCCESS;
}

#endif
