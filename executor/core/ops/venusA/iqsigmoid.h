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

static uint32_t iqsigmoid_workspace_bytes(uint16_t dtype, uint32_t size,
                                           int32_t x_in_psram,
                                           int32_t y_in_psram,
                                           int32_t need_i32_buf) {
    uint32_t bytes = x_in_psram ? ALIGN4(size * (dtype == Int8 ? 1U :
                                                 dtype == Int16 ? 2U : 4U)) : 0U;
    if (dtype == Int8) {
        bytes += ALIGN4(size * sizeof(int16_t)) + size * sizeof(int32_t);
    } else if (dtype == Int16) {
        bytes += size * sizeof(int32_t);
    } else if (dtype == Int32 && need_i32_buf) {
        bytes += size * sizeof(int32_t);
    }
    bytes = ALIGN4(bytes);
    return bytes + (y_in_psram ? size : 0U);
}

static uint32_t iqsigmoid_split_size(uint16_t dtype, uint32_t input_size,
                                      uint32_t workspace_size,
                                      int32_t x_in_psram,
                                      int32_t y_in_psram,
                                      int32_t need_i32_buf) {
    if (input_size == 0) {
        return 0;
    }
    uint32_t split_size = MIN(input_size, workspace_size);
    while (split_size > 0 &&
           iqsigmoid_workspace_bytes(dtype, split_size, x_in_psram,
                                      y_in_psram, need_i32_buf) > workspace_size) {
        --split_size;
    }
    return split_size;
}

int32_t iqsigmoid(tTensor *X, tTensor *Y, tTensor *Temp) {
    uint32_t input_size = getTensorSize(X);
    uint32_t workspace_size = Temp ? (uint32_t)getTensorDataSize(Temp) : 0;
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    int32_t x_in_psram = X->mem_.type_ == 1;
    int32_t y_in_psram = Y->mem_.type_ == 1;
    int32_t shift = 27 - (int32_t)X->scale_;
    int32_t need_i32_buf = X->dtype_ != Int32 || x_in_psram || shift != 0;

    #if THINKER_PARAM_CHECK
    if ((X->dtype_ != Int8 && X->dtype_ != Int16 &&
                         X->dtype_ != Int32) || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (shift < -63 || shift > 30) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    if (input_size == 0) {
        return T_SUCCESS;
    }

    uint32_t split_size = input_size;
    if (need_i32_buf || y_in_psram) {
        split_size = iqsigmoid_split_size(X->dtype_, input_size, workspace_size,
                                          x_in_psram, y_in_psram, need_i32_buf);
        if (workspace == NULL || split_size == 0) {
            return T_ERR_NO_WORKSPACE;
        }
    }

    for (uint32_t past_size = 0; past_size < input_size; past_size += split_size) {
        uint32_t cur_size = MIN(split_size, input_size - past_size);
        uint32_t elem_bytes = X->dtype_ == Int8 ? 1U : X->dtype_ == Int16 ? 2U : 4U;
        uint32_t input_bytes = x_in_psram ? ALIGN4(cur_size * elem_bytes) : 0U;
        int8_t *src = (int8_t *)X->dptr_ + past_size * elem_bytes;
        int8_t *temp_base = workspace ? workspace + input_bytes : NULL;

        if (x_in_psram) {
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(workspace, src,
                              cur_size * elem_bytes), "luna_memcpy_i8o8");
            src = workspace;
        }

        int8_t *output_base = temp_base;
        if (X->dtype_ == Int8) {
            int16_t *temp_i16 = (int16_t *)temp_base;
            int32_t *temp_i32 = (int32_t *)(temp_base + ALIGN4(cur_size * sizeof(int16_t)));
            output_base = (int8_t *)temp_i32 + cur_size * sizeof(int32_t);
            THINKER_RET_CHECK(API_LIB(scale_i8i8o16)((int8_t *)src, 1, temp_i16,
                              cur_size, 0), "luna_scale_i8i8o16");
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(temp_i16, 1, temp_i32,
                              cur_size, 0), "luna_scale_i16i16o32");
            if (shift != 0) {
                uint32_t shift_l = shift > 0 ? (uint32_t)shift : 0U;
                uint32_t shift_r = shift > 0 ? 0U : (uint32_t)(-shift);
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(temp_i32, 1UL << shift_l,
                                  temp_i32, cur_size, shift_r), "luna_scale_i32i32o32");
            }
            int8_t *dst = y_in_psram ? output_base : (int8_t *)Y->dptr_ + past_size;
            THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(temp_i32, dst, cur_size),
                              "luna_sigmoid_i32o8");
            output_base = dst;
        } else if (X->dtype_ == Int16) {
            int32_t *temp_i32 = (int32_t *)temp_base;
            output_base = (int8_t *)temp_i32 + cur_size * sizeof(int32_t);
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)((int16_t *)src, 1, temp_i32,
                              cur_size, 0), "luna_scale_i16i16o32");
            if (shift != 0) {
                uint32_t shift_l = shift > 0 ? (uint32_t)shift : 0U;
                uint32_t shift_r = shift > 0 ? 0U : (uint32_t)(-shift);
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(temp_i32, 1UL << shift_l,
                                  temp_i32, cur_size, shift_r), "luna_scale_i32i32o32");
            }
            int8_t *dst = y_in_psram ? output_base : (int8_t *)Y->dptr_ + past_size;
            THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(temp_i32, dst, cur_size),
                              "luna_sigmoid_i32o8");
            output_base = dst;
        } else {
            int32_t *src_i32 = (int32_t *)src;
            int32_t *temp_i32 = need_i32_buf ? (int32_t *)temp_base : NULL;
            output_base = need_i32_buf ? temp_base + cur_size * sizeof(int32_t) : temp_base;
            if (shift != 0) {
                uint32_t shift_l = shift > 0 ? (uint32_t)shift : 0U;
                uint32_t shift_r = shift > 0 ? 0U : (uint32_t)(-shift);
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src_i32, 1UL << shift_l,
                                  temp_i32, cur_size, shift_r), "luna_scale_i32i32o32");
                src_i32 = temp_i32;
            }
            int8_t *dst = y_in_psram ? output_base : (int8_t *)Y->dptr_ + past_size;
            THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(src_i32, dst, cur_size),
                              "luna_sigmoid_i32o8");
            output_base = dst;
        }

        if (y_in_psram) {
            opi_psram_cpy_out((int8_t *)Y->dptr_ + past_size, output_base, cur_size);
        }
    }
    return T_SUCCESS;
}

#endif
